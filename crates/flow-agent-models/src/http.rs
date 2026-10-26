//! Shared HTTP plumbing for providers: client setup, error mapping and
//! Server-Sent Events parsing.

use std::time::Duration;

use flow_agent_core::{Error, Result};
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::registry::ProviderConfig;

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Build (or reuse) the HTTP client for a provider.
pub(crate) fn client(config: &ProviderConfig) -> Result<reqwest::Client> {
    if let Some(c) = &config.http_client {
        return Ok(c.clone());
    }
    let timeout = config.request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);
    reqwest::Client::builder()
        .connect_timeout(timeout.min(Duration::from_secs(30)))
        .read_timeout(timeout)
        .user_agent(concat!("flow-agent/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| Error::Config(format!("failed to build HTTP client: {e}")))
}

/// Turn configured extra headers into a header map.
pub(crate) fn headers(pairs: &[(String, String)]) -> Result<HeaderMap> {
    let mut map = HeaderMap::new();
    for (k, v) in pairs {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| Error::Config(format!("invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v)
            .map_err(|e| Error::Config(format!("invalid value for header '{k}': {e}")))?;
        map.insert(name, value);
    }
    Ok(map)
}

/// Mark a header as sensitive (never logged by reqwest/hyper).
pub(crate) fn secret(value: &str) -> Result<HeaderValue> {
    let mut v = HeaderValue::from_str(value)
        .map_err(|_| Error::Config("API key contains invalid characters".into()))?;
    v.set_sensitive(true);
    Ok(v)
}

fn transport_error(provider: &str, e: reqwest::Error) -> Error {
    if e.is_timeout() {
        Error::Model {
            provider: provider.into(),
            message: format!("request timed out: {e}"),
            status: None,
            retryable: true,
        }
    } else {
        Error::Http(format!("{provider}: {e}"))
    }
}

async fn send(provider: &str, request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    let resp = request
        .send()
        .await
        .map_err(|e| transport_error(provider, e))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .or_else(|| v.get("error"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.chars().take(300).collect());
    let code = status.as_u16();
    Err(Error::Model {
        provider: provider.into(),
        message: format!("HTTP {code}: {detail}"),
        status: Some(code),
        retryable: matches!(code, 408 | 409 | 425 | 429) || status.is_server_error(),
    })
}

/// POST JSON and parse a JSON response.
pub(crate) async fn post_json(
    provider: &str,
    request: reqwest::RequestBuilder,
    body: &Value,
) -> Result<Value> {
    let resp = send(provider, request.json(body)).await?;
    resp.json().await.map_err(|e| transport_error(provider, e))
}

/// One Server-Sent Event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// POST JSON and return the response as a stream of SSE events.
pub(crate) async fn post_sse(
    provider: &str,
    request: reqwest::RequestBuilder,
    body: &Value,
) -> Result<BoxStream<'static, Result<SseEvent>>> {
    let resp = send(
        provider,
        request
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(body),
    )
    .await?;
    let provider = provider.to_owned();
    let bytes = resp
        .bytes_stream()
        .map(move |r| r.map_err(|e| transport_error(&provider, e)));
    Ok(sse_events(bytes).boxed())
}

/// Parse a byte stream into SSE events (handles `\n`, `\r\n`, comments and
/// multi-line `data:`).
pub(crate) fn sse_events<S, B>(bytes: S) -> impl futures::Stream<Item = Result<SseEvent>> + Send
where
    S: futures::Stream<Item = Result<B>> + Send + Unpin + 'static,
    B: AsRef<[u8]> + Send,
{
    struct St<S> {
        inner: S,
        buf: Vec<u8>,
        done: bool,
    }
    futures::stream::unfold(
        St {
            inner: bytes,
            buf: Vec::new(),
            done: false,
        },
        |mut st| async move {
            loop {
                if let Some(ev) = take_event(&mut st.buf, st.done) {
                    return Some((Ok(ev), st));
                }
                if st.done {
                    return None;
                }
                match st.inner.next().await {
                    Some(Ok(chunk)) => st.buf.extend_from_slice(chunk.as_ref()),
                    Some(Err(e)) => {
                        st.done = true;
                        st.buf.clear();
                        return Some((Err(e), st));
                    }
                    None => st.done = true,
                }
            }
        },
    )
}

/// Remove and parse the next complete event from `buf`. At end of input, a
/// trailing event without a blank line is still returned.
fn take_event(buf: &mut Vec<u8>, eof: bool) -> Option<SseEvent> {
    loop {
        let (block_end, sep_len) = match find_blank_line(buf) {
            Some(x) => x,
            None if eof && !buf.is_empty() => (buf.len(), 0),
            None => return None,
        };
        let block: Vec<u8> = buf.drain(..block_end + sep_len).take(block_end).collect();
        let text = String::from_utf8_lossy(&block);
        let mut ev = SseEvent::default();
        let mut has_data = false;
        for line in text.lines() {
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            let (field, value) = match line.split_once(':') {
                Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
                None => (line, ""),
            };
            match field {
                "event" => ev.event = Some(value.to_owned()),
                "data" => {
                    if has_data {
                        ev.data.push('\n');
                    }
                    ev.data.push_str(value);
                    has_data = true;
                }
                _ => {}
            }
        }
        if has_data || ev.event.is_some() {
            return Some(ev);
        }
        if buf.is_empty() {
            return None;
        }
    }
}

fn find_blank_line(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        if buf[i..].starts_with(b"\r\n\r\n") {
            return Some((i, 4));
        }
        if buf[i..].starts_with(b"\n\n") {
            return Some((i, 2));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn parse(chunks: Vec<&'static str>) -> Vec<SseEvent> {
        let s = futures::stream::iter(chunks.into_iter().map(Ok::<_, Error>));
        sse_events(s).map(|r| r.unwrap()).collect().await
    }

    #[tokio::test]
    async fn parses_split_events() {
        let evs = parse(vec![
            ": keepalive\n\nevent: a\nda",
            "ta: {\"x\":1}\n\n",
            "data: line1\r\ndata: line2\r\n\r\n",
            "data: [DONE]",
        ])
        .await;
        assert_eq!(evs.len(), 3);
        assert_eq!(evs[0].event.as_deref(), Some("a"));
        assert_eq!(evs[0].data, "{\"x\":1}");
        assert_eq!(evs[1].data, "line1\nline2");
        assert_eq!(evs[2].data, "[DONE]");
    }

    #[test]
    fn headers_validate() {
        assert!(headers(&[("x-ok".into(), "v".into())]).is_ok());
        assert!(headers(&[("bad header".into(), "v".into())]).is_err());
        assert!(secret("abc").unwrap().is_sensitive());
    }
}
