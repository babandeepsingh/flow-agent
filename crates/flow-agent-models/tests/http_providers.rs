//! OpenAI and Anthropic providers end-to-end against a local mock HTTP server
//! (no real API calls).
#![cfg(all(feature = "openai", feature = "anthropic"))]

use std::sync::{Arc, Mutex};

use flow_agent_core::prelude::*;
use flow_agent_models::ChatModel;
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Clone, Debug)]
struct Recorded {
    head: String,
    body: Value,
}

/// Serve `responses` in order (status, content type, body), one per connection.
async fn mock_server(
    responses: Vec<(u16, &'static str, String)>,
) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    tokio::spawn(async move {
        for (status, ctype, body) in responses {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let (head, body_start) = loop {
                let n = sock.read(&mut tmp).await.unwrap();
                buf.extend_from_slice(&tmp[..n]);
                if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break (String::from_utf8_lossy(&buf[..i]).to_string(), i + 4);
                }
            };
            let len: usize = head
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse().ok())?
                })
                .unwrap_or(0);
            while buf.len() < body_start + len {
                let n = sock.read(&mut tmp).await.unwrap();
                buf.extend_from_slice(&tmp[..n]);
            }
            let req_body: Value =
                serde_json::from_slice(&buf[body_start..body_start + len]).unwrap_or(Value::Null);
            seen2.lock().unwrap().push(Recorded {
                head,
                body: req_body,
            });
            let resp = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}"), seen)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn sse(events: &[(Option<&str>, Value)]) -> String {
    let mut s = String::new();
    for (name, data) in events {
        if let Some(n) = name {
            s.push_str(&format!("event: {n}\n"));
        }
        s.push_str(&format!("data: {data}\n\n"));
    }
    s
}

#[tokio::test]
async fn openai_invoke_and_stream() {
    let full = json!({
        "id": "x",
        "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}
    })
    .to_string();
    let mut stream = sse(&[
        (None, json!({"choices": [{"delta": {"content": "Hel"}}]})),
        (None, json!({"choices": [{"delta": {"content": "lo"}}]})),
        (
            None,
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "t", "arguments": "{\"a\""}}]}}]}),
        ),
        (
            None,
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}, "finish_reason": "tool_calls"}]}),
        ),
        (
            None,
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
        ),
    ]);
    stream.push_str("data: [DONE]\n\n");
    let (url, seen) = mock_server(vec![
        (200, "application/json", full),
        (200, "text/event-stream", stream),
    ])
    .await;

    let model = ChatModel::builder("openai:gpt-test")
        .base_url(format!("{url}/v1"))
        .api_key("sk-test")
        .http_client(client())
        .max_tokens(50)
        .build()
        .unwrap();

    let out = model.invoke(json!("hello")).await.unwrap();
    assert_eq!(out["content"], "Hi!");
    assert_eq!(out["usage"]["input_tokens"], 3);

    let events: Vec<Event> = model
        .stream(json!("hello"))
        .map(|e| e.unwrap())
        .collect()
        .await;
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::Token { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");
    let Some(Event::Output { value }) = events.last() else {
        panic!("no output")
    };
    assert_eq!(value["tool_calls"][0]["args"], json!({"a": 1}));
    assert_eq!(value["usage"]["output_tokens"], 2);
    assert_eq!(value["metadata"]["finish_reason"], "tool_calls");

    let seen = seen.lock().unwrap().clone();
    assert!(seen[0].head.starts_with("POST /v1/chat/completions"));
    assert!(
        seen[0]
            .head
            .to_lowercase()
            .contains("authorization: bearer sk-test")
    );
    assert_eq!(seen[0].body["model"], "gpt-test");
    assert_eq!(seen[0].body["max_tokens"], 50);
    assert_eq!(seen[1].body["stream"], true);
}

#[tokio::test]
async fn anthropic_invoke_and_stream() {
    let full = json!({
        "id": "msg_1",
        "content": [{"type": "text", "text": "Bonjour"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 2}
    })
    .to_string();
    let stream = sse(&[
        (
            Some("message_start"),
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 8}}}),
        ),
        (Some("ping"), json!({"type": "ping"})),
        (
            Some("content_block_start"),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            Some("content_block_delta"),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sa"}}),
        ),
        (
            Some("content_block_delta"),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lut"}}),
        ),
        (
            Some("content_block_stop"),
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            Some("message_delta"),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        ),
        (Some("message_stop"), json!({"type": "message_stop"})),
    ]);
    let (url, seen) = mock_server(vec![
        (200, "application/json", full),
        (200, "text/event-stream", stream),
    ])
    .await;

    let model = ChatModel::builder("anthropic:claude-test")
        .base_url(url)
        .api_key("sk-ant-test")
        .http_client(client())
        .build()
        .unwrap();

    let input = json!([["system", "Be French."], ["user", "Hi"]]);
    let out = model.invoke(input.clone()).await.unwrap();
    assert_eq!(out["content"], "Bonjour");

    let value = flow_agent_core::runnable::collect_output(model.stream(input))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(value["content"], "Salut");
    assert_eq!(
        value["usage"],
        json!({"input_tokens": 8, "output_tokens": 3, "total_tokens": 11})
    );

    let seen = seen.lock().unwrap().clone();
    let head = seen[0].head.to_lowercase();
    assert!(seen[0].head.starts_with("POST /v1/messages"));
    assert!(head.contains("x-api-key: sk-ant-test"));
    assert!(head.contains("anthropic-version: 2023-06-01"));
    assert_eq!(seen[0].body["system"], "Be French.");
    assert_eq!(seen[0].body["messages"][0]["role"], "user");
}

#[tokio::test(start_paused = true)]
async fn http_errors_map_to_model_errors_and_retry() {
    let (url, seen) = mock_server(vec![
        (
            429,
            "application/json",
            json!({"error": {"message": "slow down"}}).to_string(),
        ),
        (
            200,
            "application/json",
            json!({"content": [{"type": "text", "text": "ok"}]}).to_string(),
        ),
        (
            401,
            "application/json",
            json!({"error": {"type": "authentication_error", "message": "bad key"}}).to_string(),
        ),
    ])
    .await;
    let model = ChatModel::builder("anthropic:claude-test")
        .base_url(url)
        .api_key("k")
        .http_client(client())
        .build()
        .unwrap();

    // 429 is retried, then succeeds.
    assert_eq!(model.invoke(json!("x")).await.unwrap()["content"], "ok");
    assert_eq!(seen.lock().unwrap().len(), 2);

    // 401 is not retried.
    match model.invoke(json!("x")).await.unwrap_err() {
        Error::Model {
            status,
            message,
            retryable,
            ..
        } => {
            assert_eq!(status, Some(401));
            assert!(message.contains("bad key"));
            assert!(!retryable);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(seen.lock().unwrap().len(), 3);
}
