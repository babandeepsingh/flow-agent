//! OpenAI Chat Completions API, also used for OpenAI-compatible servers
//! (vLLM, Ollama, LM Studio, gateways) via `base_url`.

use async_trait::async_trait;
use flow_agent_core::{Content, Error, Message, Result, Role, ToolCall, Usage};
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::http;
use crate::provider::{
    ChatChunk, ChatProvider, ChatRequest, ChunkStream, FinishReason, ToolCallChunk, ToolChoice,
};
use crate::registry::ProviderConfig;

const PROVIDER: &str = "openai";
/// Default API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI (and compatible) provider. Usually created through
/// `ChatModel::builder("openai:<model>")`.
#[derive(Clone, Debug)]
pub struct OpenAiProvider {
    model: String,
    base_url: String,
    official: bool,
    client: reqwest::Client,
    headers: reqwest::header::HeaderMap,
}

impl OpenAiProvider {
    /// Create the provider. The API key is required for the official API and
    /// optional when `base_url` points elsewhere (local servers).
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
            .trim_end_matches('/')
            .to_owned();
        let official = base_url == DEFAULT_BASE_URL;
        let mut headers = http::headers(&config.headers)?;
        match &config.api_key {
            Some(key) => {
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    http::secret(&format!("Bearer {key}"))?,
                );
            }
            None if official => {
                return Err(Error::Config(
                    "OpenAI requires an API key: ChatModel::builder(..).api_key(key)".into(),
                ));
            }
            None => {}
        }
        Ok(OpenAiProvider {
            model: config.model.clone(),
            client: http::client(&config)?,
            base_url,
            official,
            headers,
        })
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// The JSON body for a request.
    pub fn build_body(&self, req: &ChatRequest, stream: bool) -> Value {
        let mut body = Map::new();
        body.insert("model".into(), json!(self.model));
        body.insert(
            "messages".into(),
            Value::Array(req.messages.iter().map(message_to_wire).collect()),
        );
        if !req.tools.is_empty() {
            body.insert(
                "tools".into(),
                Value::Array(
                    req.tools
                        .iter()
                        .map(|t| {
                            json!({
                                "type": "function",
                                "function": {
                                    "name": t.name,
                                    "description": t.description,
                                    "parameters": t.parameters,
                                }
                            })
                        })
                        .collect(),
                ),
            );
            body.insert(
                "tool_choice".into(),
                match &req.tool_choice {
                    ToolChoice::Auto => json!("auto"),
                    ToolChoice::None => json!("none"),
                    ToolChoice::Required => json!("required"),
                    ToolChoice::Tool(name) => {
                        json!({"type": "function", "function": {"name": name}})
                    }
                },
            );
        }
        let o = &req.options;
        if let Some(t) = o.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(p) = o.top_p {
            body.insert("top_p".into(), json!(p));
        }
        if let Some(n) = o.max_tokens {
            // The official API deprecated `max_tokens` in favour of
            // `max_completion_tokens`; compatible servers mostly know only the former.
            let key = if self.official {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            body.insert(key.into(), json!(n));
        }
        if !o.stop.is_empty() {
            body.insert("stop".into(), json!(o.stop));
        }
        if stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({"include_usage": true}));
        }
        for (k, v) in &o.extra {
            body.insert(k.clone(), v.clone());
        }
        Value::Object(body)
    }
}

fn message_to_wire(m: &Message) -> Value {
    let role = m.role.as_str();
    let content = match &m.content {
        Content::Text(t) => json!(t),
        Content::Parts(parts) => Value::Array(
            parts
                .iter()
                .map(|p| serde_json::to_value(p).unwrap_or(Value::Null))
                .collect(),
        ),
    };
    let mut v = json!({ "role": role, "content": content });
    if let Some(name) = &m.name {
        if m.role != Role::Tool {
            v["name"] = json!(name);
        }
    }
    match m.role {
        Role::Tool => {
            v["tool_call_id"] = json!(m.tool_call_id.clone().unwrap_or_default());
        }
        Role::Assistant if m.has_tool_calls() => {
            if m.text().is_empty() {
                v["content"] = Value::Null;
            }
            v["tool_calls"] = Value::Array(
                m.tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": { "name": tc.name, "arguments": tc.args.to_string() }
                        })
                    })
                    .collect(),
            );
        }
        _ => {}
    }
    v
}

fn finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

fn usage(v: &Value) -> Option<Usage> {
    let u = v.get("usage")?;
    if u.is_null() {
        return None;
    }
    Some(Usage::new(
        u["prompt_tokens"].as_u64().unwrap_or(0),
        u["completion_tokens"].as_u64().unwrap_or(0),
    ))
}

fn bad_response(message: impl Into<String>) -> Error {
    Error::Model {
        provider: PROVIDER.into(),
        message: message.into(),
        status: None,
        retryable: false,
    }
}

/// Parse a non-streaming response.
pub fn parse_response(v: &Value) -> Result<Message> {
    let choice = v
        .pointer("/choices/0")
        .ok_or_else(|| bad_response("response has no choices"))?;
    let m = &choice["message"];
    let mut msg = Message::assistant(m["content"].as_str().unwrap_or_default());
    if let Some(calls) = m["tool_calls"].as_array() {
        for c in calls {
            let name = c
                .pointer("/function/name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_response("tool call without a name"))?;
            let raw = c
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let args = if raw.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(raw).map_err(|e| {
                    bad_response(format!(
                        "tool call '{name}' has invalid JSON arguments: {e}"
                    ))
                })?
            };
            let mut tc = ToolCall::new(name, args);
            if let Some(id) = c["id"].as_str() {
                tc.id = id.to_owned();
            }
            msg.tool_calls.push(tc);
        }
    }
    msg.usage = usage(v);
    if let Some(fr) = choice["finish_reason"].as_str() {
        msg.metadata
            .insert("finish_reason".into(), json!(finish_reason(fr).as_str()));
    }
    if let Some(id) = v["id"].as_str() {
        msg.metadata.insert("provider_id".into(), json!(id));
    }
    Ok(msg)
}

/// Parse one streaming `data:` payload.
pub fn parse_stream_chunk(v: &Value) -> Result<ChatChunk> {
    if let Some(err) = v.get("error") {
        return Err(bad_response(format!(
            "stream error: {}",
            err.get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        )));
    }
    let mut chunk = ChatChunk {
        usage: usage(v),
        ..Default::default()
    };
    if let Some(choice) = v.pointer("/choices/0") {
        let delta = &choice["delta"];
        chunk.text = delta["content"].as_str().map(str::to_owned);
        if let Some(calls) = delta["tool_calls"].as_array() {
            for (i, c) in calls.iter().enumerate() {
                chunk.tool_call_chunks.push(ToolCallChunk {
                    index: c["index"].as_u64().map(|n| n as usize).unwrap_or(i),
                    id: c["id"].as_str().map(str::to_owned),
                    name: c
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    args_delta: c
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
        }
        chunk.finish_reason = choice["finish_reason"].as_str().map(finish_reason);
    }
    Ok(chunk)
}

#[async_trait]
impl ChatProvider for OpenAiProvider {
    fn provider(&self) -> &str {
        PROVIDER
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn generate(&self, request: ChatRequest) -> Result<Message> {
        let body = self.build_body(&request, false);
        let req = self.client.post(self.url()).headers(self.headers.clone());
        let v = http::post_json(PROVIDER, req, &body).await?;
        parse_response(&v)
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChunkStream> {
        let body = self.build_body(&request, true);
        let req = self.client.post(self.url()).headers(self.headers.clone());
        let events = http::post_sse(PROVIDER, req, &body).await?;
        let chunks = events
            .take_while(|e| {
                let done = matches!(e, Ok(ev) if ev.data.trim() == "[DONE]");
                futures::future::ready(!done)
            })
            .filter_map(|e| async move {
                match e {
                    Err(e) => Some(Err(e)),
                    Ok(ev) if ev.data.trim().is_empty() => None,
                    Ok(ev) => Some(
                        serde_json::from_str::<Value>(&ev.data)
                            .map_err(|e| bad_response(format!("invalid stream JSON: {e}")))
                            .and_then(|v| parse_stream_chunk(&v)),
                    ),
                }
            });
        Ok(chunks.boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flow_agent_core::ToolDefinition;

    fn provider(base_url: Option<&str>) -> OpenAiProvider {
        let mut cfg = ProviderConfig::new("gpt-test");
        cfg.api_key = Some("sk-test".into());
        cfg.base_url = base_url.map(str::to_owned);
        OpenAiProvider::new(cfg).unwrap()
    }

    #[test]
    fn key_required_only_for_official_api() {
        assert!(OpenAiProvider::new(ProviderConfig::new("m")).is_err());
        let mut cfg = ProviderConfig::new("m");
        cfg.base_url = Some("http://localhost:11434/v1/".into());
        let p = OpenAiProvider::new(cfg).unwrap();
        assert_eq!(p.url(), "http://localhost:11434/v1/chat/completions");
    }

    #[test]
    fn body_shape() {
        let p = provider(None);
        let req = ChatRequest {
            messages: vec![
                Message::system("sys"),
                Message::user("hi"),
                Message::assistant("").with_tool_call("t", json!({"a": 1})),
                Message::tool("call_1", json!({"ok": true})),
            ],
            tools: vec![ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: json!({"type": "object"}),
            }],
            tool_choice: ToolChoice::Tool("t".into()),
            options: crate::ModelOptions {
                max_tokens: Some(10),
                temperature: Some(0.5),
                ..Default::default()
            },
        };
        let b = p.build_body(&req, true);
        assert_eq!(b["model"], "gpt-test");
        assert_eq!(b["messages"][2]["content"], Value::Null);
        assert_eq!(
            b["messages"][2]["tool_calls"][0]["function"]["arguments"],
            r#"{"a":1}"#
        );
        assert_eq!(b["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(b["tools"][0]["function"]["name"], "t");
        assert_eq!(b["tool_choice"]["function"]["name"], "t");
        assert_eq!(b["max_completion_tokens"], 10);
        assert_eq!(b["stream"], true);

        let local = provider(Some("http://localhost:8000/v1"));
        assert_eq!(local.build_body(&req, false)["max_tokens"], 10);
    }

    #[test]
    fn parse_full_response() {
        let v = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "w", "arguments": "{\"city\":\"Paris\"}"}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let m = parse_response(&v).unwrap();
        assert_eq!(m.tool_calls[0].id, "call_9");
        assert_eq!(m.tool_calls[0].args["city"], "Paris");
        assert_eq!(m.usage, Some(Usage::new(10, 5)));
        assert_eq!(m.metadata["finish_reason"], "tool_calls");
        assert!(parse_response(&json!({})).is_err());
    }

    #[test]
    fn parse_chunks() {
        let c = parse_stream_chunk(&json!({"choices": [{"delta": {"content": "Hi"}}]})).unwrap();
        assert_eq!(c.text.as_deref(), Some("Hi"));
        let c = parse_stream_chunk(&json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c", "function": {"name": "t", "arguments": "{\""}}]}, "finish_reason": null}]})).unwrap();
        assert_eq!(c.tool_call_chunks[0].index, 1);
        assert_eq!(c.tool_call_chunks[0].args_delta, "{\"");
        let c = parse_stream_chunk(
            &json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
        )
        .unwrap();
        assert_eq!(c.usage, Some(Usage::new(1, 2)));
        assert!(parse_stream_chunk(&json!({"error": {"message": "x"}})).is_err());
    }

    #[test]
    fn multipart_content_passes_through() {
        let m: Message = serde_json::from_value(json!({
            "role": "user",
            "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
        }))
        .unwrap();
        let w = message_to_wire(&m);
        assert_eq!(w["content"][1]["type"], "image_url");
    }
}
