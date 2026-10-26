//! Anthropic Messages API (Claude).

use std::collections::HashMap;

use async_trait::async_trait;
use flow_agent_core::{Content, ContentPart, Error, Message, Result, Role, ToolCall, Usage};
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::http;
use crate::provider::{
    ChatChunk, ChatProvider, ChatRequest, ChunkStream, FinishReason, ToolCallChunk, ToolChoice,
};
use crate::registry::ProviderConfig;

const PROVIDER: &str = "anthropic";
/// Default API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// `anthropic-version` header sent with every request.
pub const API_VERSION: &str = "2023-06-01";
/// `max_tokens` used when none is set (the API requires one).
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Anthropic provider. Usually created through
/// `ChatModel::builder("anthropic:<model>")`.
#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    model: String,
    base_url: String,
    client: reqwest::Client,
    headers: reqwest::header::HeaderMap,
}

impl AnthropicProvider {
    /// Create the provider. An API key is required unless a custom `base_url`
    /// (e.g. an authenticating gateway) is used.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
            .trim_end_matches('/')
            .to_owned();
        let mut headers = http::headers(&config.headers)?;
        headers.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static(API_VERSION),
        );
        match &config.api_key {
            Some(key) => {
                headers.insert("x-api-key", http::secret(key)?);
            }
            None if base_url == DEFAULT_BASE_URL => {
                return Err(Error::Config(
                    "Anthropic requires an API key: ChatModel::builder(..).api_key(key)".into(),
                ));
            }
            None => {}
        }
        Ok(AnthropicProvider {
            model: config.model.clone(),
            client: http::client(&config)?,
            base_url,
            headers,
        })
    }

    fn url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// The JSON body for a request.
    pub fn build_body(&self, req: &ChatRequest, stream: bool) -> Value {
        let mut body = Map::new();
        body.insert("model".into(), json!(self.model));
        let o = &req.options;
        body.insert(
            "max_tokens".into(),
            json!(o.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        );

        let system: Vec<String> = req
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(Message::text)
            .collect();
        if !system.is_empty() {
            body.insert("system".into(), json!(system.join("\n\n")));
        }
        body.insert(
            "messages".into(),
            Value::Array(messages_to_wire(&req.messages)),
        );

        if !req.tools.is_empty() {
            body.insert(
                "tools".into(),
                Value::Array(
                    req.tools
                        .iter()
                        .map(|t| {
                            json!({
                                "name": t.name,
                                "description": t.description,
                                "input_schema": t.parameters,
                            })
                        })
                        .collect(),
                ),
            );
            body.insert(
                "tool_choice".into(),
                match &req.tool_choice {
                    ToolChoice::Auto => json!({"type": "auto"}),
                    ToolChoice::None => json!({"type": "none"}),
                    ToolChoice::Required => json!({"type": "any"}),
                    ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
                },
            );
        }
        if let Some(t) = o.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(p) = o.top_p {
            body.insert("top_p".into(), json!(p));
        }
        if !o.stop.is_empty() {
            body.insert("stop_sequences".into(), json!(o.stop));
        }
        if stream {
            body.insert("stream".into(), json!(true));
        }
        for (k, v) in &o.extra {
            body.insert(k.clone(), v.clone());
        }
        Value::Object(body)
    }
}

fn part_to_block(p: &ContentPart) -> Value {
    match p {
        ContentPart::Text { text } => json!({"type": "text", "text": text}),
        ContentPart::ImageUrl { image_url } => {
            // data:<media_type>;base64,<data>
            let parsed = image_url
                .url
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(";base64,"));
            match parsed {
                Some((media_type, data)) => json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data}
                }),
                None => json!({"type": "image", "source": {"type": "url", "url": image_url.url}}),
            }
        }
        _ => json!({"type": "text", "text": ""}),
    }
}

fn content_blocks(m: &Message) -> Vec<Value> {
    match &m.content {
        Content::Text(t) if t.is_empty() => vec![],
        Content::Text(t) => vec![json!({"type": "text", "text": t})],
        Content::Parts(parts) => parts.iter().map(part_to_block).collect(),
    }
}

/// Convert messages to Anthropic's format: system messages are removed (sent
/// separately), tool results become `tool_result` blocks in a user turn, and
/// consecutive turns with the same role are merged (the API requires
/// alternation).
fn messages_to_wire(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    for m in messages {
        let (role, blocks) = match m.role {
            Role::System => continue,
            Role::User => ("user", content_blocks(m)),
            Role::Assistant => {
                let mut blocks = content_blocks(m);
                for tc in &m.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": if tc.args.is_object() { tc.args.clone() } else { json!({}) },
                    }));
                }
                ("assistant", blocks)
            }
            Role::Tool => {
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.text(),
                });
                if m.metadata.get("is_error") == Some(&Value::Bool(true)) {
                    block["is_error"] = json!(true);
                }
                ("user", vec![block])
            }
        };
        if blocks.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((r, existing)) if r == role => existing.extend(blocks),
            _ => out.push((role.to_owned(), blocks)),
        }
    }
    out.into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect()
}

fn stop_reason(s: &str) -> FinishReason {
    match s {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
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
    let blocks = v["content"]
        .as_array()
        .ok_or_else(|| bad_response("response has no content"))?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => text.push_str(b["text"].as_str().unwrap_or_default()),
            Some("tool_use") => {
                let name = b["name"]
                    .as_str()
                    .ok_or_else(|| bad_response("tool_use block without a name"))?;
                let mut tc = ToolCall::new(name, b["input"].clone());
                if let Some(id) = b["id"].as_str() {
                    tc.id = id.to_owned();
                }
                calls.push(tc);
            }
            _ => {}
        }
    }
    let mut msg = Message::assistant(text);
    msg.tool_calls = calls;
    if let Some(u) = v.get("usage") {
        msg.usage = Some(Usage::new(
            u["input_tokens"].as_u64().unwrap_or(0),
            u["output_tokens"].as_u64().unwrap_or(0),
        ));
    }
    if let Some(sr) = v["stop_reason"].as_str() {
        msg.metadata
            .insert("finish_reason".into(), json!(stop_reason(sr).as_str()));
    }
    if let Some(id) = v["id"].as_str() {
        msg.metadata.insert("provider_id".into(), json!(id));
    }
    Ok(msg)
}

/// Tracks state across streaming events (block → tool index, input tokens).
#[derive(Default)]
pub struct StreamState {
    input_tokens: u64,
    block_to_tool: HashMap<u64, usize>,
    next_tool: usize,
}

impl StreamState {
    /// Handle one event; returns a chunk when the event carries content.
    pub fn handle(&mut self, event: &Value) -> Result<Option<ChatChunk>> {
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                self.input_tokens = event
                    .pointer("/message/usage/input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                Ok(None)
            }
            "content_block_start" => {
                let block = &event["content_block"];
                if block["type"] != "tool_use" {
                    return Ok(None);
                }
                let index = self.next_tool;
                self.next_tool += 1;
                self.block_to_tool
                    .insert(event["index"].as_u64().unwrap_or(0), index);
                Ok(Some(ChatChunk {
                    tool_call_chunks: vec![ToolCallChunk {
                        index,
                        id: block["id"].as_str().map(str::to_owned),
                        name: block["name"].as_str().map(str::to_owned),
                        args_delta: String::new(),
                    }],
                    ..Default::default()
                }))
            }
            "content_block_delta" => {
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => Ok(Some(ChatChunk {
                        text: delta["text"].as_str().map(str::to_owned),
                        ..Default::default()
                    })),
                    Some("input_json_delta") => {
                        let block = event["index"].as_u64().unwrap_or(0);
                        let index = *self
                            .block_to_tool
                            .get(&block)
                            .ok_or_else(|| bad_response("input_json_delta for unknown block"))?;
                        Ok(Some(ChatChunk {
                            tool_call_chunks: vec![ToolCallChunk {
                                index,
                                args_delta: delta["partial_json"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }))
                    }
                    _ => Ok(None),
                }
            }
            "message_delta" => Ok(Some(ChatChunk {
                usage: event
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                    .map(|out| Usage::new(self.input_tokens, out)),
                finish_reason: event
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(stop_reason),
                ..Default::default()
            })),
            "error" => {
                let kind = event
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("error");
                Err(Error::Model {
                    provider: PROVIDER.into(),
                    message: format!(
                        "stream {kind}: {}",
                        event
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                    ),
                    status: None,
                    retryable: matches!(
                        kind,
                        "overloaded_error" | "api_error" | "rate_limit_error"
                    ),
                })
            }
            _ => Ok(None),
        }
    }
}

#[async_trait]
impl ChatProvider for AnthropicProvider {
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
        let mut state = StreamState::default();
        let chunks = events
            .map(move |e| {
                let ev = e?;
                if ev.data.trim().is_empty() {
                    return Ok(None);
                }
                let v: Value = serde_json::from_str(&ev.data)
                    .map_err(|e| bad_response(format!("invalid stream JSON: {e}")))?;
                state.handle(&v)
            })
            .filter_map(|r| async move { r.transpose() });
        Ok(chunks.boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChunkAggregator;
    use flow_agent_core::ToolDefinition;

    fn provider() -> AnthropicProvider {
        let mut cfg = ProviderConfig::new("claude-test");
        cfg.api_key = Some("sk-ant-test".into());
        AnthropicProvider::new(cfg).unwrap()
    }

    #[test]
    fn key_required() {
        assert!(AnthropicProvider::new(ProviderConfig::new("m")).is_err());
    }

    #[test]
    fn body_shape_and_alternation() {
        let req = ChatRequest {
            messages: vec![
                Message::system("be brief"),
                Message::user("weather?"),
                Message::assistant("checking").with_tool_call("w", json!({"city": "Paris"})),
                Message::tool("call_1", "sunny"),
                Message::tool("call_2", "warm"),
                Message::user("thanks"),
            ],
            tools: vec![ToolDefinition {
                name: "w".into(),
                description: "weather".into(),
                parameters: json!({"type": "object"}),
            }],
            tool_choice: ToolChoice::Required,
            ..Default::default()
        };
        let b = provider().build_body(&req, false);
        assert_eq!(b["system"], "be brief");
        assert_eq!(b["max_tokens"], DEFAULT_MAX_TOKENS);
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "{msgs:#?}");
        assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][1]["tool_use_id"], "call_2");
        assert_eq!(msgs[2]["content"][2]["text"], "thanks");
        assert_eq!(b["tool_choice"]["type"], "any");
        assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn images() {
        let m: Message = serde_json::from_value(json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        ]}))
        .unwrap();
        let w = messages_to_wire(&[m]);
        assert_eq!(w[0]["content"][0]["source"]["media_type"], "image/png");
        assert_eq!(w[0]["content"][1]["source"]["type"], "url");
    }

    #[test]
    fn parse_full_response() {
        let m = parse_response(&json!({
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "w", "input": {"city": "Paris"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 7}
        }))
        .unwrap();
        assert_eq!(m.text(), "Let me check.");
        assert_eq!(m.tool_calls[0].id, "toolu_1");
        assert_eq!(m.usage, Some(Usage::new(12, 7)));
        assert_eq!(m.metadata["finish_reason"], "tool_calls");
    }

    #[test]
    fn stream_events_aggregate() {
        let events = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "w"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Oslo\"}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}),
            json!({"type": "message_stop"}),
        ];
        let mut st = StreamState::default();
        let mut agg = ChunkAggregator::new("anthropic");
        for e in &events {
            if let Some(c) = st.handle(e).unwrap() {
                agg.push(&c);
            }
        }
        let m = agg.finish().unwrap();
        assert_eq!(m.text(), "Hi there");
        assert_eq!(m.tool_calls[0].id, "toolu_2");
        assert_eq!(m.tool_calls[0].args["city"], "Oslo");
        assert_eq!(m.usage, Some(Usage::new(9, 4)));

        let err = StreamState::default()
            .handle(
                &json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
            )
            .unwrap_err();
        assert!(err.is_retryable());
    }
}
