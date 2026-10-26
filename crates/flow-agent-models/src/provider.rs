//! The provider interface and the normalized request/response types.

use std::collections::BTreeMap;

use async_trait::async_trait;
use flow_agent_core::{Error, Message, Result, ToolCall, ToolDefinition, Usage};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A stream of normalized chunks from a provider.
pub type ChunkStream = BoxStream<'static, Result<ChatChunk>>;

/// Whether and which tool the model must call.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides (default).
    #[default]
    Auto,
    /// The model must not call tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this tool.
    Tool(String),
}

/// Sampling and length options sent with each request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelOptions {
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Max output tokens. Anthropic requires one; 4096 is used when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Extra provider-specific fields merged into the request body as-is.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// A normalized chat request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Conversation so far.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Tool selection mode.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Sampling options.
    #[serde(default)]
    pub options: ModelOptions,
}

/// Why the model stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    /// Natural end or stop sequence.
    Stop,
    /// Hit the token limit.
    Length,
    /// Wants tools called.
    ToolCalls,
    /// Blocked by a content filter.
    ContentFilter,
    /// Anything else.
    Other,
}

impl FinishReason {
    /// Snake-case name.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other => "other",
        }
    }
}

/// A fragment of a tool call while streaming.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallChunk {
    /// Position of the tool call within the message.
    pub index: usize,
    /// Call id (usually only on the first fragment).
    pub id: Option<String>,
    /// Tool name (usually only on the first fragment).
    pub name: Option<String>,
    /// Raw JSON fragment of the arguments.
    pub args_delta: String,
}

/// A normalized streaming chunk.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Text delta.
    pub text: Option<String>,
    /// Tool call deltas.
    #[serde(default)]
    pub tool_call_chunks: Vec<ToolCallChunk>,
    /// Usage, usually on the last chunk.
    pub usage: Option<Usage>,
    /// Finish reason, on the last chunk.
    pub finish_reason: Option<FinishReason>,
}

/// A chat model backend (OpenAI, Anthropic, a fake...).
///
/// Implement this to add a provider; wrap it with
/// [`ChatModel::from_provider`](crate::ChatModel::from_provider).
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Provider name, e.g. `"anthropic"`.
    fn provider(&self) -> &str;

    /// Model name, e.g. `"claude-sonnet-5-5"`.
    fn model(&self) -> &str;

    /// Generate one complete assistant message.
    async fn generate(&self, request: ChatRequest) -> Result<Message>;

    /// Stream the response. Errors that happen before streaming starts
    /// (bad status, auth) are returned directly.
    ///
    /// Default: [`generate`](ChatProvider::generate) and return the message as
    /// a single chunk.
    async fn stream(&self, request: ChatRequest) -> Result<ChunkStream> {
        let message = self.generate(request).await?;
        let chunk = message_to_chunk(&message);
        Ok(Box::pin(futures::stream::once(async move { Ok(chunk) })))
    }
}

/// Express a complete message as one chunk.
pub fn message_to_chunk(message: &Message) -> ChatChunk {
    let text = message.text();
    ChatChunk {
        text: (!text.is_empty()).then_some(text),
        tool_call_chunks: message
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, tc)| ToolCallChunk {
                index,
                id: Some(tc.id.clone()),
                name: Some(tc.name.clone()),
                args_delta: tc.args.to_string(),
            })
            .collect(),
        usage: message.usage,
        finish_reason: message
            .metadata
            .get("finish_reason")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
    }
}

#[derive(Default)]
struct PartialCall {
    id: Option<String>,
    name: Option<String>,
    args: String,
}

/// Accumulates streamed chunks into the final assistant message, so `stream`
/// and `generate` produce the same result.
///
/// ```
/// use flow_agent_models::{ChatChunk, ChunkAggregator, ToolCallChunk};
///
/// let mut agg = ChunkAggregator::new("test");
/// agg.push(&ChatChunk { text: Some("Hel".into()), ..Default::default() });
/// agg.push(&ChatChunk { text: Some("lo".into()), ..Default::default() });
/// agg.push(&ChatChunk {
///     tool_call_chunks: vec![ToolCallChunk { index: 0, id: Some("c1".into()), name: Some("t".into()), args_delta: "{\"a\":".into() }],
///     ..Default::default()
/// });
/// agg.push(&ChatChunk {
///     tool_call_chunks: vec![ToolCallChunk { index: 0, args_delta: "1}".into(), ..Default::default() }],
///     ..Default::default()
/// });
/// let msg = agg.finish().unwrap();
/// assert_eq!(msg.text(), "Hello");
/// assert_eq!(msg.tool_calls[0].args["a"], 1);
/// ```
pub struct ChunkAggregator {
    provider: String,
    text: String,
    calls: BTreeMap<usize, PartialCall>,
    usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
}

impl ChunkAggregator {
    /// A new aggregator; `provider` is used in error messages.
    pub fn new(provider: impl Into<String>) -> Self {
        ChunkAggregator {
            provider: provider.into(),
            text: String::new(),
            calls: BTreeMap::new(),
            usage: None,
            finish_reason: None,
        }
    }

    /// Add a chunk.
    pub fn push(&mut self, chunk: &ChatChunk) {
        if let Some(t) = &chunk.text {
            self.text.push_str(t);
        }
        for tc in &chunk.tool_call_chunks {
            let call = self.calls.entry(tc.index).or_default();
            if call.id.is_none() {
                call.id.clone_from(&tc.id);
            }
            if call.name.is_none() {
                call.name.clone_from(&tc.name);
            }
            call.args.push_str(&tc.args_delta);
        }
        if let Some(u) = chunk.usage {
            self.usage = Some(u);
        }
        if chunk.finish_reason.is_some() {
            self.finish_reason = chunk.finish_reason;
        }
    }

    /// Build the final message.
    pub fn finish(self) -> Result<Message> {
        let mut msg = Message::assistant(self.text);
        for (index, call) in self.calls {
            let name = call.name.ok_or_else(|| Error::Model {
                provider: self.provider.clone(),
                message: format!("streamed tool call #{index} has no name"),
                status: None,
                retryable: false,
            })?;
            let args = if call.args.trim().is_empty() {
                Value::Object(Map::new())
            } else {
                serde_json::from_str(&call.args).map_err(|e| Error::Model {
                    provider: self.provider.clone(),
                    message: format!("tool call '{name}' has invalid JSON arguments: {e}"),
                    status: None,
                    retryable: false,
                })?
            };
            let mut tc = ToolCall::new(name, args);
            if let Some(id) = call.id {
                tc.id = id;
            }
            msg.tool_calls.push(tc);
        }
        msg.usage = self.usage;
        if let Some(fr) = self.finish_reason {
            msg.metadata
                .insert("finish_reason".into(), Value::String(fr.as_str().into()));
        }
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chunk_roundtrip_through_aggregator() {
        let mut m = Message::assistant("hi").with_tool_call("t", json!({"x": [1, 2]}));
        m.usage = Some(Usage::new(3, 4));
        m.metadata
            .insert("finish_reason".into(), json!("tool_calls"));
        let mut agg = ChunkAggregator::new("p");
        agg.push(&message_to_chunk(&m));
        let back = agg.finish().unwrap();
        assert_eq!(back.text(), "hi");
        assert_eq!(back.tool_calls, m.tool_calls);
        assert_eq!(back.usage, m.usage);
        assert_eq!(back.metadata["finish_reason"], "tool_calls");
    }

    #[test]
    fn bad_args_and_missing_name_are_errors() {
        let mut agg = ChunkAggregator::new("p");
        agg.push(&ChatChunk {
            tool_call_chunks: vec![ToolCallChunk {
                index: 0,
                name: Some("t".into()),
                args_delta: "{oops".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert!(agg.finish().is_err());

        let mut agg = ChunkAggregator::new("p");
        agg.push(&ChatChunk {
            tool_call_chunks: vec![ToolCallChunk::default()],
            ..Default::default()
        });
        assert!(agg.finish().is_err());
    }
}
