//! A scripted, offline chat model for tests and examples.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::{Error, Message, Result, Role, Usage};
use serde_json::Value;

use crate::model::ChatModel;
use crate::provider::{
    ChatChunk, ChatProvider, ChatRequest, ChunkStream, FinishReason, ToolCallChunk,
};

#[derive(Default)]
struct State {
    responses: Vec<Message>,
    next: usize,
    cycle: bool,
    echo: bool,
    fail_times: u32,
    calls: Vec<ChatRequest>,
}

/// A chat model that returns scripted responses. No network, deterministic.
///
/// Clones share state, so a clone kept by the test can inspect
/// [`calls`](FakeChatModel::calls) after the model was moved into a chain.
///
/// ```
/// use flow_agent_core::{Message, Runnable, json};
/// use flow_agent_models::{ChatModel, FakeChatModel};
///
/// # tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(async {
/// let fake = FakeChatModel::new([
///     Message::assistant("").with_tool_call("get_weather", json!({"city": "Paris"})),
///     Message::assistant("It's sunny in Paris."),
/// ]);
/// let model: ChatModel = fake.clone().into();
///
/// let first = model.invoke(json!("Weather in Paris?")).await.unwrap();
/// assert_eq!(first["tool_calls"][0]["name"], "get_weather");
/// let second = model.invoke(json!("...")).await.unwrap();
/// assert_eq!(second["content"], "It's sunny in Paris.");
/// assert_eq!(fake.calls().len(), 2);
/// # });
/// ```
///
/// When streamed, text is split into word tokens (`"It's"`, `" sunny"`, ...)
/// and tool calls arrive as one chunk each.
#[derive(Clone)]
pub struct FakeChatModel {
    model: String,
    delay: Option<Duration>,
    state: Arc<Mutex<State>>,
}

impl std::fmt::Debug for FakeChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeChatModel")
            .field("model", &self.model)
            .finish()
    }
}

impl FakeChatModel {
    /// Return `responses` in order, then fail with a model error.
    pub fn new<I: IntoIterator<Item = Message>>(responses: I) -> Self {
        FakeChatModel {
            model: "fake".into(),
            delay: None,
            state: Arc::new(Mutex::new(State {
                responses: responses.into_iter().collect(),
                ..Default::default()
            })),
        }
    }

    /// Always answer with the text of the last user message.
    pub fn echo() -> Self {
        let f = Self::new([]);
        f.lock().echo = true;
        f
    }

    /// Start over from the first response when the script runs out.
    pub fn cycle(self) -> Self {
        self.lock().cycle = true;
        self
    }

    /// Fail the first `n` calls with a retryable error (to test retries).
    pub fn fail_times(self, n: u32) -> Self {
        self.lock().fail_times = n;
        self
    }

    /// Wait this long before answering (to test timeouts and cancellation).
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Set the reported model name (default `"fake"`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Every request received so far.
    pub fn calls(&self) -> Vec<ChatRequest> {
        self.lock().calls.clone()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while holding the lock can only come from a panicking test;
        // keep going with the data either way.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    async fn respond(&self, request: ChatRequest) -> Result<Message> {
        if let Some(d) = self.delay {
            tokio::time::sleep(d).await;
        }
        let mut s = self.lock();
        s.calls.push(request.clone());
        if s.fail_times > 0 {
            s.fail_times -= 1;
            return Err(Error::Model {
                provider: "fake".into(),
                message: "scripted failure".into(),
                status: Some(503),
                retryable: true,
            });
        }
        let mut msg = if s.echo {
            let text = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == Role::User)
                .map(Message::text)
                .unwrap_or_default();
            Message::assistant(text)
        } else {
            if s.next >= s.responses.len() && s.cycle && !s.responses.is_empty() {
                s.next = 0;
            }
            let Some(m) = s.responses.get(s.next).cloned() else {
                return Err(Error::Model {
                    provider: "fake".into(),
                    message: format!(
                        "no more scripted responses (got call #{}, script has {})",
                        s.calls.len(),
                        s.responses.len()
                    ),
                    status: None,
                    retryable: false,
                });
            };
            s.next += 1;
            m
        };
        let input: usize = request.messages.iter().map(|m| m.text().len()).sum();
        msg.usage
            .get_or_insert(Usage::new(input as u64 / 4, msg.text().len() as u64 / 4));
        let reason = if msg.has_tool_calls() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };
        msg.metadata
            .entry("finish_reason")
            .or_insert_with(|| Value::String(reason.as_str().into()));
        Ok(msg)
    }
}

#[async_trait]
impl ChatProvider for FakeChatModel {
    fn provider(&self) -> &str {
        "fake"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn generate(&self, request: ChatRequest) -> Result<Message> {
        self.respond(request).await
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChunkStream> {
        let msg = self.respond(request).await?;
        let mut chunks: Vec<Result<ChatChunk>> = split_words(&msg.text())
            .into_iter()
            .map(|t| {
                Ok(ChatChunk {
                    text: Some(t),
                    ..Default::default()
                })
            })
            .collect();
        for (index, tc) in msg.tool_calls.iter().enumerate() {
            chunks.push(Ok(ChatChunk {
                tool_call_chunks: vec![ToolCallChunk {
                    index,
                    id: Some(tc.id.clone()),
                    name: Some(tc.name.clone()),
                    args_delta: tc.args.to_string(),
                }],
                ..Default::default()
            }));
        }
        chunks.push(Ok(ChatChunk {
            usage: msg.usage,
            finish_reason: msg
                .metadata
                .get("finish_reason")
                .and_then(|v| serde_json::from_value(v.clone()).ok()),
            ..Default::default()
        }));
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

impl From<FakeChatModel> for ChatModel {
    fn from(fake: FakeChatModel) -> Self {
        ChatModel::from_provider(Arc::new(fake))
    }
}

/// Split into word tokens that keep their leading whitespace.
fn split_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_whitespace() && !cur.trim().is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_keeps_text() {
        let parts = split_words("It's  sunny\nin Paris.");
        assert_eq!(parts.concat(), "It's  sunny\nin Paris.");
        assert_eq!(parts[0], "It's");
        assert_eq!(parts[1], "  sunny");
        assert!(split_words("").is_empty());
    }

    #[tokio::test]
    async fn script_exhaustion_and_cycle() {
        let req = ChatRequest {
            messages: vec![Message::user("x")],
            ..Default::default()
        };
        let f = FakeChatModel::new([Message::assistant("a")]);
        assert!(f.generate(req.clone()).await.is_ok());
        assert!(f.generate(req.clone()).await.is_err());

        let f = FakeChatModel::new([Message::assistant("a"), Message::assistant("b")]).cycle();
        let mut out = vec![];
        for _ in 0..3 {
            out.push(f.generate(req.clone()).await.unwrap().text());
        }
        assert_eq!(out, ["a", "b", "a"]);
    }

    #[tokio::test]
    async fn echo_and_fail_times() {
        let f = FakeChatModel::echo().fail_times(1);
        let req = ChatRequest {
            messages: vec![Message::user("ping")],
            ..Default::default()
        };
        assert!(f.generate(req.clone()).await.unwrap_err().is_retryable());
        assert_eq!(f.generate(req).await.unwrap().text(), "ping");
        assert_eq!(f.calls().len(), 2);
    }
}
