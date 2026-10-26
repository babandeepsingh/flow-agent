//! [`ChatModel`]: one API over every provider.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::{
    Error, Event, JsonOutputParser, Message, Result, RetryPolicy, RunConfig, Runnable,
    ToolDefinition, ToolRegistry,
};
use futures::StreamExt;
use serde_json::Value;
use tracing::Instrument;
use uuid::Uuid;

use crate::provider::{ChatProvider, ChatRequest, ChunkAggregator, ModelOptions, ToolChoice};
use crate::registry::{ModelRegistry, ProviderConfig, parse_spec};

/// Default number of retries for retryable errors (rate limits, 5xx, network).
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// A chat model. Cheap to clone; implements [`Runnable`].
///
/// - **Input**: anything [`Message::list_from_value`] accepts (a string, a
///   message, a list of messages, or graph state with `"messages"`).
/// - **Output**: the assistant message as JSON (or the parsed object when
///   [`with_structured_output`](ChatModel::with_structured_output) is used).
///
/// When the run is being streamed (or has callbacks), the model streams from
/// the provider and emits [`Event::Token`] events, even when called through
/// `invoke`.
///
/// ```no_run
/// # async fn demo() -> flow_agent_core::Result<()> {
/// use flow_agent_core::{Runnable, json};
/// use flow_agent_models::ChatModel;
///
/// let model = ChatModel::builder("anthropic:claude-sonnet-5-5")
///     .api_key("sk-ant-...")
///     .temperature(0.2)
///     .build()?;
/// let reply = model.invoke(json!("Explain Rc vs Arc in one sentence.")).await?;
/// println!("{}", reply["content"]);
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct ChatModel {
    provider: Arc<dyn ChatProvider>,
    spec: String,
    options: ModelOptions,
    tools: Vec<ToolDefinition>,
    tool_choice: ToolChoice,
    structured: Option<Value>,
    max_retries: u32,
    timeout: Option<Duration>,
}

impl fmt::Debug for ChatModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatModel")
            .field("spec", &self.spec)
            .field("options", &self.options)
            .field(
                "tools",
                &self.tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
            )
            .field("tool_choice", &self.tool_choice)
            .field("structured", &self.structured.is_some())
            .field("max_retries", &self.max_retries)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl ChatModel {
    /// Start building a model from a `"provider:model"` spec.
    pub fn builder(spec: impl Into<String>) -> ChatModelBuilder {
        ChatModelBuilder::new(spec)
    }

    /// Build from a spec and connection settings using the default registry.
    pub fn from_spec(spec: &str, config: ProviderConfig) -> Result<Self> {
        let (provider, model) = parse_spec(spec)?;
        let config = ProviderConfig {
            model: model.to_owned(),
            ..config
        };
        Ok(Self::from_provider(
            ModelRegistry::default().create(provider, config)?,
        ))
    }

    /// Wrap any provider (custom backends, [`FakeChatModel`](crate::FakeChatModel)).
    pub fn from_provider(provider: Arc<dyn ChatProvider>) -> Self {
        ChatModel {
            spec: format!("{}:{}", provider.provider(), provider.model()),
            provider,
            options: ModelOptions::default(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            structured: None,
            max_retries: DEFAULT_MAX_RETRIES,
            timeout: None,
        }
    }

    /// `"provider:model"`.
    pub fn spec(&self) -> &str {
        &self.spec
    }

    /// Provider name.
    pub fn provider_name(&self) -> &str {
        self.provider.provider()
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        self.provider.model()
    }

    /// Request options.
    pub fn options(&self) -> &ModelOptions {
        &self.options
    }

    /// Bound tool definitions.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// A copy of this model that may call the tools in `registry`.
    pub fn bind_tools(&self, registry: &ToolRegistry) -> Self {
        self.bind_tool_definitions(registry.definitions())
    }

    /// A copy of this model with these tool definitions.
    pub fn bind_tool_definitions(&self, tools: Vec<ToolDefinition>) -> Self {
        let mut m = self.clone();
        m.tools = tools;
        m.structured = None;
        m
    }

    /// Set the tool choice.
    pub fn with_tool_choice(self, choice: ToolChoice) -> Self {
        let mut m = self;
        m.tool_choice = choice;
        m
    }

    /// Make the model answer with JSON matching `schema`; `invoke` then returns
    /// the parsed object instead of a message.
    ///
    /// Implemented by forcing a call to a tool whose parameters are `schema`
    /// (name from `schema["title"]`, default `"structured_output"`).
    pub fn with_structured_output(self, schema: Value) -> Self {
        let name = schema
            .get("title")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("structured_output")
            .to_owned();
        let description = schema
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("Respond with structured output matching this schema.")
            .to_owned();
        let mut m = self;
        m.tools = vec![ToolDefinition {
            name: name.clone(),
            description,
            parameters: schema.clone(),
        }];
        m.tool_choice = ToolChoice::Tool(name);
        m.structured = Some(schema);
        m
    }

    /// Set the request options.
    pub fn with_options(self, options: ModelOptions) -> Self {
        let mut m = self;
        m.options = options;
        m
    }

    /// Set the retry count for retryable errors.
    pub fn with_max_retries(self, max_retries: u32) -> Self {
        let mut m = self;
        m.max_retries = max_retries;
        m
    }

    /// Set a timeout for each call (including streaming).
    ///
    /// Shadows `RunnableExt::with_timeout` so the result is still a `ChatModel`.
    pub fn with_timeout(self, timeout: Duration) -> Self {
        let mut m = self;
        m.timeout = Some(timeout);
        m
    }

    fn request(&self, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            messages,
            tools: self.tools.clone(),
            tool_choice: if self.tools.is_empty() {
                ToolChoice::Auto
            } else {
                self.tool_choice.clone()
            },
            options: self.options.clone(),
        }
    }

    /// Generate one assistant message. Retries, timeout, cancellation,
    /// events and tracing are handled here.
    ///
    /// When a streamed call fails mid-stream and is retried, listeners may see
    /// the tokens of the failed attempt before those of the retry.
    pub async fn generate(&self, messages: Vec<Message>, config: &RunConfig) -> Result<Message> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "no messages to send to the model".into(),
            ));
        }
        let streaming = config.has_listeners();
        let run_id = Uuid::now_v7();
        let span = tracing::info_span!(
            "flow_agent.model",
            provider = %self.provider.provider(),
            model = %self.provider.model(),
            stream = streaming,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );
        if config.trace_content {
            tracing::debug!(parent: &span, messages = ?messages, "model input");
        }

        config
            .emit(Event::ModelStart {
                run_id,
                node: config.node.clone(),
                model: self.spec.clone(),
            })
            .await;

        let request = self.request(messages);
        let policy = RetryPolicy::exponential(self.max_retries.saturating_add(1));
        let result = policy
            .run(|_attempt| self.call_once(request.clone(), streaming, run_id, config))
            .instrument(span.clone())
            .await;

        match result {
            Ok(mut message) => {
                if message.id.is_none() {
                    message.id = Some(Uuid::now_v7().to_string());
                }
                if let Some(u) = message.usage {
                    span.record("input_tokens", u.input_tokens);
                    span.record("output_tokens", u.output_tokens);
                }
                if let Some(fr) = message
                    .metadata
                    .get("finish_reason")
                    .and_then(Value::as_str)
                {
                    span.record("finish_reason", fr);
                }
                if config.trace_content {
                    tracing::debug!(parent: &span, message = ?message, "model output");
                }
                config
                    .emit(Event::ModelEnd {
                        run_id,
                        node: config.node.clone(),
                        message: message.clone(),
                    })
                    .await;
                Ok(message)
            }
            Err(e) => {
                tracing::warn!(parent: &span, error = %e, "model call failed");
                config.emit_error(&e).await;
                Err(e)
            }
        }
    }

    async fn call_once(
        &self,
        request: ChatRequest,
        streaming: bool,
        run_id: Uuid,
        config: &RunConfig,
    ) -> Result<Message> {
        config.check_cancelled()?;
        let work = async {
            if !streaming {
                return self.provider.generate(request).await;
            }
            let mut stream = self.provider.stream(request).await?;
            let mut agg = ChunkAggregator::new(self.provider.provider());
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if let Some(text) = chunk.text.as_ref().filter(|t| !t.is_empty()) {
                    config
                        .emit(Event::Token {
                            run_id,
                            node: config.node.clone(),
                            text: text.clone(),
                        })
                        .await;
                }
                for tc in &chunk.tool_call_chunks {
                    config
                        .emit(Event::ToolCallChunk {
                            run_id,
                            node: config.node.clone(),
                            index: tc.index,
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            args_delta: tc.args_delta.clone(),
                        })
                        .await;
                }
                agg.push(&chunk);
            }
            agg.finish()
        };
        let work = async {
            match self.timeout {
                Some(t) => tokio::time::timeout(t, work)
                    .await
                    .map_err(|_| Error::Timeout {
                        what: format!("model '{}'", self.spec),
                        after: t,
                    })?,
                None => work.await,
            }
        };
        tokio::select! {
            res = work => res,
            _ = config.cancellation_token().cancelled() => Err(Error::Cancelled),
        }
    }

    fn parse_structured(&self, schema: &Value, message: &Message) -> Result<Value> {
        let parser = JsonOutputParser::with_schema(schema.clone());
        match message.tool_calls.first() {
            Some(tc) => parser.parse(&tc.args.to_string()),
            None => parser.parse(&message.text()),
        }
    }
}

#[async_trait]
impl Runnable for ChatModel {
    fn name(&self) -> &str {
        &self.spec
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let messages = Message::list_from_value(input)?;
        let message = self.generate(messages, config).await?;
        match &self.structured {
            Some(schema) => self.parse_structured(schema, &message),
            None => Ok(message.to_value()),
        }
    }
}

/// Builder for [`ChatModel`]. Created by [`ChatModel::builder`].
#[must_use]
pub struct ChatModelBuilder {
    spec: String,
    config: ProviderConfig,
    options: ModelOptions,
    max_retries: u32,
    timeout: Option<Duration>,
    registry: Option<ModelRegistry>,
}

impl fmt::Debug for ChatModelBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatModelBuilder")
            .field("spec", &self.spec)
            .field("config", &self.config)
            .field("options", &self.options)
            .finish()
    }
}

impl ChatModelBuilder {
    fn new(spec: impl Into<String>) -> Self {
        ChatModelBuilder {
            spec: spec.into(),
            config: ProviderConfig::default(),
            options: ModelOptions::default(),
            max_retries: DEFAULT_MAX_RETRIES,
            timeout: None,
            registry: None,
        }
    }

    /// API key. Read it from wherever your app keeps secrets; the library
    /// never reads environment variables.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.config.api_key = (!key.is_empty()).then_some(key);
        self
    }

    /// Override the API base URL (e.g. `http://localhost:11434/v1` for Ollama).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = Some(url.into());
        self
    }

    /// Add an HTTP header to every request.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.push((name.into(), value.into()));
        self
    }

    /// Per-request HTTP timeout (connect and idle read). Default 120s.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = Some(timeout);
        self
    }

    /// Use your own `reqwest` client (proxies, custom TLS...).
    #[cfg(feature = "http")]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "openai", feature = "anthropic"))))]
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.config.http_client = Some(client);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.options.temperature = Some(t);
        self
    }

    /// Max output tokens.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.options.max_tokens = Some(n);
        self
    }

    /// Nucleus sampling.
    pub fn top_p(mut self, p: f64) -> Self {
        self.options.top_p = Some(p);
        self
    }

    /// Add a stop sequence.
    pub fn stop(mut self, s: impl Into<String>) -> Self {
        self.options.stop.push(s.into());
        self
    }

    /// Extra provider-specific request field.
    pub fn extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.options.extra.insert(key.into(), value);
        self
    }

    /// Retries for retryable errors (default 2).
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Timeout for each model call, including the whole stream.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Resolve the provider with this registry instead of [`ModelRegistry::default`].
    pub fn registry(mut self, registry: &ModelRegistry) -> Self {
        self.registry = Some(registry.clone());
        self
    }

    /// Build the model.
    pub fn build(self) -> Result<ChatModel> {
        let (provider, model) = parse_spec(&self.spec)?;
        let config = ProviderConfig {
            model: model.to_owned(),
            ..self.config
        };
        let registry = self.registry.unwrap_or_default();
        let mut m = ChatModel::from_provider(registry.create(provider, config)?);
        m.options = self.options;
        m.max_retries = self.max_retries;
        m.timeout = self.timeout;
        Ok(m)
    }
}

/// Helper for `examples/` and tests: the text of the last message in a value.
pub fn last_text(value: &Value) -> String {
    flow_agent_core::message::last_message(value)
        .map(|m| m.text())
        .unwrap_or_else(|| match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
}
