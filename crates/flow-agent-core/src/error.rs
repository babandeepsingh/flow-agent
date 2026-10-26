//! The single error type used across flow-agent.

use std::time::Duration;

use serde_json::{Value, json};

/// Result alias used throughout flow-agent.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every error flow-agent can return.
///
/// Errors carry enough structure to decide what to do programmatically
/// ([`Error::is_retryable`]) and can be turned into JSON for APIs
/// ([`Error::to_value`]). Add context while propagating with
/// [`Error::context`] or [`ResultExt::context`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A model provider returned an error or an unusable response.
    #[error("model error from {provider}: {message}")]
    Model {
        /// Provider name, e.g. `"anthropic"`.
        provider: String,
        /// Human-readable message.
        message: String,
        /// HTTP status code, when there was one.
        status: Option<u16>,
        /// Whether retrying the same request may succeed (rate limits, 5xx, network).
        retryable: bool,
    },

    /// A tool handler failed.
    #[error("tool '{tool}' failed: {message}")]
    Tool {
        /// Tool name.
        tool: String,
        /// Human-readable message.
        message: String,
    },

    /// A tool was requested that is not registered.
    #[error("tool not found: '{0}'")]
    ToolNotFound(String),

    /// Arguments passed to a tool don't match its schema.
    #[error("invalid arguments for tool '{tool}': {message}")]
    InvalidToolArgs {
        /// Tool name.
        tool: String,
        /// What is wrong.
        message: String,
    },

    /// A prompt template is malformed or a variable is missing.
    #[error("template error: {message}")]
    Template {
        /// What is wrong.
        message: String,
        /// Variables that were required but not provided.
        missing: Vec<String>,
    },

    /// Model output could not be parsed.
    #[error("parse error: {message}")]
    Parse {
        /// What is wrong.
        message: String,
        /// The text that failed to parse (possibly truncated).
        input: String,
    },

    /// Input has the wrong shape for the component that received it.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The graph failed validation. Contains every problem found.
    #[error("invalid graph: {}", .0.join("; "))]
    InvalidGraph(Vec<String>),

    /// A state update could not be applied.
    #[error("invalid update for key '{key}': {message}")]
    InvalidUpdate {
        /// State key.
        key: String,
        /// What is wrong.
        message: String,
    },

    /// A graph node failed after all retries.
    #[error("node '{node}' failed at step {step}: {source}")]
    NodeFailed {
        /// Node name.
        node: String,
        /// Superstep number.
        step: u32,
        /// The underlying error.
        #[source]
        source: Box<Error>,
    },

    /// The graph ran more supersteps than allowed.
    #[error("recursion limit of {limit} supersteps reached")]
    RecursionLimit {
        /// The configured limit.
        limit: u32,
    },

    /// An operation took too long.
    #[error("{what} timed out after {after:?}")]
    Timeout {
        /// What timed out.
        what: String,
        /// The configured timeout.
        after: Duration,
    },

    /// The run was cancelled through its cancellation token.
    #[error("run cancelled")]
    Cancelled,

    /// A checkpointer failed.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// Invalid configuration (unknown provider, missing API key, missing checkpointer...).
    #[error("configuration error: {0}")]
    Config(String),

    /// JSON (de)serialization failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// An HTTP transport error.
    #[error("http error: {0}")]
    Http(String),

    /// An error with additional context.
    #[error("{context}: {source}")]
    Context {
        /// What was being done.
        context: String,
        /// The underlying error.
        #[source]
        source: Box<Error>,
    },

    /// Any other error.
    #[error(transparent)]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wrap this error with a description of what was being done.
    pub fn context(self, context: impl Into<String>) -> Self {
        Error::Context {
            context: context.into(),
            source: Box::new(self),
        }
    }

    /// Build an [`Error::Other`] from any error type.
    pub fn other(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Error::Other(Box::new(err))
    }

    /// Build an [`Error::Tool`].
    pub fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Error::Tool {
            tool: tool.into(),
            message: message.into(),
        }
    }

    /// Build an [`Error::Parse`], truncating very long inputs.
    pub fn parse(message: impl Into<String>, input: impl Into<String>) -> Self {
        let mut input = input.into();
        if input.len() > 500 {
            let mut cut = 500;
            while !input.is_char_boundary(cut) {
                cut -= 1;
            }
            input.truncate(cut);
            input.push('…');
        }
        Error::Parse {
            message: message.into(),
            input,
        }
    }

    /// Whether retrying the same operation may succeed.
    ///
    /// True for retryable model errors (rate limits, 5xx, network), timeouts
    /// and HTTP transport errors. Context wrappers are looked through.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Model { retryable, .. } => *retryable,
            Error::Timeout { .. } | Error::Http(_) => true,
            Error::Context { source, .. } | Error::NodeFailed { source, .. } => {
                source.is_retryable()
            }
            _ => false,
        }
    }

    /// The innermost error, looking through [`Error::Context`] and [`Error::NodeFailed`].
    pub fn root(&self) -> &Error {
        match self {
            Error::Context { source, .. } | Error::NodeFailed { source, .. } => source.root(),
            other => other,
        }
    }

    /// A short, stable, snake_case name for the error kind (e.g. `"timeout"`).
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Model { .. } => "model",
            Error::Tool { .. } => "tool",
            Error::ToolNotFound(_) => "tool_not_found",
            Error::InvalidToolArgs { .. } => "invalid_tool_args",
            Error::Template { .. } => "template",
            Error::Parse { .. } => "parse",
            Error::InvalidInput(_) => "invalid_input",
            Error::InvalidGraph(_) => "invalid_graph",
            Error::InvalidUpdate { .. } => "invalid_update",
            Error::NodeFailed { .. } => "node_failed",
            Error::RecursionLimit { .. } => "recursion_limit",
            Error::Timeout { .. } => "timeout",
            Error::Cancelled => "cancelled",
            Error::Checkpoint(_) => "checkpoint",
            Error::Config(_) => "config",
            Error::Serde(_) => "serde",
            Error::Http(_) => "http",
            Error::Context { .. } => "context",
            Error::Other(_) => "other",
        }
    }

    /// JSON representation for APIs and events:
    /// `{"kind": "...", "message": "...", "retryable": bool, ...}`.
    pub fn to_value(&self) -> Value {
        let mut v = json!({
            "kind": self.kind(),
            "message": self.to_string(),
            "retryable": self.is_retryable(),
        });
        let extra = match self {
            Error::Model {
                provider, status, ..
            } => json!({ "provider": provider, "status": status }),
            Error::Tool { tool, .. } | Error::InvalidToolArgs { tool, .. } => {
                json!({ "tool": tool })
            }
            Error::Template { missing, .. } => json!({ "missing": missing }),
            Error::InvalidGraph(problems) => json!({ "problems": problems }),
            Error::NodeFailed { node, step, source } => {
                json!({ "node": node, "step": step, "source": source.to_value() })
            }
            Error::Context { source, .. } => json!({ "source": source.to_value() }),
            _ => Value::Null,
        };
        if let (Value::Object(map), Value::Object(extra)) = (&mut v, extra) {
            map.extend(extra);
        }
        v
    }
}

/// Adds `.context(..)` to any `Result<T, Error>`.
pub trait ResultExt<T> {
    /// Wrap the error, if any, with a description of what was being done.
    fn context(self, context: impl Into<String>) -> Result<T>;
}

impl<T, E: Into<Error>> ResultExt<T> for std::result::Result<T, E> {
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| e.into().context(context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_wraps_and_root_unwraps() {
        let err = Error::Cancelled.context("running chain");
        assert_eq!(err.to_string(), "running chain: run cancelled");
        assert!(matches!(err.root(), Error::Cancelled));
    }

    #[test]
    fn retryable_looks_through_context() {
        let err = Error::Model {
            provider: "x".into(),
            message: "rate limited".into(),
            status: Some(429),
            retryable: true,
        }
        .context("calling model");
        assert!(err.is_retryable());
        assert!(!Error::Cancelled.is_retryable());
    }

    #[test]
    fn to_value_has_kind_and_extras() {
        let v = Error::InvalidGraph(vec!["a".into(), "b".into()]).to_value();
        assert_eq!(v["kind"], "invalid_graph");
        assert_eq!(v["problems"], json!(["a", "b"]));
    }

    #[test]
    fn parse_truncates_long_input() {
        let Error::Parse { input, .. } = Error::parse("bad", "é".repeat(400)) else {
            panic!("expected parse error");
        };
        assert!(input.len() <= 504);
        assert!(input.ends_with('…'));
    }

    #[test]
    fn result_ext_context() {
        let r: std::result::Result<(), serde_json::Error> =
            serde_json::from_str::<()>("nope").map(|_| ());
        let err = r.context("decoding").unwrap_err();
        assert_eq!(err.kind(), "context");
    }
}
