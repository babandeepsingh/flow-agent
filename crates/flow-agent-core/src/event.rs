//! Streaming events and callback handlers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

use crate::config::RunConfig;
use crate::error::Error;
use crate::message::Message;

/// Something that happened during a run.
///
/// One enum is shared by models, tools, chains and graphs so a single stream
/// can be forwarded as JSON (SSE, WebSocket) without mapping. Serialized with
/// a `"type"` tag:
///
/// ```
/// use flow_agent_core::Event;
/// let e = Event::Token { run_id: uuid::Uuid::nil(), node: None, text: "Hi".into() };
/// assert_eq!(serde_json::to_value(&e).unwrap()["type"], "token");
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Event {
    /// A text token streamed from a model.
    Token {
        /// Run that produced the token.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// The token text.
        text: String,
    },
    /// A fragment of a tool call being streamed from a model.
    ToolCallChunk {
        /// Run that produced the chunk.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// Index of the tool call within the message.
        index: usize,
        /// Tool call id, usually only on the first chunk.
        id: Option<String>,
        /// Tool name, usually only on the first chunk.
        name: Option<String>,
        /// Raw JSON fragment of the arguments.
        args_delta: String,
    },
    /// A model call started.
    ModelStart {
        /// Run id of the model call.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// `"provider:model"`.
        model: String,
    },
    /// A model call finished.
    ModelEnd {
        /// Run id of the model call.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// The complete assistant message.
        message: Message,
    },
    /// A tool call started.
    ToolStart {
        /// Run id of the tool call.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// Tool name.
        tool: String,
        /// Tool call id.
        call_id: String,
        /// Arguments.
        args: Value,
    },
    /// A tool call finished.
    ToolEnd {
        /// Run id of the tool call.
        run_id: Uuid,
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// Tool name.
        tool: String,
        /// Tool call id.
        call_id: String,
        /// Result (or error description when `is_error`).
        output: Value,
        /// Whether the tool failed.
        is_error: bool,
    },
    /// A graph node started.
    NodeStart {
        /// Superstep.
        step: u32,
        /// Node name.
        node: String,
        /// Unique id of this node execution.
        task_id: Uuid,
    },
    /// A graph node finished.
    NodeEnd {
        /// Superstep.
        step: u32,
        /// Node name.
        node: String,
        /// Unique id of this node execution.
        task_id: Uuid,
        /// The partial state update the node returned.
        update: Value,
        /// Wall-clock duration.
        duration_ms: u64,
    },
    /// State after a superstep's updates were applied.
    StateUpdate {
        /// Superstep.
        step: u32,
        /// Nodes that wrote in this step.
        nodes: Vec<String>,
        /// Full state.
        values: Value,
    },
    /// The run paused waiting for human input.
    Interrupt {
        /// Superstep.
        step: u32,
        /// Node that interrupted.
        node: String,
        /// Id to correlate the resume value.
        interrupt_id: String,
        /// Payload for the human (question, draft...).
        value: Value,
    },
    /// A checkpoint was saved.
    Checkpoint {
        /// Checkpoint id.
        checkpoint_id: String,
        /// Superstep.
        step: u32,
    },
    /// An application-defined event.
    Custom {
        /// Graph node, when running inside a graph.
        node: Option<String>,
        /// Event name.
        name: String,
        /// Payload.
        data: Value,
    },
    /// The final output of the run. Always the last event of a successful stream.
    Output {
        /// The output value.
        value: Value,
    },
}

/// The kind of an [`Event`], for filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum EventKind {
    Token,
    ToolCallChunk,
    ModelStart,
    ModelEnd,
    ToolStart,
    ToolEnd,
    NodeStart,
    NodeEnd,
    StateUpdate,
    Interrupt,
    Checkpoint,
    Custom,
    Output,
}

impl Event {
    /// This event's kind.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Token { .. } => EventKind::Token,
            Event::ToolCallChunk { .. } => EventKind::ToolCallChunk,
            Event::ModelStart { .. } => EventKind::ModelStart,
            Event::ModelEnd { .. } => EventKind::ModelEnd,
            Event::ToolStart { .. } => EventKind::ToolStart,
            Event::ToolEnd { .. } => EventKind::ToolEnd,
            Event::NodeStart { .. } => EventKind::NodeStart,
            Event::NodeEnd { .. } => EventKind::NodeEnd,
            Event::StateUpdate { .. } => EventKind::StateUpdate,
            Event::Interrupt { .. } => EventKind::Interrupt,
            Event::Checkpoint { .. } => EventKind::Checkpoint,
            Event::Custom { .. } => EventKind::Custom,
            Event::Output { .. } => EventKind::Output,
        }
    }

    /// Serialize to JSON.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Receives every event of a run, for logging, metrics or tracing exporters.
///
/// ```
/// use flow_agent_core::{CallbackHandler, Event, RunConfig};
///
/// struct CountTokens(std::sync::atomic::AtomicUsize);
///
/// #[async_trait::async_trait]
/// impl CallbackHandler for CountTokens {
///     async fn on_event(&self, event: &Event, _config: &RunConfig) {
///         if matches!(event, Event::Token { .. }) {
///             self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    /// Called for every event.
    async fn on_event(&self, event: &Event, config: &RunConfig);

    /// Called when a runnable fails. Default: does nothing.
    async fn on_error(&self, error: &Error, config: &RunConfig) {
        let _ = (error, config);
    }
}

/// Internal fan-out of events to active streams. Cloning shares the targets.
#[derive(Clone, Default)]
pub(crate) struct EventSink {
    targets: Vec<UnboundedSender<Event>>,
}

impl EventSink {
    pub(crate) fn with_target(&self, tx: UnboundedSender<Event>) -> Self {
        let mut targets = self.targets.clone();
        targets.retain(|t| !t.is_closed());
        targets.push(tx);
        EventSink { targets }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub(crate) fn send(&self, event: &Event) {
        for t in &self.targets {
            // A closed receiver just means nobody listens any more.
            let _ = t.send(event.clone());
        }
    }
}
