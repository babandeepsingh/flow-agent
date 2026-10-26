//! [`ToolNode`]: runs the tool calls of the last message.

use async_trait::async_trait;
use flow_agent_core::{Error, Message, Result, RunConfig, Runnable, ToolRegistry};
use serde_json::{Value, json};

/// Executes the tool calls of the last message in `state["messages"]`
/// concurrently and returns `{"messages": [tool results...]}`.
///
/// Tool failures are sent back to the model as error messages by default, so
/// it can recover; use [`ToolNode::propagate_errors`] to fail the node instead.
///
/// ```
/// use flow_agent_core::{Tool, ToolRegistry};
/// use flow_agent_graph::ToolNode;
/// let tools = ToolRegistry::new().with(Tool::new("noop", "Does nothing"));
/// let node = ToolNode::new(tools);
/// ```
#[derive(Clone, Debug)]
pub struct ToolNode {
    tools: ToolRegistry,
    propagate_errors: bool,
    messages_key: String,
}

impl ToolNode {
    /// A node running `tools`.
    pub fn new(tools: impl Into<ToolRegistry>) -> Self {
        ToolNode {
            tools: tools.into(),
            propagate_errors: false,
            messages_key: "messages".into(),
        }
    }

    /// Fail the node when any tool fails (instead of telling the model).
    pub fn propagate_errors(mut self, on: bool) -> Self {
        self.propagate_errors = on;
        self
    }

    /// Read and write messages under a different state key.
    pub fn messages_key(mut self, key: impl Into<String>) -> Self {
        self.messages_key = key.into();
        self
    }

    /// The tools.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }
}

#[async_trait]
impl Runnable for ToolNode {
    fn name(&self) -> &str {
        "tools"
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let source = match input.get(&self.messages_key) {
            Some(m) => m.clone(),
            None => input,
        };
        let last = Message::list_from_value(source)?
            .pop()
            .ok_or_else(|| Error::InvalidInput("ToolNode: no messages in state".into()))?;
        let results = self.tools.call_many(&last.tool_calls, config).await;
        if self.propagate_errors {
            if let Some(r) = results.iter().find(|r| r.is_error) {
                return Err(Error::tool(
                    r.name.clone(),
                    r.output["error"].as_str().unwrap_or("tool failed"),
                ));
            }
        }
        let messages: Vec<Value> = results.iter().map(|r| r.to_message().to_value()).collect();
        Ok(json!({ self.messages_key.clone(): messages }))
    }
}
