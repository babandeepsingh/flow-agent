//! Ready-made agents for [flow-agent](https://github.com/babandeepsingh/flow-agent).
//!
//! [`create_react_agent`] builds a ReAct-style agent: the model decides which
//! tools to call, the tools run, their results go back to the model, and the
//! loop repeats until the model answers without tool calls.
//!
//! ```
//! use flow_agent_core::{Message, Runnable, Tool, json};
//! use flow_agent_models::{ChatModel, FakeChatModel};
//! use flow_agent_prebuilt::create_react_agent;
//!
//! # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
//! let weather = Tool::new("get_weather", "Get the weather for a city")
//!     .param("city", "string", "City name")
//!     .handler(|args| async move { Ok(json!({ "city": args["city"], "forecast": "sunny" })) });
//!
//! // A scripted model for this example; use ChatModel::builder("anthropic:...") for real.
//! let model: ChatModel = FakeChatModel::new([
//!     Message::assistant("").with_tool_call("get_weather", json!({ "city": "Paris" })),
//!     Message::assistant("It's sunny in Paris."),
//! ]).into();
//!
//! let agent = create_react_agent(model, [weather]).unwrap();
//! let state = agent.invoke(json!("What's the weather in Paris?")).await.unwrap();
//! assert_eq!(flow_agent_models::last_text(&state), "It's sunny in Paris.");
//! # });
//! ```
//!
//! The result is a plain [`CompiledGraph`], so streaming, checkpoints,
//! interrupts and resume work exactly as for hand-built graphs.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use std::sync::Arc;

use flow_agent_core::{
    Checkpointer, Error, Message, Result, Role, Tool, ToolRegistry, ToolResult, Value, json,
};
use flow_agent_graph::{
    CompileOptions, CompiledGraph, END, NodeContext, Reducer, START, StateGraph, tools_condition,
};
use flow_agent_models::ChatModel;

/// Default limit of model calls per user turn.
pub const DEFAULT_MAX_ITERATIONS: usize = 25;

/// Text of the final message when [`ReactAgentBuilder::max_iterations`] is hit.
pub const ITERATION_LIMIT_MESSAGE: &str =
    "Sorry, I could not finish this request within the allowed number of steps.";

/// What the agent does when a tool fails.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolErrorMode {
    /// Send the error to the model as the tool's result so it can recover (default).
    #[default]
    ReturnToModel,
    /// Fail the run with the tool's error.
    Propagate,
}

/// Build a ReAct agent from a model and tools with default settings.
///
/// Input: a string, a message, a list of messages or `{"messages": [...]}`.
/// Output: the final state `{"messages": [...]}`.
pub fn create_react_agent<I>(model: ChatModel, tools: I) -> Result<CompiledGraph>
where
    I: IntoIterator<Item = Tool>,
{
    ReactAgent::builder(model).tools(tools).build()
}

/// Entry point for [`ReactAgentBuilder`].
#[derive(Debug)]
pub struct ReactAgent;

impl ReactAgent {
    /// Start configuring an agent.
    pub fn builder(model: ChatModel) -> ReactAgentBuilder {
        ReactAgentBuilder {
            model,
            tools: ToolRegistry::new(),
            system_prompt: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            parallel_tool_calls: true,
            tool_errors: ToolErrorMode::default(),
            interrupt_before_tools: false,
            checkpointer: None,
            name: "react_agent".into(),
        }
    }
}

/// Configures a ReAct agent.
///
/// ```
/// use std::sync::Arc;
/// use flow_agent_checkpoint::MemorySaver;
/// use flow_agent_models::{ChatModel, FakeChatModel};
/// use flow_agent_prebuilt::{ReactAgent, ToolErrorMode};
///
/// let model: ChatModel = FakeChatModel::echo().into();
/// let agent = ReactAgent::builder(model)
///     .system_prompt("You are a concise travel assistant.")
///     .max_iterations(10)
///     .tool_errors(ToolErrorMode::ReturnToModel)
///     .interrupt_before_tools(true)
///     .checkpointer(Arc::new(MemorySaver::new()))
///     .build()
///     .unwrap();
/// ```
#[must_use]
pub struct ReactAgentBuilder {
    model: ChatModel,
    tools: ToolRegistry,
    system_prompt: Option<String>,
    max_iterations: usize,
    parallel_tool_calls: bool,
    tool_errors: ToolErrorMode,
    interrupt_before_tools: bool,
    checkpointer: Option<Arc<dyn Checkpointer>>,
    name: String,
}

impl std::fmt::Debug for ReactAgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReactAgentBuilder")
            .field("model", &self.model.spec())
            .field("tools", &self.tools.names())
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl ReactAgentBuilder {
    /// Add tools.
    pub fn tools<I: IntoIterator<Item = Tool>>(mut self, tools: I) -> Self {
        for t in tools {
            self.tools.add(t);
        }
        self
    }

    /// Add one tool.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.add(tool);
        self
    }

    /// Use an existing registry (replaces tools added so far).
    pub fn tool_registry(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// System prompt sent before the conversation on every model call (not
    /// stored in the state).
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Max model calls per user turn. When reached, the agent stops with
    /// [`ITERATION_LIMIT_MESSAGE`] instead of looping forever.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    /// Run several tool calls from one model response concurrently (default true).
    pub fn parallel_tool_calls(mut self, on: bool) -> Self {
        self.parallel_tool_calls = on;
        self
    }

    /// How tool failures are handled.
    pub fn tool_errors(mut self, mode: ToolErrorMode) -> Self {
        self.tool_errors = mode;
        self
    }

    /// Pause before every tool execution so a human can approve (needs a
    /// checkpointer). Resume with `agent.resume(Value::Null, &config)`, or edit
    /// the pending call first with `update_state`.
    pub fn interrupt_before_tools(mut self, on: bool) -> Self {
        self.interrupt_before_tools = on;
        self
    }

    /// Persist conversations (multi-turn threads, resume, interrupts).
    pub fn checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Graph name used in traces.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build the agent graph.
    pub fn build(self) -> Result<CompiledGraph> {
        if self.interrupt_before_tools && self.checkpointer.is_none() {
            return Err(Error::Config(
                "interrupt_before_tools needs a checkpointer to resume from".into(),
            ));
        }
        let model = if self.tools.is_empty() {
            self.model
        } else {
            self.model.bind_tools(&self.tools)
        };
        let system = self.system_prompt.map(Message::system);
        let max_iterations = self.max_iterations;

        let agent = move |state: Value, ctx: NodeContext| {
            let model = model.clone();
            let system = system.clone();
            async move {
                let history = Message::list_from_value(state["messages"].clone())?;
                if iterations_this_turn(&history) >= max_iterations {
                    return Ok(json!({
                        "messages": [Message::assistant(ITERATION_LIMIT_MESSAGE).to_value()]
                    }));
                }
                let mut messages: Vec<Message> = system.into_iter().collect();
                messages.extend(history.into_iter().filter(|m| m.role != Role::System));
                let reply = model.generate(messages, ctx.config()).await?;
                Ok(json!({ "messages": [reply.to_value()] }))
            }
        };

        let tools = Arc::new(self.tools);
        let parallel = self.parallel_tool_calls;
        let errors = self.tool_errors;
        let tools_node = move |state: Value, ctx: NodeContext| {
            let tools = tools.clone();
            async move {
                let last = Message::list_from_value(state["messages"].clone())?
                    .pop()
                    .ok_or_else(|| Error::InvalidInput("no messages".into()))?;
                let results: Vec<ToolResult> = if parallel {
                    tools.call_many(&last.tool_calls, ctx.config()).await
                } else {
                    let mut out = Vec::new();
                    for call in &last.tool_calls {
                        out.push(tools.call(call, ctx.config()).await);
                    }
                    out
                };
                if errors == ToolErrorMode::Propagate {
                    if let Some(r) = results.iter().find(|r| r.is_error) {
                        return Err(Error::tool(
                            r.name.clone(),
                            r.output["error"].as_str().unwrap_or("tool failed"),
                        ));
                    }
                }
                let messages: Vec<Value> =
                    results.iter().map(|r| r.to_message().to_value()).collect();
                Ok(json!({ "messages": messages }))
            }
        };

        let graph = StateGraph::new()
            .name(self.name)
            .key("messages", Reducer::Messages)
            .add_node("agent", agent)
            .add_node("tools", tools_node)
            .add_edge(START, "agent")
            .add_conditional_edges_with("agent", tools_condition, [("tools", "tools"), (END, END)])
            .add_edge("tools", "agent");

        let mut options = CompileOptions::new();
        if let Some(cp) = self.checkpointer {
            options = options.checkpointer(cp);
        }
        if self.interrupt_before_tools {
            options = options.interrupt_before(["tools"]);
        }
        graph.compile_with(options)
    }
}

/// Assistant messages since the last user message.
fn iterations_this_turn(history: &[Message]) -> usize {
    history
        .iter()
        .rev()
        .take_while(|m| m.role != Role::User)
        .filter(|m| m.role == Role::Assistant)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_iterations_in_current_turn() {
        let h = vec![
            Message::user("a"),
            Message::assistant("x"),
            Message::user("b"),
            Message::assistant(""),
            Message::tool("1", "r"),
            Message::assistant(""),
        ];
        assert_eq!(iterations_this_turn(&h), 2);
        assert_eq!(iterations_this_turn(&[]), 0);
    }
}
