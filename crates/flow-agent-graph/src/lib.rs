//! LangGraph-style stateful graphs for [flow-agent](https://github.com/babandeepsingh/flow-agent).
//!
//! Build a [`StateGraph`] of nodes and edges over a JSON state, then
//! [`compile`](StateGraph::compile) it into a [`CompiledGraph`] that you can
//! `invoke`, `stream`, interrupt and resume.
//!
//! - **State**: a JSON object; each key has a [`Reducer`] (overwrite, append,
//!   merge, messages, custom).
//! - **Nodes**: async closures `|state, ctx| async { Ok(json!({...})) }` or any
//!   [`Runnable`](flow_agent_core::Runnable) (models, chains, other graphs).
//! - **Edges**: static, conditional (router functions), fan-out (several
//!   targets run in parallel), fan-in ([`StateGraph::add_edge_from`]) and
//!   map-reduce ([`Send`]).
//! - **Execution**: supersteps run all scheduled nodes concurrently, then apply
//!   their updates in a deterministic order. Per-node timeouts and retries, a
//!   recursion limit and cancellation are built in.
//! - **Persistence**: with a [`Checkpointer`](flow_agent_core::Checkpointer)
//!   every step is saved, enabling human-in-the-loop
//!   ([`NodeContext::interrupt`]), resume, state editing and replay.
//!
//! ```
//! use flow_agent_core::{Message, Runnable, ToolRegistry, Tool, json};
//! use flow_agent_graph::{Reducer, StateGraph, ToolNode, START, tools_condition};
//! use flow_agent_models::{ChatModel, FakeChatModel};
//!
//! # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
//! let tools = ToolRegistry::new().with(
//!     Tool::new("get_weather", "Weather for a city")
//!         .param("city", "string", "City")
//!         .handler(|a| async move { Ok(json!({"forecast": "sunny", "city": a["city"]})) }),
//! );
//! let model: ChatModel = FakeChatModel::new([
//!     Message::assistant("").with_tool_call("get_weather", json!({"city": "Paris"})),
//!     Message::assistant("It's sunny in Paris."),
//! ]).into();
//!
//! let app = StateGraph::new()
//!     .key("messages", Reducer::Messages)
//!     .add_node("agent", model.bind_tools(&tools))
//!     .add_node("tools", ToolNode::new(tools))
//!     .add_edge(START, "agent")
//!     .add_conditional_edges("agent", tools_condition)
//!     .add_edge("tools", "agent")
//!     .compile()
//!     .unwrap();
//!
//! let out = app.invoke(json!({"messages": ["Weather in Paris?"]})).await.unwrap();
//! assert_eq!(out["messages"].as_array().unwrap().len(), 4);
//! # });
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod builder;
mod compiled;
mod node;
mod reducer;
mod tool_node;

pub use builder::{CompileOptions, StateGraph};
pub use compiled::{CompiledGraph, INTERRUPT_KEY, RunOutput, StateSnapshot};
pub use node::{
    Command, END, FnMarker, IntoNode, Node, NodeContext, NodeOptions, NodeOutput, Route,
    RouteTarget, RunnableMarker, START, Send, tools_condition,
};
pub use reducer::Reducer;
pub use tool_node::ToolNode;
