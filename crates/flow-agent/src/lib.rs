//! # flow-agent
//!
//! LangChain and LangGraph ideas in Rust: LLM chains, tool-using agents and
//! stateful, resumable graphs. JSON in, JSON out, async on tokio. This crate
//! re-exports everything; add just this one dependency.
//!
//! ```
//! use flow_agent::prelude::*;
//!
//! # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
//! let weather = Tool::new("get_weather", "Get the weather for a city")
//!     .param("city", "string", "City name")
//!     .handler(|args| async move { Ok(json!({ "city": args["city"], "forecast": "sunny" })) });
//!
//! // Real model: ChatModel::builder("anthropic:claude-sonnet-5-5").api_key(key).build()?
//! let model: ChatModel = FakeChatModel::new([
//!     Message::assistant("").with_tool_call("get_weather", json!({ "city": "Paris" })),
//!     Message::assistant("It's sunny in Paris."),
//! ]).into();
//!
//! let agent = create_react_agent(model, [weather])?;
//! let state = agent.invoke(json!("What's the weather in Paris?")).await?;
//! assert_eq!(last_text(&state), "It's sunny in Paris.");
//! # Ok::<_, flow_agent::Error>(())
//! # }).unwrap();
//! ```
//!
//! | Module | Crate | What |
//! |---|---|---|
//! | [`core`] | `flow-agent-core` | `Runnable`, chaining, messages, prompts, parsers, tools, config, events, errors |
//! | [`models`] | `flow-agent-models` | `ChatModel`, OpenAI/Anthropic providers, `FakeChatModel` |
//! | [`graph`] | `flow-agent-graph` | `StateGraph`, reducers, execution, interrupts |
//! | [`checkpoint`] | `flow-agent-checkpoint` | Memory, SQLite and Postgres checkpointers |
//! | [`prebuilt`] | `flow-agent-prebuilt` | Ready-made ReAct agent |
//!
//! # Feature flags
//!
//! | Feature | Default | Adds |
//! |---|:---:|---|
//! | `anthropic` | yes | Claude models |
//! | `openai` | yes | OpenAI and compatible servers |
//! | `sqlite` | | `SqliteSaver` |
//! | `postgres` | | `PostgresSaver` |
//! | `schemars` | | `Tool::typed` |
//! | `full` | | all of the above |

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub use flow_agent_checkpoint as checkpoint;
pub use flow_agent_core as core;
pub use flow_agent_graph as graph;
pub use flow_agent_models as models;
pub use flow_agent_prebuilt as prebuilt;

pub use flow_agent_core::{Error, Result, Value, json};

/// Everything you need for typical use: `use flow_agent::prelude::*;`
pub mod prelude {
    pub use std::sync::Arc;

    pub use flow_agent_checkpoint::MemorySaver;
    #[cfg(feature = "postgres")]
    pub use flow_agent_checkpoint::PostgresSaver;
    #[cfg(feature = "sqlite")]
    pub use flow_agent_checkpoint::SqliteSaver;
    pub use flow_agent_core::prelude::*;
    pub use flow_agent_core::runnable::stream_owned;
    pub use flow_agent_core::{BoxRunnable, Checkpointer, ToolRegistry, ToolResult};
    pub use flow_agent_graph::Reducer;
    pub use flow_agent_graph::{
        Command, CompileOptions, CompiledGraph, END, NodeContext, NodeOptions, RunOutput, START,
        StateGraph, StateSnapshot, ToolNode, tools_condition,
    };
    pub use flow_agent_models::{ChatModel, FakeChatModel, ModelRegistry, ToolChoice, last_text};
    pub use flow_agent_prebuilt::{ReactAgent, ToolErrorMode, create_react_agent};
}
