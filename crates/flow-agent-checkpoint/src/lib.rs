//! Checkpointers for [flow-agent](https://github.com/babandeepsingh/flow-agent) graphs.
//!
//! A checkpointer saves the graph state after every step, keyed by
//! `RunConfig::thread_id`. That enables multi-turn conversations, resuming
//! after a crash or deploy, human-in-the-loop interrupts, state editing and
//! replay ("time travel").
//!
//! | Saver | Feature | Use for |
//! |---|---|---|
//! | [`MemorySaver`] | always | tests, prototypes, single process |
//! | `SqliteSaver` | `sqlite` | single-node services, CLIs, desktop apps |
//! | `PostgresSaver` | `postgres` | multi-instance production services |
//!
//! ```
//! use std::sync::Arc;
//! use flow_agent_checkpoint::MemorySaver;
//! use flow_agent_graph::{CompileOptions, StateGraph, NodeContext, START, END};
//! use flow_agent_core::{Runnable, RunConfig, Value, json};
//!
//! # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
//! let app = StateGraph::new()
//!     .add_node("count", |s: Value, _c: NodeContext| async move {
//!         Ok(json!({ "turns": s["turns"].as_i64().unwrap_or(0) + 1 }))
//!     })
//!     .add_edge(START, "count")
//!     .add_edge("count", END)
//!     .compile_with(CompileOptions::new().checkpointer(Arc::new(MemorySaver::new())))
//!     .unwrap();
//!
//! let config = RunConfig::new().thread_id("user-42");
//! app.invoke_with(json!({}), &config).await.unwrap();
//! let out = app.invoke_with(json!({}), &config).await.unwrap();
//! assert_eq!(out["turns"], 2); // state carried over between calls
//! # });
//! ```
//!
//! The [`Checkpointer`] trait itself lives in
//! `flow-agent-core`; [`testing::run_conformance`] checks custom backends.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod memory;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
mod sql;
pub mod testing;

pub use flow_agent_core::{Checkpoint, Checkpointer, ListFilter, PendingWrite};
pub use memory::MemorySaver;
#[cfg(feature = "postgres")]
#[cfg_attr(docsrs, doc(cfg(feature = "postgres")))]
pub use sql::PostgresSaver;
#[cfg(feature = "sqlite")]
#[cfg_attr(docsrs, doc(cfg(feature = "sqlite")))]
pub use sql::SqliteSaver;
