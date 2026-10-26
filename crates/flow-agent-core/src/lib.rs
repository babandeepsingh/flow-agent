//! Core building blocks for [flow-agent](https://github.com/babandeepsingh/flow-agent).
//!
//! Everything executable is a [`Runnable`]: JSON ([`serde_json::Value`]) in,
//! JSON out. Runnables chain with [`RunnableExt::pipe`], stream [`Event`]s and
//! receive a [`RunConfig`] carrying thread id, tags, callbacks and cancellation.
//!
//! ```
//! use flow_agent_core::prelude::*;
//! use serde_json::json;
//!
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! let chain = PromptTemplate::new("Hello, {name}!")
//!     .pipe(RunnableLambda::new("shout", |v| async move {
//!         Ok(json!(v.as_str().unwrap_or_default().to_uppercase()))
//!     }));
//!
//! assert_eq!(chain.invoke(json!({"name": "crab"})).await?, json!("HELLO, CRAB!"));
//! # Ok::<_, flow_agent_core::Error>(())
//! # }).unwrap();
//! ```
//!
//! Most users depend on the `flow-agent` umbrella crate instead of this one.
//!
//! # Feature flags
//!
//! - `schemars`: [`Tool::typed`] with JSON Schema derived from a Rust type.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod checkpoint;
pub mod config;
pub mod error;
pub mod event;
pub mod message;
pub mod parser;
pub mod prompt;
pub mod retry;
pub mod runnable;
pub mod tool;

pub use checkpoint::{Checkpoint, Checkpointer, Interrupt, ListFilter, PendingWrite};
pub use config::RunConfig;
pub use error::{Error, Result, ResultExt};
pub use event::{CallbackHandler, Event, EventKind};
pub use message::{Content, ContentPart, ImageUrl, Message, Role, ToolCall, Usage};
pub use parser::{JsonOutputParser, StrOutputParser, ToolCallsParser};
pub use prompt::{ChatPromptTemplate, PromptTemplate};
pub use retry::RetryPolicy;
pub use runnable::{
    BoxRunnable, EventStream, Runnable, RunnableExt, RunnableLambda, RunnableParallel,
    RunnablePassthrough, RunnableSequence,
};
pub use tool::{Tool, ToolDefinition, ToolRegistry, ToolResult};

/// Re-export so users can write `Value` and `json!` without adding serde_json.
pub use serde_json::{self, Value, json};

/// The most commonly used items.
pub mod prelude {
    pub use crate::{
        CallbackHandler, ChatPromptTemplate, Error, Event, EventKind, JsonOutputParser, Message,
        PromptTemplate, Result, ResultExt, RetryPolicy, Role, RunConfig, Runnable, RunnableExt,
        RunnableLambda, RunnableParallel, RunnablePassthrough, StrOutputParser, Tool, ToolCall,
        ToolRegistry, Value, json,
    };
}
