//! Chat models for [flow-agent](https://github.com/babandeepsingh/flow-agent).
//!
//! [`ChatModel`] gives one API over every provider: build it from a
//! `"provider:model"` string, call it like any [`Runnable`](flow_agent_core::Runnable),
//! bind tools, request structured output, and stream tokens. Provider
//! differences (wire formats, streaming events, tool-call encoding) are
//! normalized to flow-agent's [`Message`](flow_agent_core::Message) and
//! [`ToolCall`](flow_agent_core::ToolCall).
//!
//! ```no_run
//! # async fn demo() -> flow_agent_core::Result<()> {
//! use flow_agent_core::{Runnable, json};
//! use flow_agent_models::ChatModel;
//!
//! // The library never reads env vars; your application passes the key.
//! let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
//! let model = ChatModel::builder("anthropic:claude-sonnet-5-5")
//!     .api_key(key)
//!     .build()?;
//! let reply = model.invoke(json!("Hi!")).await?;
//! # Ok(()) }
//! ```
//!
//! For tests, use [`FakeChatModel`]: scripted responses, no network.
//!
//! # Feature flags
//!
//! | Feature | Provider prefix |
//! |---|---|
//! | `openai` | `openai:` (also OpenAI-compatible servers via `base_url`) |
//! | `anthropic` | `anthropic:` |
//!
//! Both are off by default in this crate and on by default in the `flow-agent`
//! umbrella crate.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod fake;
mod model;
mod provider;
mod registry;

#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub mod anthropic;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub mod openai;

pub use fake::FakeChatModel;
pub use model::{ChatModel, ChatModelBuilder, DEFAULT_MAX_RETRIES, last_text};
pub use provider::{
    ChatChunk, ChatProvider, ChatRequest, ChunkAggregator, ChunkStream, FinishReason, ModelOptions,
    ToolCallChunk, ToolChoice, message_to_chunk,
};
pub use registry::{ModelRegistry, ProviderConfig, ProviderFactory, parse_spec};
