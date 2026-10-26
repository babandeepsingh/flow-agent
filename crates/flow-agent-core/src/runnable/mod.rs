//! The [`Runnable`] trait and composition helpers.

mod combinators;
mod lambda;
mod stream;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::Value;

use crate::config::RunConfig;
use crate::error::Result;
use crate::event::Event;
use crate::retry::RetryPolicy;

pub use combinators::{
    RunnableParallel, RunnablePassthrough, RunnableSequence, WithConfig, WithFallbacks, WithRetry,
    WithTimeout,
};
pub use lambda::RunnableLambda;
pub use stream::{EventStream, stream_owned};

/// A cheap-to-clone, type-erased runnable.
pub type BoxRunnable = Arc<dyn Runnable>;

/// Anything that turns JSON input into JSON output: prompts, models, parsers,
/// tools, lambdas, chains and compiled graphs.
///
/// Only [`invoke_with`](Runnable::invoke_with) and [`name`](Runnable::name)
/// must be implemented; streaming and batching have sensible defaults.
///
/// ```
/// use flow_agent_core::{Runnable, RunConfig, Result};
/// use serde_json::{json, Value};
///
/// struct Shout;
///
/// #[async_trait::async_trait]
/// impl Runnable for Shout {
///     fn name(&self) -> &str { "shout" }
///     async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
///         Ok(json!(input.as_str().unwrap_or_default().to_uppercase()))
///     }
/// }
///
/// # tokio_test(async {
/// assert_eq!(Shout.invoke(json!("hi")).await.unwrap(), json!("HI"));
/// # });
/// # fn tokio_test<F: std::future::Future>(f: F) -> F::Output {
/// #     tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
/// # }
/// ```
#[async_trait]
pub trait Runnable: Send + Sync {
    /// Name used in traces and events.
    fn name(&self) -> &str;

    /// Run with an explicit config.
    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value>;

    /// Run with a default config.
    async fn invoke(&self, input: Value) -> Result<Value> {
        self.invoke_with(input, &RunConfig::default()).await
    }

    /// Stream events emitted during the run, ending with [`Event::Output`].
    ///
    /// The default runs [`invoke_with`](Runnable::invoke_with) and forwards
    /// everything nested components emit (model tokens, tool calls...).
    fn stream_with<'a>(&'a self, input: Value, config: &RunConfig) -> EventStream<'a> {
        stream::stream_invoke(self, input, config)
    }

    /// Stream with a default config.
    fn stream<'a>(&'a self, input: Value) -> EventStream<'a> {
        self.stream_with(input, &RunConfig::default())
    }

    /// Run many inputs concurrently (bounded by `config.max_concurrency`).
    /// Results keep input order; one failure doesn't abort the others.
    async fn batch_with(&self, inputs: Vec<Value>, config: &RunConfig) -> Vec<Result<Value>> {
        let limit = config.max_concurrency.max(1);
        futures::stream::iter(inputs)
            .map(|input| async move { self.invoke_with(input, &config.child()).await })
            .buffered(limit)
            .collect()
            .await
    }

    /// Batch with a default config.
    async fn batch(&self, inputs: Vec<Value>) -> Vec<Result<Value>> {
        self.batch_with(inputs, &RunConfig::default()).await
    }
}

#[async_trait]
impl<T: Runnable + ?Sized> Runnable for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        (**self).invoke_with(input, config).await
    }

    fn stream_with<'a>(&'a self, input: Value, config: &RunConfig) -> EventStream<'a> {
        (**self).stream_with(input, config)
    }

    async fn batch_with(&self, inputs: Vec<Value>, config: &RunConfig) -> Vec<Result<Value>> {
        (**self).batch_with(inputs, config).await
    }
}

/// Chaining helpers available on every [`Runnable`].
pub trait RunnableExt: Runnable + Sized + 'static {
    /// Feed this runnable's output into `next`.
    ///
    /// ```
    /// use flow_agent_core::{PromptTemplate, RunnableExt, RunnableLambda};
    /// use serde_json::json;
    ///
    /// let chain = PromptTemplate::new("Hello {name}")
    ///     .pipe(RunnableLambda::new("upper", |v| async move {
    ///         Ok(json!(v.as_str().unwrap_or_default().to_uppercase()))
    ///     }));
    /// ```
    fn pipe<R: Runnable + 'static>(self, next: R) -> RunnableSequence {
        RunnableSequence::new(vec![Arc::new(self), Arc::new(next)])
    }

    /// Retry failures according to `policy`.
    fn with_retry(self, policy: RetryPolicy) -> WithRetry {
        WithRetry::new(Arc::new(self), policy)
    }

    /// On error, try each fallback in order.
    fn with_fallbacks<I, R>(self, fallbacks: I) -> WithFallbacks
    where
        I: IntoIterator<Item = R>,
        R: Runnable + 'static,
    {
        WithFallbacks::new(
            Arc::new(self),
            fallbacks
                .into_iter()
                .map(|r| Arc::new(r) as BoxRunnable)
                .collect(),
        )
    }

    /// Fail with [`Error::Timeout`](crate::Error::Timeout) if a call takes longer than `timeout`.
    fn with_timeout(self, timeout: Duration) -> WithTimeout {
        WithTimeout::new(Arc::new(self), timeout)
    }

    /// Merge default tags, metadata and options into every call's config.
    fn with_config(self, config: RunConfig) -> WithConfig {
        WithConfig::new(Arc::new(self), config)
    }

    /// Type-erase into a [`BoxRunnable`].
    fn boxed(self) -> BoxRunnable {
        Arc::new(self)
    }
}

impl<T: Runnable + Sized + 'static> RunnableExt for T {}

/// Collect a stream into its final [`Event::Output`] value.
pub async fn collect_output(mut stream: BoxStream<'_, Result<Event>>) -> Result<Option<Value>> {
    let mut out = None;
    while let Some(ev) = stream.next().await {
        if let Event::Output { value } = ev? {
            out = Some(value);
        }
    }
    Ok(out)
}
