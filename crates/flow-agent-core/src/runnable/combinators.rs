//! Sequence, parallel, passthrough, retry, fallbacks, timeout and config wrappers.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value};
use tracing::Instrument;

use super::{BoxRunnable, Runnable};
use crate::config::RunConfig;
use crate::error::{Error, Result};
use crate::message::type_name;
use crate::retry::RetryPolicy;

/// Runs steps one after another, feeding each output into the next step.
/// Created by [`RunnableExt::pipe`](super::RunnableExt::pipe).
#[derive(Clone)]
pub struct RunnableSequence {
    name: String,
    steps: Vec<BoxRunnable>,
}

impl std::fmt::Debug for RunnableSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.steps.iter().map(|s| s.name()).collect();
        f.debug_struct("RunnableSequence")
            .field("steps", &names)
            .finish()
    }
}

impl RunnableSequence {
    /// A sequence of the given steps.
    pub fn new(steps: Vec<BoxRunnable>) -> Self {
        RunnableSequence {
            name: "sequence".into(),
            steps,
        }
    }

    /// Append another step. Shadows [`RunnableExt::pipe`](super::RunnableExt::pipe)
    /// so long chains stay flat.
    pub fn pipe<R: Runnable + 'static>(mut self, next: R) -> Self {
        self.steps.push(Arc::new(next));
        self
    }

    /// Set the name used in traces.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The steps.
    pub fn steps(&self) -> &[BoxRunnable] {
        &self.steps
    }
}

#[async_trait]
impl Runnable for RunnableSequence {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let mut value = input;
        for (i, step) in self.steps.iter().enumerate() {
            config.check_cancelled()?;
            let span = tracing::debug_span!("flow_agent.step", index = i, step = step.name());
            value = step
                .invoke_with(value, config)
                .instrument(span)
                .await
                .map_err(|e| e.context(format!("step {i} ('{}')", step.name())))?;
        }
        Ok(value)
    }
}

/// Runs branches concurrently on the same input; output is `{"branch": output, ...}`.
///
/// ```
/// use flow_agent_core::{RunnableLambda, RunnableParallel};
/// use serde_json::json;
///
/// let both = RunnableParallel::new()
///     .branch("same", RunnableLambda::new("id", |v| async move { Ok(v) }))
///     .branch("len", RunnableLambda::new("len", |v| async move {
///         Ok(json!(v.as_str().map(str::len).unwrap_or(0)))
///     }));
/// ```
#[derive(Clone, Default)]
pub struct RunnableParallel {
    branches: Vec<(String, BoxRunnable)>,
}

impl RunnableParallel {
    /// No branches yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named branch.
    pub fn branch<R: Runnable + 'static>(mut self, key: impl Into<String>, runnable: R) -> Self {
        self.branches.push((key.into(), Arc::new(runnable)));
        self
    }
}

#[async_trait]
impl Runnable for RunnableParallel {
    fn name(&self) -> &str {
        "parallel"
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        config.check_cancelled()?;
        let futs = self.branches.iter().map(|(key, r)| {
            let input = input.clone();
            async move {
                r.invoke_with(input, config)
                    .await
                    .map(|v| (key.clone(), v))
                    .map_err(|e| e.context(format!("branch '{key}'")))
            }
        });
        let results = futures::future::try_join_all(futs).await?;
        Ok(Value::Object(results.into_iter().collect()))
    }
}

/// Passes input through unchanged, optionally adding computed keys.
///
/// ```
/// use flow_agent_core::{RunnableLambda, RunnablePassthrough};
/// use serde_json::json;
///
/// // {"q": "hi"} -> {"q": "hi", "upper": "HI"}
/// let p = RunnablePassthrough::new().assign("upper", RunnableLambda::new("up", |v| async move {
///     Ok(json!(v["q"].as_str().unwrap_or_default().to_uppercase()))
/// }));
/// ```
#[derive(Clone, Default)]
pub struct RunnablePassthrough {
    assign: Vec<(String, BoxRunnable)>,
}

impl RunnablePassthrough {
    /// Identity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `key` computed by `runnable` from the (object) input.
    pub fn assign<R: Runnable + 'static>(mut self, key: impl Into<String>, runnable: R) -> Self {
        self.assign.push((key.into(), Arc::new(runnable)));
        self
    }
}

#[async_trait]
impl Runnable for RunnablePassthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        if self.assign.is_empty() {
            return Ok(input);
        }
        let Value::Object(mut map) = input else {
            return Err(Error::InvalidInput(format!(
                "RunnablePassthrough::assign needs an object input, got {}",
                type_name(&input)
            )));
        };
        let snapshot = Value::Object(map.clone());
        let futs = self.assign.iter().map(|(key, r)| {
            let input = snapshot.clone();
            async move { r.invoke_with(input, config).await.map(|v| (key.clone(), v)) }
        });
        let results: Map<String, Value> = futures::future::try_join_all(futs)
            .await?
            .into_iter()
            .collect();
        map.extend(results);
        Ok(Value::Object(map))
    }
}

/// Retries the inner runnable. Created by [`RunnableExt::with_retry`](super::RunnableExt::with_retry).
#[derive(Clone)]
pub struct WithRetry {
    inner: BoxRunnable,
    policy: RetryPolicy,
}

impl WithRetry {
    /// Wrap `inner`.
    pub fn new(inner: BoxRunnable, policy: RetryPolicy) -> Self {
        WithRetry { inner, policy }
    }
}

#[async_trait]
impl Runnable for WithRetry {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        self.policy
            .run(|_attempt| {
                let input = input.clone();
                async move {
                    config.check_cancelled()?;
                    self.inner.invoke_with(input, config).await
                }
            })
            .await
    }
}

/// Tries fallbacks in order when the primary fails.
/// Created by [`RunnableExt::with_fallbacks`](super::RunnableExt::with_fallbacks).
#[derive(Clone)]
pub struct WithFallbacks {
    primary: BoxRunnable,
    fallbacks: Vec<BoxRunnable>,
}

impl WithFallbacks {
    /// Wrap `primary`.
    pub fn new(primary: BoxRunnable, fallbacks: Vec<BoxRunnable>) -> Self {
        WithFallbacks { primary, fallbacks }
    }
}

#[async_trait]
impl Runnable for WithFallbacks {
    fn name(&self) -> &str {
        self.primary.name()
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let mut last = match self.primary.invoke_with(input.clone(), config).await {
            Ok(v) => return Ok(v),
            Err(e) => e,
        };
        for fb in &self.fallbacks {
            if matches!(last.root(), Error::Cancelled) {
                break;
            }
            tracing::debug!(error = %last, fallback = fb.name(), "trying fallback");
            match fb.invoke_with(input.clone(), config).await {
                Ok(v) => return Ok(v),
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

/// Fails with [`Error::Timeout`] when the inner call is too slow.
/// Created by [`RunnableExt::with_timeout`](super::RunnableExt::with_timeout).
#[derive(Clone)]
pub struct WithTimeout {
    inner: BoxRunnable,
    timeout: Duration,
}

impl WithTimeout {
    /// Wrap `inner`.
    pub fn new(inner: BoxRunnable, timeout: Duration) -> Self {
        WithTimeout { inner, timeout }
    }
}

#[async_trait]
impl Runnable for WithTimeout {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        tokio::time::timeout(self.timeout, self.inner.invoke_with(input, config))
            .await
            .map_err(|_| Error::Timeout {
                what: format!("'{}'", self.inner.name()),
                after: self.timeout,
            })?
    }
}

/// Merges default config into every call.
/// Created by [`RunnableExt::with_config`](super::RunnableExt::with_config).
#[derive(Clone)]
pub struct WithConfig {
    inner: BoxRunnable,
    defaults: RunConfig,
}

impl WithConfig {
    /// Wrap `inner`.
    pub fn new(inner: BoxRunnable, defaults: RunConfig) -> Self {
        WithConfig { inner, defaults }
    }
}

#[async_trait]
impl Runnable for WithConfig {
    fn name(&self) -> &str {
        self.defaults
            .run_name
            .as_deref()
            .unwrap_or_else(|| self.inner.name())
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let merged = config.merged_with(&self.defaults);
        let fut = self.inner.invoke_with(input, &merged);
        match merged.timeout {
            Some(t) if config.timeout.is_none() => {
                tokio::time::timeout(t, fut)
                    .await
                    .map_err(|_| Error::Timeout {
                        what: format!("'{}'", self.inner.name()),
                        after: t,
                    })?
            }
            _ => fut.await,
        }
    }
}
