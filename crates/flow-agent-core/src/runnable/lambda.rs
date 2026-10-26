//! Wrapping async closures as runnables.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;

use super::Runnable;
use crate::config::RunConfig;
use crate::error::Result;

type LambdaFn = Arc<dyn Fn(Value, RunConfig) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// An async closure as a [`Runnable`].
///
/// ```
/// use flow_agent_core::RunnableLambda;
/// use serde_json::json;
///
/// let double = RunnableLambda::new("double", |v| async move {
///     Ok(json!(v.as_i64().unwrap_or(0) * 2))
/// });
/// ```
#[derive(Clone)]
pub struct RunnableLambda {
    name: String,
    f: LambdaFn,
}

impl fmt::Debug for RunnableLambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunnableLambda")
            .field("name", &self.name)
            .finish()
    }
}

impl RunnableLambda {
    /// Wrap `f(input)`.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        RunnableLambda {
            name: name.into(),
            f: Arc::new(move |v, _cfg| Box::pin(f(v))),
        }
    }

    /// Wrap `f(input, config)`, for closures that need the run config.
    pub fn with_config<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value, RunConfig) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        RunnableLambda {
            name: name.into(),
            f: Arc::new(move |v, cfg| Box::pin(f(v, cfg))),
        }
    }
}

#[async_trait]
impl Runnable for RunnableLambda {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        config.check_cancelled()?;
        (self.f)(input, config.clone()).await
    }
}
