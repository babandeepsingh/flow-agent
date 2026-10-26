//! Retry policy with exponential backoff and jitter.

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{Error, Result};

type RetryPredicate = Arc<dyn Fn(&Error) -> bool + Send + Sync>;

/// How to retry a failing operation.
///
/// ```
/// use flow_agent_core::RetryPolicy;
/// use std::time::Duration;
///
/// let policy = RetryPolicy::exponential(3)
///     .initial(Duration::from_millis(200))
///     .max(Duration::from_secs(5));
/// assert_eq!(policy.max_attempts, 3);
/// ```
///
/// By default only errors where [`Error::is_retryable`] is true are retried.
#[derive(Clone)]
pub struct RetryPolicy {
    /// Total attempts including the first (minimum 1).
    pub max_attempts: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Upper bound for any delay.
    pub max_delay: Duration,
    /// Delay multiplier per attempt.
    pub multiplier: f64,
    /// Randomize delays (±50%) to avoid thundering herds.
    pub jitter: bool,
    retry_on: Option<RetryPredicate>,
}

impl fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("multiplier", &self.multiplier)
            .field("jitter", &self.jitter)
            .field("custom_predicate", &self.retry_on.is_some())
            .finish()
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::exponential(3)
    }
}

impl RetryPolicy {
    /// Exponential backoff: 500ms, 1s, 2s... capped at 30s, with jitter.
    pub fn exponential(max_attempts: u32) -> Self {
        RetryPolicy {
            max_attempts: max_attempts.max(1),
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: true,
            retry_on: None,
        }
    }

    /// Never retry.
    pub fn none() -> Self {
        Self::exponential(1)
    }

    /// Set the first retry delay.
    pub fn initial(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay.
    pub fn max(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the delay multiplier.
    pub fn multiplier(mut self, m: f64) -> Self {
        self.multiplier = m.max(1.0);
        self
    }

    /// Enable or disable jitter.
    pub fn jitter(mut self, on: bool) -> Self {
        self.jitter = on;
        self
    }

    /// Decide which errors are retried (replaces the default [`Error::is_retryable`]).
    pub fn retry_if(mut self, f: impl Fn(&Error) -> bool + Send + Sync + 'static) -> Self {
        self.retry_on = Some(Arc::new(f));
        self
    }

    /// Whether `err` should be retried.
    pub fn should_retry(&self, err: &Error) -> bool {
        if matches!(err.root(), Error::Cancelled) {
            return false;
        }
        match &self.retry_on {
            Some(f) => f(err),
            None => err.is_retryable(),
        }
    }

    /// Delay before retry number `retry` (1-based).
    pub fn delay_for(&self, retry: u32) -> Duration {
        let exp = self.multiplier.powi(retry.saturating_sub(1).min(64) as i32);
        let base = self.initial_delay.as_secs_f64() * exp;
        let capped = base.min(self.max_delay.as_secs_f64());
        let secs = if self.jitter {
            capped * (0.5 + random_unit())
        } else {
            capped
        };
        Duration::from_secs_f64(secs.min(self.max_delay.as_secs_f64()).max(0.0))
    }

    /// Run `op` until it succeeds, a non-retryable error occurs, or attempts run out.
    /// `op` receives the 1-based attempt number.
    pub async fn run<T, F, Fut>(&self, mut op: F) -> Result<T>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut attempt = 1;
        loop {
            match op(attempt).await {
                Ok(v) => return Ok(v),
                Err(e) if attempt < self.max_attempts && self.should_retry(&e) => {
                    let delay = self.delay_for(attempt);
                    tracing::debug!(attempt, ?delay, error = %e, "retrying");
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// A random number in `[0, 1)`, using the OS RNG behind `uuid` v4.
fn random_unit() -> f64 {
    let bits = uuid::Uuid::new_v4().as_u128() as u64 >> 11;
    bits as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn retryable() -> Error {
        Error::Http("boom".into())
    }

    #[test]
    fn delays_grow_and_cap() {
        let p = RetryPolicy::exponential(5)
            .initial(Duration::from_millis(100))
            .max(Duration::from_millis(300))
            .jitter(false);
        assert_eq!(p.delay_for(1), Duration::from_millis(100));
        assert_eq!(p.delay_for(2), Duration::from_millis(200));
        assert_eq!(p.delay_for(3), Duration::from_millis(300));
        assert_eq!(p.delay_for(10), Duration::from_millis(300));
    }

    #[test]
    fn jitter_stays_in_bounds() {
        let p = RetryPolicy::exponential(5).initial(Duration::from_millis(100));
        for _ in 0..100 {
            let d = p.delay_for(1);
            assert!(d >= Duration::from_millis(50) && d <= Duration::from_millis(150));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retries_until_success() {
        let calls = AtomicU32::new(0);
        let out = RetryPolicy::exponential(3)
            .run(|_| async {
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(retryable())
                } else {
                    Ok(7)
                }
            })
            .await
            .unwrap();
        assert_eq!(out, 7);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn stops_on_non_retryable_and_exhaustion() {
        let calls = AtomicU32::new(0);
        let r: Result<()> = RetryPolicy::exponential(5)
            .run(|_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Error::InvalidInput("no".into()))
            })
            .await;
        assert!(r.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let calls = AtomicU32::new(0);
        let r: Result<()> = RetryPolicy::exponential(2)
            .run(|_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(retryable())
            })
            .await;
        assert!(r.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn custom_predicate() {
        let calls = AtomicU32::new(0);
        let _ = RetryPolicy::exponential(3)
            .retry_if(|e| matches!(e, Error::InvalidInput(_)))
            .run(|_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(Error::InvalidInput("x".into()))
            })
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}
