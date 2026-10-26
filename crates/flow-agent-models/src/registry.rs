//! Resolving `"provider:model"` strings to providers.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use flow_agent_core::{Error, Result};

use crate::provider::ChatProvider;

/// Connection settings handed to a provider factory.
///
/// API keys are always passed explicitly: flow-agent never reads environment
/// variables.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct ProviderConfig {
    /// Model name (the part after `provider:`).
    pub model: String,
    /// API key.
    pub api_key: Option<String>,
    /// Override the API base URL (proxies, gateways, OpenAI-compatible servers).
    pub base_url: Option<String>,
    /// Extra HTTP headers.
    pub headers: Vec<(String, String)>,
    /// Per-request HTTP timeout (connect + idle read). Default 120s.
    pub request_timeout: Option<Duration>,
    /// Use this HTTP client instead of building one.
    #[cfg(feature = "http")]
    pub http_client: Option<reqwest::Client>,
}

impl ProviderConfig {
    /// Config for `model` with everything else unset.
    pub fn new(model: impl Into<String>) -> Self {
        ProviderConfig {
            model: model.into(),
            ..Default::default()
        }
    }
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field(
                "headers",
                &self.headers.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

/// Builds a provider from its config.
pub type ProviderFactory =
    Arc<dyn Fn(ProviderConfig) -> Result<Arc<dyn ChatProvider>> + Send + Sync>;

/// Maps provider prefixes (`"openai"`, `"anthropic"`, your own) to factories.
///
/// This is a value you own and pass around, not a global.
/// [`ModelRegistry::default`] contains the built-in providers enabled by
/// feature flags.
///
/// ```
/// use std::sync::Arc;
/// use flow_agent_core::Message;
/// use flow_agent_models::{ChatModel, FakeChatModel, ModelRegistry};
///
/// let registry = ModelRegistry::default().register("mock", |cfg| {
///     Ok(Arc::new(FakeChatModel::echo().with_model(cfg.model)))
/// });
/// let model = ChatModel::builder("mock:any").registry(&registry).build().unwrap();
/// assert_eq!(model.spec(), "fake:any");
/// ```
#[derive(Clone)]
pub struct ModelRegistry {
    factories: BTreeMap<String, ProviderFactory>,
}

impl fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("providers", &self.providers())
            .finish()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        #[allow(unused_mut)]
        let mut r = Self::empty();
        #[cfg(feature = "openai")]
        {
            r = r.register("openai", |cfg| {
                Ok(Arc::new(crate::openai::OpenAiProvider::new(cfg)?) as Arc<dyn ChatProvider>)
            });
        }
        #[cfg(feature = "anthropic")]
        {
            r = r.register("anthropic", |cfg| {
                Ok(Arc::new(crate::anthropic::AnthropicProvider::new(cfg)?)
                    as Arc<dyn ChatProvider>)
            });
        }
        r
    }
}

impl ModelRegistry {
    /// A registry with no providers.
    pub fn empty() -> Self {
        ModelRegistry {
            factories: BTreeMap::new(),
        }
    }

    /// Register (or replace) a provider factory.
    pub fn register<F>(mut self, name: impl Into<String>, factory: F) -> Self
    where
        F: Fn(ProviderConfig) -> Result<Arc<dyn ChatProvider>> + Send + Sync + 'static,
    {
        self.factories.insert(name.into(), Arc::new(factory));
        self
    }

    /// Registered provider names.
    pub fn providers(&self) -> Vec<&str> {
        self.factories.keys().map(String::as_str).collect()
    }

    /// Create a provider for `provider` with `config`.
    pub fn create(&self, provider: &str, config: ProviderConfig) -> Result<Arc<dyn ChatProvider>> {
        match self.factories.get(provider) {
            Some(f) => f(config),
            None => Err(Error::Config(unknown_provider_message(
                provider,
                &self.providers(),
            ))),
        }
    }
}

/// Split `"provider:model"`. The model part may itself contain `:`.
pub fn parse_spec(spec: &str) -> Result<(&str, &str)> {
    match spec.split_once(':') {
        Some((p, m)) if !p.is_empty() && !m.is_empty() => Ok((p, m)),
        _ => Err(Error::Config(format!(
            "invalid model spec '{spec}': expected \"provider:model\", e.g. \"anthropic:claude-sonnet-5-5\""
        ))),
    }
}

fn unknown_provider_message(provider: &str, known: &[&str]) -> String {
    let hint = match provider {
        "openai" | "anthropic" => {
            format!(" Enable the `{provider}` feature of flow-agent (or flow-agent-models).")
        }
        _ => String::from(" Register it with ModelRegistry::register."),
    };
    format!(
        "unknown model provider '{provider}' (available: {}).{hint}",
        if known.is_empty() {
            "none".to_string()
        } else {
            known.join(", ")
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_parsing() {
        assert_eq!(
            parse_spec("openai:llama3.1:8b").unwrap(),
            ("openai", "llama3.1:8b")
        );
        assert!(parse_spec("claude").is_err());
        assert!(parse_spec(":x").is_err());
        assert!(parse_spec("x:").is_err());
    }

    #[test]
    fn unknown_provider_has_hint() {
        let err = ModelRegistry::empty()
            .create("openai", ProviderConfig::new("m"))
            .err()
            .unwrap();
        assert!(err.to_string().contains("feature"), "{err}");
        let err = ModelRegistry::empty()
            .create("mine", ProviderConfig::new("m"))
            .err()
            .unwrap();
        assert!(err.to_string().contains("register"), "{err}");
    }

    #[test]
    fn debug_redacts_key() {
        let mut c = ProviderConfig::new("m");
        c.api_key = Some("sk-secret".into());
        assert!(!format!("{c:?}").contains("sk-secret"));
    }
}
