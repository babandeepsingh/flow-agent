//! Tools the model can call, and a registry to execute them.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::Instrument;
use uuid::Uuid;

use crate::config::RunConfig;
use crate::error::{Error, Result};
use crate::event::Event;
use crate::message::{Message, ToolCall, type_name};
use crate::parser::check_schema;
use crate::runnable::Runnable;

type Handler = Arc<dyn Fn(Value, RunConfig) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// What a model sees about a tool: name, description and JSON Schema of the arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// What the tool does (shown to the model).
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
}

/// A function the model can call.
///
/// ```
/// use flow_agent_core::Tool;
/// use serde_json::json;
///
/// let weather = Tool::new("get_weather", "Get the current weather for a city")
///     .param("city", "string", "City name, e.g. 'Paris'")
///     .optional_param("unit", "string", "celsius or fahrenheit")
///     .handler(|args| async move {
///         Ok(json!({ "city": args["city"], "forecast": "sunny" }))
///     });
/// assert_eq!(weather.definition().parameters["required"], json!(["city"]));
/// ```
#[derive(Clone)]
pub struct Tool {
    name: String,
    description: String,
    schema: Value,
    handler: Option<Handler>,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("schema", &self.schema)
            .field("has_handler", &self.handler.is_some())
            .finish()
    }
}

impl Tool {
    /// A tool with no parameters and no handler yet.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Tool {
            name: name.into(),
            description: description.into(),
            schema: json!({ "type": "object", "properties": {}, "required": [] }),
            handler: None,
        }
    }

    fn add_param(mut self, name: String, ty: &str, description: &str, required: bool) -> Self {
        if !self.schema.is_object() {
            self.schema = json!({ "type": "object" });
        }
        let schema = &mut self.schema;
        if !schema["properties"].is_object() {
            schema["properties"] = json!({});
        }
        schema["properties"][&name] = json!({ "type": ty, "description": description });
        if !schema["required"].is_array() {
            schema["required"] = json!([]);
        }
        if let Value::Array(req) = &mut schema["required"] {
            req.retain(|v| v.as_str() != Some(name.as_str()));
            if required {
                req.push(Value::String(name));
            }
        }
        self
    }

    /// Add a required parameter. `ty` is a JSON Schema type: `string`,
    /// `number`, `integer`, `boolean`, `array` or `object`.
    pub fn param(self, name: impl Into<String>, ty: &str, description: &str) -> Self {
        self.add_param(name.into(), ty, description, true)
    }

    /// Add an optional parameter.
    pub fn optional_param(self, name: impl Into<String>, ty: &str, description: &str) -> Self {
        self.add_param(name.into(), ty, description, false)
    }

    /// Replace the argument schema with a full JSON Schema object.
    pub fn input_schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    /// Set the async function that runs the tool. It receives the arguments object.
    pub fn handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        self.handler = Some(Arc::new(move |args, _cfg| Box::pin(f(args))));
        self
    }

    /// Like [`Tool::handler`], but the function also receives the [`RunConfig`]
    /// (thread id, metadata such as the current user...).
    pub fn handler_with_config<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value, RunConfig) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        self.handler = Some(Arc::new(move |args, cfg| Box::pin(f(args, cfg))));
        self
    }

    /// A tool with typed arguments; the JSON Schema is derived with `schemars`.
    ///
    /// ```
    /// use flow_agent_core::Tool;
    ///
    /// #[derive(serde::Deserialize, schemars::JsonSchema)]
    /// struct Add { a: i64, b: i64 }
    ///
    /// let add = Tool::typed("add", "Add two numbers", |x: Add| async move { Ok(x.a + x.b) });
    /// ```
    #[cfg(feature = "schemars")]
    #[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
    pub fn typed<A, R, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        A: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
        R: Serialize,
        F: Fn(A) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R>> + Send + 'static,
    {
        let name = name.into();
        let mut schema = serde_json::to_value(schemars::schema_for!(A)).unwrap_or(json!({}));
        if let Value::Object(map) = &mut schema {
            map.remove("$schema");
            map.remove("title");
        }
        let f = Arc::new(f);
        let tool_name = name.clone();
        Tool::new(name, description)
            .input_schema(schema)
            .handler(move |args| {
                let f = Arc::clone(&f);
                let tool_name = tool_name.clone();
                async move {
                    let typed: A =
                        serde_json::from_value(args).map_err(|e| Error::InvalidToolArgs {
                            tool: tool_name,
                            message: e.to_string(),
                        })?;
                    Ok(serde_json::to_value(f(typed).await?)?)
                }
            })
    }

    /// Tool name.
    pub fn tool_name(&self) -> &str {
        &self.name
    }

    /// Tool description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Argument schema.
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// The definition sent to models.
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.schema.clone(),
        }
    }

    /// Normalize and validate arguments against the schema.
    fn prepare_args(&self, args: Value) -> Result<Value> {
        let invalid = |message: String| Error::InvalidToolArgs {
            tool: self.name.clone(),
            message,
        };
        // Some models send arguments as a JSON-encoded string.
        let args = match args {
            Value::String(s) if !s.trim().is_empty() => serde_json::from_str(&s)
                .map_err(|e| invalid(format!("arguments are not JSON: {e}")))?,
            Value::Null => json!({}),
            Value::String(_) => json!({}),
            other => other,
        };
        if self.schema.get("type").and_then(Value::as_str) == Some("object") && !args.is_object() {
            return Err(invalid(format!(
                "expected an object, got {}",
                type_name(&args)
            )));
        }
        check_schema(&args, &self.schema).map_err(invalid)?;
        Ok(args)
    }

    /// Validate `args` and run the handler, with a tracing span.
    pub async fn call(&self, args: Value, config: &RunConfig) -> Result<Value> {
        let handler = self
            .handler
            .as_ref()
            .ok_or_else(|| Error::Config(format!("tool '{}' has no handler", self.name)))?;
        let args = self.prepare_args(args)?;
        config.check_cancelled()?;
        handler(args, config.clone()).await.map_err(|e| match e {
            e @ (Error::Tool { .. } | Error::InvalidToolArgs { .. } | Error::Cancelled) => e,
            other => other.context(format!("tool '{}'", self.name)),
        })
    }
}

#[async_trait]
impl Runnable for Tool {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        let span = tracing::info_span!("flow_agent.tool", tool = %self.name, call_id = tracing::field::Empty, is_error = tracing::field::Empty);
        let res = self.call(input, config).instrument(span.clone()).await;
        span.record("is_error", res.is_err());
        res
    }
}

/// The outcome of one tool call. Errors are captured, never panicked on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call id this answers.
    pub call_id: String,
    /// Tool name.
    pub name: String,
    /// Output, or `{"error": "..."}` when `is_error`.
    pub output: Value,
    /// Whether the call failed.
    pub is_error: bool,
}

impl ToolResult {
    /// A `tool` message to send back to the model.
    pub fn to_message(&self) -> Message {
        let content = if self.is_error {
            Value::String(format!(
                "Error: {}",
                self.output
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("tool failed")
            ))
        } else {
            self.output.clone()
        };
        let mut m = Message::tool(self.call_id.clone(), content).with_name(self.name.clone());
        if self.is_error {
            m.metadata.insert("is_error".into(), Value::Bool(true));
        }
        m
    }
}

/// A set of tools addressed by name.
///
/// ```
/// use flow_agent_core::{Tool, ToolRegistry};
/// let tools = ToolRegistry::new()
///     .with(Tool::new("a", "first"))
///     .with(Tool::new("b", "second"));
/// assert_eq!(tools.names(), vec!["a", "b"]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<Tool>>,
    index: HashMap<String, usize>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool (builder style). A tool with the same name is replaced.
    pub fn with(mut self, tool: Tool) -> Self {
        self.add(tool);
        self
    }

    /// Add a tool. A tool with the same name is replaced.
    pub fn add(&mut self, tool: Tool) {
        let tool = Arc::new(tool);
        match self.index.get(tool.tool_name()) {
            Some(&i) => self.tools[i] = tool,
            None => {
                self.index
                    .insert(tool.tool_name().to_owned(), self.tools.len());
                self.tools.push(tool);
            }
        }
    }

    /// Look up a tool.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.index.get(name).map(|&i| self.tools[i].as_ref())
    }

    /// Tool names in insertion order.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.tool_name()).collect()
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether there are no tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Definitions for all tools, to pass to a model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Execute one tool call, emitting `ToolStart`/`ToolEnd` events. Never fails:
    /// errors (unknown tool, bad args, handler error) become an error [`ToolResult`].
    pub async fn call(&self, call: &ToolCall, config: &RunConfig) -> ToolResult {
        let run_id = Uuid::now_v7();
        config
            .emit(Event::ToolStart {
                run_id,
                node: config.node.clone(),
                tool: call.name.clone(),
                call_id: call.id.clone(),
                args: call.args.clone(),
            })
            .await;

        let span = tracing::info_span!(
            "flow_agent.tool",
            tool = %call.name,
            call_id = %call.id,
            is_error = tracing::field::Empty,
        );
        let res = match self.get(&call.name) {
            Some(tool) => {
                tool.call(call.args.clone(), config)
                    .instrument(span.clone())
                    .await
            }
            None => Err(Error::ToolNotFound(call.name.clone())),
        };
        span.record("is_error", res.is_err());

        let result = match res {
            Ok(output) => ToolResult {
                call_id: call.id.clone(),
                name: call.name.clone(),
                output,
                is_error: false,
            },
            Err(e) => {
                tracing::warn!(parent: &span, error = %e, "tool call failed");
                config.emit_error(&e).await;
                ToolResult {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    output: json!({ "error": e.to_string() }),
                    is_error: true,
                }
            }
        };

        config
            .emit(Event::ToolEnd {
                run_id,
                node: config.node.clone(),
                tool: result.name.clone(),
                call_id: result.call_id.clone(),
                output: result.output.clone(),
                is_error: result.is_error,
            })
            .await;
        result
    }

    /// Execute several calls concurrently (bounded by `config.max_concurrency`).
    /// Results keep the order of `calls`.
    pub async fn call_many(&self, calls: &[ToolCall], config: &RunConfig) -> Vec<ToolResult> {
        futures::stream::iter(calls)
            .map(|c| self.call(c, config))
            .buffered(config.max_concurrency.max(1))
            .collect()
            .await
    }
}

impl FromIterator<Tool> for ToolRegistry {
    fn from_iter<I: IntoIterator<Item = Tool>>(iter: I) -> Self {
        let mut r = ToolRegistry::new();
        for t in iter {
            r.add(t);
        }
        r
    }
}

impl From<Vec<Tool>> for ToolRegistry {
    fn from(tools: Vec<Tool>) -> Self {
        tools.into_iter().collect()
    }
}

impl<const N: usize> From<[Tool; N]> for ToolRegistry {
    fn from(tools: [Tool; N]) -> Self {
        tools.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo() -> Tool {
        Tool::new("echo", "Echo text")
            .param("text", "string", "Text")
            .handler(|args| async move { Ok(args["text"].clone()) })
    }

    #[tokio::test]
    async fn validates_args() {
        let t = echo();
        assert_eq!(t.invoke(json!({"text": "hi"})).await.unwrap(), json!("hi"));
        assert_eq!(
            t.invoke(json!(r#"{"text": "s"}"#)).await.unwrap(),
            json!("s")
        );
        assert!(matches!(
            t.invoke(json!({})).await.unwrap_err(),
            Error::InvalidToolArgs { .. }
        ));
        assert!(matches!(
            t.invoke(json!({"text": 3})).await.unwrap_err(),
            Error::InvalidToolArgs { .. }
        ));
        assert!(t.invoke(json!([1])).await.is_err());
    }

    #[tokio::test]
    async fn missing_handler_is_error() {
        assert!(matches!(
            Tool::new("x", "y").invoke(json!({})).await.unwrap_err(),
            Error::Config(_)
        ));
    }

    #[test]
    fn param_replaces_and_optional() {
        let t = Tool::new("t", "d")
            .param("a", "string", "")
            .optional_param("a", "integer", "");
        assert_eq!(t.schema()["required"], json!([]));
        assert_eq!(t.schema()["properties"]["a"]["type"], "integer");
    }

    #[tokio::test]
    async fn registry_calls_and_captures_errors() {
        let failing = Tool::new("fail", "Always fails")
            .handler(|_| async { Err(Error::tool("fail", "nope")) });
        let reg = ToolRegistry::from([echo(), failing]);
        let calls = vec![
            ToolCall::new("echo", json!({"text": "a"})),
            ToolCall::new("fail", json!({})),
            ToolCall::new("missing", json!({})),
        ];
        let res = reg.call_many(&calls, &RunConfig::new()).await;
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].output, json!("a"));
        assert!(!res[0].is_error);
        assert!(res[1].is_error);
        assert!(res[2].is_error);
        assert!(
            res[2].output["error"]
                .as_str()
                .unwrap()
                .contains("not found")
        );
        let m = res[1].to_message();
        assert_eq!(m.tool_call_id.as_deref(), Some(calls[1].id.as_str()));
        assert!(m.text().starts_with("Error:"));
    }

    #[test]
    fn registry_replaces_duplicates() {
        let reg = ToolRegistry::new()
            .with(echo())
            .with(Tool::new("echo", "v2"));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("echo").unwrap().description(), "v2");
    }
}
