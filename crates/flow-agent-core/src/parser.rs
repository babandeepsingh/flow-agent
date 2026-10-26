//! Output parsers: turn model output into strings, JSON or tool calls.

use async_trait::async_trait;
use serde_json::Value;

use crate::config::RunConfig;
use crate::error::{Error, Result};
use crate::message::{Message, type_name};
use crate::runnable::Runnable;

/// Extract text from a string, a message object, or a list of messages (last one).
fn text_of(input: &Value) -> Result<String> {
    match input {
        Value::String(s) => Ok(s.clone()),
        Value::Object(map) if map.contains_key("role") => {
            Ok(Message::from_value(input.clone())?.text())
        }
        Value::Object(map) if map.contains_key("text") => Ok(text_of(&map["text"])?),
        Value::Array(_) | Value::Object(_) => Message::list_from_value(input.clone())?
            .pop()
            .map(|m| m.text())
            .ok_or_else(|| Error::InvalidInput("no messages to parse".into())),
        other => Err(Error::InvalidInput(format!(
            "cannot parse text from {}",
            type_name(other)
        ))),
    }
}

/// Returns the text of a message (or a string unchanged) as a JSON string.
///
/// ```
/// use flow_agent_core::{Runnable, StrOutputParser};
/// use serde_json::json;
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let out = StrOutputParser::new().invoke(json!({"role": "assistant", "content": "hi"})).await.unwrap();
/// assert_eq!(out, json!("hi"));
/// # });
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct StrOutputParser;

impl StrOutputParser {
    /// Create the parser.
    pub fn new() -> Self {
        StrOutputParser
    }
}

#[async_trait]
impl Runnable for StrOutputParser {
    fn name(&self) -> &str {
        "str_output_parser"
    }

    async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
        text_of(&input).map(Value::String)
    }
}

/// Parses JSON out of model text. Tolerates ```` ```json ```` fences and prose
/// around the JSON.
///
/// ```
/// use flow_agent_core::JsonOutputParser;
/// use serde_json::json;
///
/// let p = JsonOutputParser::new();
/// let v = p.parse("Sure! ```json\n{\"a\": 1}\n``` Hope that helps.").unwrap();
/// assert_eq!(v, json!({"a": 1}));
/// ```
#[derive(Clone, Debug, Default)]
pub struct JsonOutputParser {
    schema: Option<Value>,
}

impl JsonOutputParser {
    /// Create the parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Also check the result against a JSON Schema's top-level `type`,
    /// `required` keys and property `type`s (a lightweight check, not a full
    /// JSON Schema validator).
    pub fn with_schema(schema: Value) -> Self {
        JsonOutputParser {
            schema: Some(schema),
        }
    }

    /// Parse text into JSON.
    pub fn parse(&self, text: &str) -> Result<Value> {
        let v = extract_json(text)?;
        if let Some(schema) = &self.schema {
            check_schema(&v, schema).map_err(|m| Error::parse(m, text))?;
        }
        Ok(v)
    }
}

#[async_trait]
impl Runnable for JsonOutputParser {
    fn name(&self) -> &str {
        "json_output_parser"
    }

    async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
        self.parse(&text_of(&input)?)
    }
}

/// Find and parse the JSON value in `text`.
pub(crate) fn extract_json(text: &str) -> Result<Value> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str(trimmed) {
        return Ok(v);
    }
    // ```json ... ``` or ``` ... ```
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        if let Some(end) = body.find("```") {
            if let Ok(v) = serde_json::from_str(body[..end].trim()) {
                return Ok(v);
            }
        }
    }
    // First '{' or '[' to the matching last '}' or ']'.
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(s), Some(e)) = (trimmed.find(open), trimmed.rfind(close)) {
            if s < e {
                if let Ok(v) = serde_json::from_str(&trimmed[s..=e]) {
                    return Ok(v);
                }
            }
        }
    }
    Err(Error::parse("no valid JSON found in model output", text))
}

fn json_type_matches(v: &Value, ty: &str) -> bool {
    match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => true,
    }
}

/// Lightweight schema check: top-level type, required keys, property types.
pub(crate) fn check_schema(v: &Value, schema: &Value) -> std::result::Result<(), String> {
    if let Some(ty) = schema.get("type").and_then(Value::as_str) {
        if !json_type_matches(v, ty) {
            return Err(format!("expected {ty}, got {}", type_name(v)));
        }
    }
    let Some(obj) = v.as_object() else {
        return Ok(());
    };
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        let missing: Vec<&str> = required
            .iter()
            .filter_map(Value::as_str)
            .filter(|k| !obj.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            return Err(format!("missing required keys: {}", missing.join(", ")));
        }
    }
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for (k, prop) in props {
            if let (Some(val), Some(ty)) = (obj.get(k), prop.get("type").and_then(Value::as_str)) {
                if !json_type_matches(val, ty) {
                    return Err(format!("key '{k}': expected {ty}, got {}", type_name(val)));
                }
            }
        }
    }
    Ok(())
}

/// Returns the tool calls of an assistant message as `[{"id", "name", "args"}]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolCallsParser;

impl ToolCallsParser {
    /// Create the parser.
    pub fn new() -> Self {
        ToolCallsParser
    }
}

#[async_trait]
impl Runnable for ToolCallsParser {
    fn name(&self) -> &str {
        "tool_calls_parser"
    }

    async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
        let msg = Message::list_from_value(input)?
            .pop()
            .ok_or_else(|| Error::InvalidInput("no message to parse tool calls from".into()))?;
        Ok(serde_json::to_value(msg.tool_calls)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_variants() {
        assert_eq!(extract_json("[1,2]").unwrap(), json!([1, 2]));
        assert_eq!(
            extract_json("```\n{\"a\":true}\n```").unwrap(),
            json!({"a": true})
        );
        assert_eq!(
            extract_json("result: {\"a\": {\"b\": 1}} done").unwrap(),
            json!({"a": {"b": 1}})
        );
        assert!(extract_json("nothing here").is_err());
    }

    #[test]
    fn schema_check() {
        let p = JsonOutputParser::with_schema(json!({
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["name", "age"]
        }));
        assert!(p.parse(r#"{"name": "a", "age": 3}"#).is_ok());
        assert!(p.parse(r#"{"name": "a"}"#).is_err());
        assert!(p.parse(r#"{"name": "a", "age": "3"}"#).is_err());
        assert!(p.parse("[1]").is_err());
    }

    #[tokio::test]
    async fn str_parser_shapes() {
        let p = StrOutputParser::new();
        assert_eq!(p.invoke(json!("x")).await.unwrap(), json!("x"));
        assert_eq!(
            p.invoke(
                json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
            )
            .await
            .unwrap(),
            json!("b")
        );
        assert!(p.invoke(json!(1)).await.is_err());
    }

    #[tokio::test]
    async fn tool_calls_parser() {
        let m = Message::assistant("").with_tool_call("t", json!({"x": 1}));
        let out = ToolCallsParser::new().invoke(m.to_value()).await.unwrap();
        assert_eq!(out[0]["name"], "t");
        assert_eq!(out[0]["args"], json!({"x": 1}));
    }
}
