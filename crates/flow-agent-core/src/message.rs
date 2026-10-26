//! Chat messages, tool calls and token usage.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// Who authored a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions for the model.
    System,
    /// The end user. Also accepts `"human"` when deserializing.
    #[serde(alias = "human")]
    User,
    /// The model. Also accepts `"ai"` when deserializing.
    #[serde(alias = "ai")]
    Assistant,
    /// The result of a tool call.
    Tool,
}

impl Role {
    /// Parse a role name. Accepts `system`, `user`/`human`, `assistant`/`ai` and `tool`.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "system" => Ok(Role::System),
            "user" | "human" => Ok(Role::User),
            "assistant" | "ai" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            other => Err(Error::InvalidInput(format!(
                "unknown message role '{other}'"
            ))),
        }
    }

    /// The canonical lowercase name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Message content: plain text or a list of parts (text, images...).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain text.
    Text(String),
    /// Multi-part content.
    Parts(Vec<ContentPart>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_owned())
    }
}

/// One part of multi-part content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    /// A text segment.
    Text {
        /// The text.
        text: String,
    },
    /// An image referenced by URL (`https://...` or a `data:` URL).
    ImageUrl {
        /// The image.
        image_url: ImageUrl,
    },
}

/// An image reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// `https://...` or `data:image/png;base64,...`.
    pub url: String,
    /// Optional detail hint (`"low"`, `"high"`, `"auto"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool invocation requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id; tool results reference it.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments as a JSON object.
    #[serde(default)]
    pub args: Value,
}

impl ToolCall {
    /// Create a tool call with a generated id.
    pub fn new(name: impl Into<String>, args: Value) -> Self {
        ToolCall {
            id: new_call_id(),
            name: name.into(),
            args,
        }
    }
}

/// Token usage reported by a model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens.
    pub output_tokens: u64,
    /// `input_tokens + output_tokens`.
    pub total_tokens: u64,
}

impl Usage {
    /// Build usage from input and output token counts.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(self, rhs: Usage) -> Usage {
        Usage::new(
            self.input_tokens + rhs.input_tokens,
            self.output_tokens + rhs.output_tokens,
        )
    }
}

/// A chat message.
///
/// Serializes to an OpenAI-like JSON shape:
///
/// ```
/// use flow_agent_core::Message;
/// let m = Message::user("hi");
/// assert_eq!(m.to_value()["role"], "user");
/// assert_eq!(m.to_value()["content"], "hi");
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Author.
    pub role: Role,
    /// Content.
    #[serde(default)]
    pub content: Content,
    /// Tool calls requested by the model (assistant messages only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The tool call this message answers (tool messages only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional author name (or tool name for tool messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional id; the graph `Messages` reducer replaces messages with equal ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Token usage (assistant messages from a model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Free-form metadata (finish reason, provider ids...).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Value>,
}

impl Message {
    /// Create a message with the given role and content.
    pub fn new(role: Role, content: impl Into<Content>) -> Self {
        Message {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            id: None,
            usage: None,
            metadata: Map::new(),
        }
    }

    /// A system message.
    pub fn system(content: impl Into<Content>) -> Self {
        Self::new(Role::System, content)
    }

    /// A user message.
    pub fn user(content: impl Into<Content>) -> Self {
        Self::new(Role::User, content)
    }

    /// Alias for [`Message::user`].
    pub fn human(content: impl Into<Content>) -> Self {
        Self::user(content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<Content>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Alias for [`Message::assistant`].
    pub fn ai(content: impl Into<Content>) -> Self {
        Self::assistant(content)
    }

    /// A tool result answering `tool_call_id`. Non-string values are JSON-encoded.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<Value>) -> Self {
        let text = match content.into() {
            Value::String(s) => s,
            other => other.to_string(),
        };
        let mut m = Self::new(Role::Tool, text);
        m.tool_call_id = Some(tool_call_id.into());
        m
    }

    /// Add a tool call with a generated id (builder style).
    pub fn with_tool_call(mut self, name: impl Into<String>, args: Value) -> Self {
        self.tool_calls.push(ToolCall::new(name, args));
        self
    }

    /// Set the message id (builder style).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the author/tool name (builder style).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// All text content concatenated (image parts are skipped).
    pub fn text(&self) -> String {
        match &self.content {
            Content::Text(s) => s.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Whether the model asked for tools to be called.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Serialize to JSON.
    pub fn to_value(&self) -> Value {
        // Serializing this type cannot fail: all fields are plain data.
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Parse exactly one message from JSON. See [`Message::list_from_value`] for
    /// the accepted shapes; this fails if they yield anything but one message.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut list = Self::list_from_value(value)?;
        if list.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "expected exactly one message, got {}",
                list.len()
            )));
        }
        Ok(list.remove(0))
    }

    /// Coerce JSON into a list of messages. Accepts:
    ///
    /// - a string (→ one user message);
    /// - a message object `{"role": ..., "content": ...}`;
    /// - an object with a `"messages"` key (graph state);
    /// - an array of any of: message objects, strings (→ user), `["role", "content"]` pairs.
    ///
    /// ```
    /// use flow_agent_core::{Message, Role};
    /// use serde_json::json;
    /// let msgs = Message::list_from_value(json!([["system", "be brief"], "hi"])).unwrap();
    /// assert_eq!(msgs[0].role, Role::System);
    /// assert_eq!(msgs[1].text(), "hi");
    /// ```
    pub fn list_from_value(value: Value) -> Result<Vec<Self>> {
        match value {
            Value::String(s) => Ok(vec![Message::user(s)]),
            Value::Array(items) => items.into_iter().map(Self::item_from_value).collect(),
            Value::Object(mut map) => {
                if map.contains_key("role") {
                    Ok(vec![serde_json::from_value(Value::Object(map))?])
                } else if let Some(messages) = map.remove("messages") {
                    Self::list_from_value(messages)
                } else {
                    Err(Error::InvalidInput(
                        "expected a message object (with \"role\") or an object with \"messages\""
                            .into(),
                    ))
                }
            }
            other => Err(Error::InvalidInput(format!(
                "cannot convert {} into messages",
                type_name(&other)
            ))),
        }
    }

    fn item_from_value(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok(Message::user(s)),
            Value::Array(pair) if pair.len() == 2 => {
                let role = pair[0]
                    .as_str()
                    .ok_or_else(|| Error::InvalidInput("message pair role must be a string".into()))
                    .and_then(Role::parse)?;
                let content: Content = serde_json::from_value(pair[1].clone())?;
                Ok(Message::new(role, content))
            }
            Value::Object(_) => Ok(serde_json::from_value(value)?),
            other => Err(Error::InvalidInput(format!(
                "cannot convert {} into a message",
                type_name(&other)
            ))),
        }
    }
}

/// Serialize a list of messages to a JSON array.
pub fn messages_to_value(messages: &[Message]) -> Value {
    Value::Array(messages.iter().map(Message::to_value).collect())
}

/// The last message in a value accepted by [`Message::list_from_value`], if any.
///
/// Handy for reading the answer out of agent state:
///
/// ```
/// use flow_agent_core::message::last_message;
/// use serde_json::json;
/// let state = json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]});
/// assert_eq!(last_message(&state).unwrap().text(), "hello");
/// ```
pub fn last_message(value: &Value) -> Option<Message> {
    Message::list_from_value(value.clone()).ok()?.pop()
}

pub(crate) fn new_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

pub(crate) fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn constructors_and_text() {
        assert_eq!(Message::human("a").role, Role::User);
        assert_eq!(Message::ai("b").role, Role::Assistant);
        let t = Message::tool("c1", json!({"x": 1}));
        assert_eq!(t.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(t.text(), r#"{"x":1}"#);
    }

    #[test]
    fn roundtrip() {
        let m = Message::assistant("")
            .with_tool_call("search", json!({"q": "rust"}))
            .with_id("m1");
        let back: Message = serde_json::from_value(m.to_value()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn aliases_deserialize() {
        let m: Message = serde_json::from_value(json!({"role": "human", "content": "x"})).unwrap();
        assert_eq!(m.role, Role::User);
        let m: Message = serde_json::from_value(json!({"role": "ai"})).unwrap();
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.text(), "");
    }

    #[test]
    fn multipart_text() {
        let m: Message = serde_json::from_value(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look "},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "text", "text": "here"}
            ]
        }))
        .unwrap();
        assert_eq!(m.text(), "look here");
    }

    #[test]
    fn coercion_shapes() {
        assert_eq!(Message::list_from_value(json!("hi")).unwrap().len(), 1);
        assert_eq!(
            Message::list_from_value(json!({"messages": ["a", "b"]}))
                .unwrap()
                .len(),
            2
        );
        assert!(Message::list_from_value(json!(3)).is_err());
        assert!(Message::list_from_value(json!({"foo": 1})).is_err());
        assert!(Message::list_from_value(json!([["wizard", "x"]])).is_err());
        assert!(Message::from_value(json!(["a", "b"])).is_err());
    }

    #[test]
    fn usage_adds() {
        assert_eq!(Usage::new(1, 2) + Usage::new(3, 4), Usage::new(4, 6));
    }
}
