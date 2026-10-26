//! Prompt templates with `{variable}` placeholders.

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::config::RunConfig;
use crate::error::{Error, Result};
use crate::message::{Message, Role, messages_to_value, type_name};
use crate::runnable::Runnable;

#[derive(Clone, Debug, PartialEq)]
enum Segment {
    Literal(String),
    Var(String),
}

/// Parse an f-string style template. `{{` and `}}` are literal braces.
fn parse(template: &str) -> std::result::Result<Vec<Segment>, String> {
    let mut segments = Vec::new();
    let mut literal = String::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                literal.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                literal.push('}');
            }
            '{' => {
                let mut name = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some('{') | None => {
                            return Err(format!("unclosed '{{' in template: {template:?}"));
                        }
                        Some(ch) => name.push(ch),
                    }
                }
                let name = name.trim().to_owned();
                if name.is_empty() {
                    return Err(format!("empty variable '{{}}' in template: {template:?}"));
                }
                if !literal.is_empty() {
                    segments.push(Segment::Literal(std::mem::take(&mut literal)));
                }
                segments.push(Segment::Var(name));
            }
            '}' => return Err(format!("unmatched '}}' in template: {template:?}")),
            ch => literal.push(ch),
        }
    }
    if !literal.is_empty() {
        segments.push(Segment::Literal(literal));
    }
    Ok(segments)
}

fn render_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A string template such as `"Tell me about {topic}"`.
///
/// Input is a JSON object of variables (or, when the template has exactly one
/// variable, a plain string). Output is a JSON string.
///
/// ```
/// use flow_agent_core::PromptTemplate;
/// use serde_json::json;
///
/// let p = PromptTemplate::new("Tell me about {topic}. Use {{braces}}.");
/// assert_eq!(p.format(&json!({"topic": "crabs"})).unwrap(), "Tell me about crabs. Use {braces}.");
/// assert_eq!(p.variables(), vec!["topic"]);
/// ```
///
/// A malformed template (e.g. an unclosed `{`) is reported as
/// [`Error::Template`] when formatting, so construction never fails or panics.
#[derive(Clone, Debug)]
pub struct PromptTemplate {
    template: String,
    parsed: std::result::Result<Vec<Segment>, String>,
    partials: Map<String, Value>,
}

impl PromptTemplate {
    /// Create a template.
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let parsed = parse(&template);
        PromptTemplate {
            template,
            parsed,
            partials: Map::new(),
        }
    }

    /// Pre-fill a variable.
    pub fn partial(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.partials.insert(name.into(), value.into());
        self
    }

    /// The raw template string.
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Variables in order of first appearance, excluding pre-filled ones.
    pub fn variables(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        if let Ok(segs) = &self.parsed {
            for s in segs {
                if let Segment::Var(v) = s {
                    if !self.partials.contains_key(v) && !out.contains(v) {
                        out.push(v.clone());
                    }
                }
            }
        }
        out
    }

    /// Render with the given variables.
    pub fn format(&self, input: &Value) -> Result<String> {
        let segments = self.parsed.as_ref().map_err(|m| Error::Template {
            message: m.clone(),
            missing: vec![],
        })?;
        let vars = self.variables();
        let single;
        let lookup: &Map<String, Value> = match input {
            Value::Object(map) => map,
            Value::String(_) | Value::Number(_) | Value::Bool(_) if vars.len() == 1 => {
                single = Map::from_iter([(vars[0].clone(), input.clone())]);
                &single
            }
            Value::Null => {
                single = Map::new();
                &single
            }
            other => {
                return Err(Error::Template {
                    message: format!("expected an object of variables, got {}", type_name(other)),
                    missing: vars,
                });
            }
        };

        let missing: Vec<String> = vars
            .iter()
            .filter(|v| !lookup.contains_key(*v))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(Error::Template {
                message: format!("missing variables: {}", missing.join(", ")),
                missing,
            });
        }

        let mut out = String::with_capacity(self.template.len());
        for s in segments {
            match s {
                Segment::Literal(l) => out.push_str(l),
                Segment::Var(v) => {
                    let val = lookup.get(v).or_else(|| self.partials.get(v));
                    if let Some(val) = val {
                        out.push_str(&render_value(val));
                    }
                }
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl Runnable for PromptTemplate {
    fn name(&self) -> &str {
        "prompt_template"
    }

    async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
        self.format(&input).map(Value::String)
    }
}

#[derive(Clone, Debug)]
enum Entry {
    Template(Result<Role, String>, PromptTemplate),
    Placeholder { name: String, optional: bool },
    Fixed(Message),
}

/// A list of message templates that renders to a JSON array of messages.
///
/// ```
/// use flow_agent_core::ChatPromptTemplate;
/// use serde_json::json;
///
/// let prompt = ChatPromptTemplate::new()
///     .system("You are a {style} assistant.")
///     .placeholder("history")
///     .user("{question}");
///
/// let msgs = prompt.format_messages(&json!({
///     "style": "terse",
///     "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
///     "question": "What is Rust?"
/// })).unwrap();
/// assert_eq!(msgs.len(), 4);
/// assert_eq!(msgs[3].text(), "What is Rust?");
///
/// // Python-style construction from (role, template) pairs:
/// let prompt = ChatPromptTemplate::from_messages([("system", "Be {style}."), ("user", "{question}")]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ChatPromptTemplate {
    entries: Vec<Entry>,
    partials: Map<String, Value>,
}

impl ChatPromptTemplate {
    /// An empty template.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `(role, template)` pairs. Roles: `system`, `user`/`human`, `assistant`/`ai`.
    pub fn from_messages<I, R, T>(messages: I) -> Self
    where
        I: IntoIterator<Item = (R, T)>,
        R: AsRef<str>,
        T: Into<String>,
    {
        let mut p = Self::new();
        for (role, tpl) in messages {
            let role = Role::parse(role.as_ref()).map_err(|e| e.to_string());
            p.entries
                .push(Entry::Template(role, PromptTemplate::new(tpl)));
        }
        p
    }

    fn push(mut self, role: Role, template: impl Into<String>) -> Self {
        self.entries
            .push(Entry::Template(Ok(role), PromptTemplate::new(template)));
        self
    }

    /// Add a system message template.
    pub fn system(self, template: impl Into<String>) -> Self {
        self.push(Role::System, template)
    }

    /// Add a user message template.
    pub fn user(self, template: impl Into<String>) -> Self {
        self.push(Role::User, template)
    }

    /// Add an assistant message template.
    pub fn assistant(self, template: impl Into<String>) -> Self {
        self.push(Role::Assistant, template)
    }

    /// Insert the messages found in `input[name]`. Missing → error.
    pub fn placeholder(mut self, name: impl Into<String>) -> Self {
        self.entries.push(Entry::Placeholder {
            name: name.into(),
            optional: false,
        });
        self
    }

    /// Insert the messages found in `input[name]`, if present.
    pub fn optional_placeholder(mut self, name: impl Into<String>) -> Self {
        self.entries.push(Entry::Placeholder {
            name: name.into(),
            optional: true,
        });
        self
    }

    /// Add a fixed message (no templating).
    pub fn message(mut self, message: Message) -> Self {
        self.entries.push(Entry::Fixed(message));
        self
    }

    /// Pre-fill a variable for every message template.
    pub fn partial(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.partials.insert(name.into(), value.into());
        self
    }

    /// All variables (including placeholders), in order of first appearance.
    pub fn variables(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in &self.entries {
            let names = match e {
                Entry::Template(_, t) => t.variables(),
                Entry::Placeholder { name, .. } => vec![name.clone()],
                Entry::Fixed(_) => vec![],
            };
            for n in names {
                if !self.partials.contains_key(&n) && !out.contains(&n) {
                    out.push(n);
                }
            }
        }
        out
    }

    /// Render to messages.
    pub fn format_messages(&self, input: &Value) -> Result<Vec<Message>> {
        let mut vars = self.partials.clone();
        match input {
            Value::Object(m) => vars.extend(m.iter().map(|(k, v)| (k.clone(), v.clone()))),
            Value::Null => {}
            other => {
                let names = self.variables();
                if names.len() == 1 {
                    vars.insert(names[0].clone(), other.clone());
                } else {
                    return Err(Error::Template {
                        message: format!(
                            "expected an object of variables, got {}",
                            type_name(other)
                        ),
                        missing: names,
                    });
                }
            }
        }
        let vars = Value::Object(vars);

        let mut out = Vec::new();
        let mut missing = Vec::new();
        for e in &self.entries {
            match e {
                Entry::Template(role, t) => {
                    let role = role.clone().map_err(|m| Error::Template {
                        message: m,
                        missing: vec![],
                    })?;
                    match t.format(&vars) {
                        Ok(text) => out.push(Message::new(role, text)),
                        Err(Error::Template { missing: m, .. }) if !m.is_empty() => {
                            missing.extend(m)
                        }
                        Err(e) => return Err(e),
                    }
                }
                Entry::Placeholder { name, optional } => match vars.get(name) {
                    Some(Value::Null) | None if *optional => {}
                    None => missing.push(name.clone()),
                    Some(v) => out.extend(
                        Message::list_from_value(v.clone())
                            .map_err(|e| e.context(format!("placeholder '{name}'")))?,
                    ),
                },
                Entry::Fixed(m) => out.push(m.clone()),
            }
        }
        if !missing.is_empty() {
            missing.dedup();
            return Err(Error::Template {
                message: format!("missing variables: {}", missing.join(", ")),
                missing,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl Runnable for ChatPromptTemplate {
    fn name(&self) -> &str {
        "chat_prompt_template"
    }

    async fn invoke_with(&self, input: Value, _config: &RunConfig) -> Result<Value> {
        self.format_messages(&input).map(|m| messages_to_value(&m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_cases() {
        assert!(PromptTemplate::new("{a").format(&json!({"a": 1})).is_err());
        assert!(PromptTemplate::new("a}").format(&json!({})).is_err());
        assert!(PromptTemplate::new("{}").format(&json!({})).is_err());
        let p = PromptTemplate::new("{ a }-{b}-{a}");
        assert_eq!(p.variables(), vec!["a", "b"]);
        assert_eq!(p.format(&json!({"a": 1, "b": [2]})).unwrap(), "1-[2]-1");
    }

    #[test]
    fn missing_variables_listed() {
        let err = PromptTemplate::new("{a}{b}")
            .format(&json!({"a": 1}))
            .unwrap_err();
        match err {
            Error::Template { missing, .. } => assert_eq!(missing, vec!["b"]),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn single_variable_accepts_scalar_and_partials() {
        let p = PromptTemplate::new("Hi {name}");
        assert_eq!(p.format(&json!("Ann")).unwrap(), "Hi Ann");
        let p = PromptTemplate::new("{greet} {name}").partial("greet", "Yo");
        assert_eq!(p.variables(), vec!["name"]);
        assert_eq!(p.format(&json!("Bo")).unwrap(), "Yo Bo");
    }

    #[test]
    fn chat_template() {
        let p = ChatPromptTemplate::from_messages([("system", "be {x}"), ("human", "{q}")])
            .optional_placeholder("extra");
        let msgs = p.format_messages(&json!({"x": "nice", "q": "?"})).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].role, Role::User);

        let err = ChatPromptTemplate::new()
            .user("{a}")
            .placeholder("h")
            .format_messages(&json!({}))
            .unwrap_err();
        match err {
            Error::Template { missing, .. } => assert_eq!(missing, vec!["a", "h"]),
            other => panic!("unexpected {other:?}"),
        }

        let bad_role = ChatPromptTemplate::from_messages([("wizard", "x")]);
        assert!(bad_role.format_messages(&json!({})).is_err());
    }

    #[tokio::test]
    async fn runnable_output() {
        let v = ChatPromptTemplate::new()
            .user("{q}")
            .invoke(json!("hey"))
            .await
            .unwrap();
        assert_eq!(v, json!([{"role": "user", "content": "hey"}]));
    }
}
