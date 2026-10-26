//! How node updates are merged into graph state.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use flow_agent_core::{Error, Message, Result};
use serde_json::{Map, Value};

type CustomFn = Arc<dyn Fn(Value, Value) -> Result<Value> + Send + Sync>;

/// Merges a new value for a state key into the old one.
///
/// Keys without a declared reducer use [`Reducer::Overwrite`].
#[derive(Clone)]
#[non_exhaustive]
pub enum Reducer {
    /// Replace the old value. Two writes to the same key in one superstep are an error.
    Overwrite,
    /// Concatenate arrays; non-array values are appended as one element.
    Append,
    /// Shallow object merge (new keys win).
    Merge,
    /// Recursive object merge.
    DeepMerge,
    /// Chat messages: append, replace messages with the same `id`, and remove
    /// messages given as `{"remove": "<id>"}`. Messages without an id get one.
    Messages,
    /// Your own `fn(old, new) -> merged`. `old` is `null` the first time.
    Custom(CustomFn),
}

impl fmt::Debug for Reducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Reducer::Overwrite => "Overwrite",
            Reducer::Append => "Append",
            Reducer::Merge => "Merge",
            Reducer::DeepMerge => "DeepMerge",
            Reducer::Messages => "Messages",
            Reducer::Custom(_) => "Custom",
        })
    }
}

impl Reducer {
    /// A custom reducer.
    ///
    /// ```
    /// use flow_agent_graph::Reducer;
    /// use serde_json::json;
    /// let sum = Reducer::custom(|old, new| Ok(json!(old.as_i64().unwrap_or(0) + new.as_i64().unwrap_or(0))));
    /// ```
    pub fn custom(f: impl Fn(Value, Value) -> Result<Value> + Send + Sync + 'static) -> Self {
        Reducer::Custom(Arc::new(f))
    }

    /// Merge `new` into `old` (`None` when the key is unset).
    pub fn apply(&self, key: &str, old: Option<Value>, new: Value) -> Result<Value> {
        match self {
            Reducer::Overwrite => Ok(new),
            Reducer::Append => {
                let mut out = match old {
                    Some(Value::Array(a)) => a,
                    None | Some(Value::Null) => Vec::new(),
                    Some(other) => vec![other],
                };
                match new {
                    Value::Array(a) => out.extend(a),
                    Value::Null => {}
                    other => out.push(other),
                }
                Ok(Value::Array(out))
            }
            Reducer::Merge | Reducer::DeepMerge => {
                let deep = matches!(self, Reducer::DeepMerge);
                match (old, new) {
                    (None | Some(Value::Null), new @ Value::Object(_)) => Ok(new),
                    (Some(Value::Object(mut a)), Value::Object(b)) => {
                        merge_into(&mut a, b, deep);
                        Ok(Value::Object(a))
                    }
                    (_, Value::Null) => Ok(Value::Null),
                    _ => Err(Error::InvalidUpdate {
                        key: key.into(),
                        message: "Merge reducer needs object values".into(),
                    }),
                }
            }
            Reducer::Messages => merge_messages(old, new).map_err(|e| Error::InvalidUpdate {
                key: key.into(),
                message: e.to_string(),
            }),
            Reducer::Custom(f) => f(old.unwrap_or(Value::Null), new),
        }
    }
}

fn merge_into(a: &mut Map<String, Value>, b: Map<String, Value>, deep: bool) {
    for (k, v) in b {
        match (a.get_mut(&k), v) {
            (Some(Value::Object(ea)), Value::Object(eb)) if deep => merge_into(ea, eb, true),
            (_, v) => {
                a.insert(k, v);
            }
        }
    }
}

fn merge_messages(old: Option<Value>, new: Value) -> Result<Value> {
    let mut list: Vec<Message> = match old {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => Message::list_from_value(v)?,
    };
    let items = match new {
        Value::Array(a) => a,
        Value::Null => Vec::new(),
        other => vec![other],
    };
    for item in items {
        if let Some(id) = item.get("remove").and_then(Value::as_str) {
            if item.get("role").is_none() {
                list.retain(|m| m.id.as_deref() != Some(id));
                continue;
            }
        }
        for mut m in Message::list_from_value(Value::Array(vec![item]))? {
            match m
                .id
                .as_deref()
                .and_then(|id| list.iter().position(|x| x.id.as_deref() == Some(id)))
            {
                Some(pos) => list[pos] = m,
                None => {
                    if m.id.is_none() {
                        m.id = Some(uuid::Uuid::now_v7().to_string());
                    }
                    list.push(m);
                }
            }
        }
    }
    Ok(Value::Array(list.iter().map(Message::to_value).collect()))
}

/// Per-key reducers.
#[derive(Clone, Debug, Default)]
pub(crate) struct Schema {
    pub reducers: BTreeMap<String, Reducer>,
}

impl Schema {
    /// Apply ordered `(writer, update)` pairs to `state`. Two writers setting the
    /// same `Overwrite` key in one batch is an error.
    pub fn apply(&self, state: &mut Map<String, Value>, updates: &[(String, Value)]) -> Result<()> {
        let mut overwritten: BTreeMap<&str, &str> = BTreeMap::new();
        for (writer, update) in updates {
            let obj = match update {
                Value::Null => continue,
                Value::Object(o) => o,
                other => {
                    return Err(Error::InvalidUpdate {
                        key: "*".into(),
                        message: format!(
                            "node '{writer}' returned {}, expected an object of state keys",
                            short(other)
                        ),
                    });
                }
            };
            let mut seen = BTreeSet::new();
            for (k, v) in obj {
                let reducer = self.reducers.get(k).unwrap_or(&Reducer::Overwrite);
                if matches!(reducer, Reducer::Overwrite) && seen.insert(k.as_str()) {
                    if let Some(prev) = overwritten.insert(k.as_str(), writer.as_str()) {
                        if prev != writer {
                            return Err(Error::InvalidUpdate {
                                key: k.clone(),
                                message: format!(
                                    "nodes '{prev}' and '{writer}' both wrote it in the same step; \
                                     declare a reducer with StateGraph::key"
                                ),
                            });
                        }
                    }
                }
                let merged = reducer.apply(k, state.remove(k), v.clone())?;
                state.insert(k.clone(), merged);
            }
        }
        Ok(())
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 60 {
        format!("{}…", s.chars().take(60).collect::<String>())
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn append_and_merge() {
        let r = Reducer::Append;
        assert_eq!(r.apply("k", None, json!(1)).unwrap(), json!([1]));
        assert_eq!(
            r.apply("k", Some(json!([1])), json!([2, 3])).unwrap(),
            json!([1, 2, 3])
        );
        let m = Reducer::Merge;
        assert_eq!(
            m.apply("k", Some(json!({"a": {"x": 1}})), json!({"a": {"y": 2}}))
                .unwrap(),
            json!({"a": {"y": 2}})
        );
        assert_eq!(
            Reducer::DeepMerge
                .apply("k", Some(json!({"a": {"x": 1}})), json!({"a": {"y": 2}}))
                .unwrap(),
            json!({"a": {"x": 1, "y": 2}})
        );
        assert!(m.apply("k", Some(json!(1)), json!({"a": 1})).is_err());
    }

    #[test]
    fn messages_append_replace_remove() {
        let r = Reducer::Messages;
        let s = r.apply("m", None, json!("hi")).unwrap();
        let id = s[0]["id"].as_str().unwrap().to_owned();
        assert_eq!(s[0]["role"], "user");
        let s = r
            .apply("m", Some(s), json!([{"role": "user", "content": "edited", "id": id}, {"role": "assistant", "content": "yo"}]))
            .unwrap();
        assert_eq!(s.as_array().unwrap().len(), 2);
        assert_eq!(s[0]["content"], "edited");
        let s = r.apply("m", Some(s), json!({"remove": id})).unwrap();
        assert_eq!(s.as_array().unwrap().len(), 1);
        assert!(r.apply("m", None, json!(5)).is_err());
    }

    #[test]
    fn overwrite_conflicts_detected() {
        let schema = Schema::default();
        let mut st = Map::new();
        let err = schema
            .apply(
                &mut st,
                &[("a".into(), json!({"x": 1})), ("b".into(), json!({"x": 2}))],
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidUpdate { .. }));

        let mut schema = Schema::default();
        schema.reducers.insert("x".into(), Reducer::Append);
        let mut st = Map::new();
        schema
            .apply(
                &mut st,
                &[("a".into(), json!({"x": 1})), ("b".into(), json!({"x": 2}))],
            )
            .unwrap();
        assert_eq!(st["x"], json!([1, 2]));
        assert!(schema.apply(&mut st, &[("a".into(), json!([1]))]).is_err());
    }
}
