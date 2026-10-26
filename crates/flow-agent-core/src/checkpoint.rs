//! The [`Checkpointer`] trait and checkpoint data types.
//!
//! Implementations (memory, SQLite, Postgres) live in the
//! `flow-agent-checkpoint` crate. The trait is here so graphs don't depend on
//! any storage backend.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::Result;

/// Current checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// A pause requested by a node, waiting for human input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interrupt {
    /// Unique id used to match the resume value.
    pub id: String,
    /// Node that interrupted.
    pub node: String,
    /// Payload shown to the human.
    pub value: Value,
}

/// A state update produced by a node that finished during a superstep that
/// did not complete (because another node interrupted or failed).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingWrite {
    /// Id of the node execution.
    pub task_id: String,
    /// Node name.
    pub node: String,
    /// The partial update.
    pub update: Value,
}

/// A snapshot of graph state after a superstep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version ([`CHECKPOINT_VERSION`]).
    pub v: u32,
    /// Checkpoint id (UUID v7, sortable by time).
    pub id: String,
    /// Thread this checkpoint belongs to.
    pub thread_id: String,
    /// `""` for the root graph; `"parent:child"` for subgraphs.
    #[serde(default)]
    pub namespace: String,
    /// Previous checkpoint in this thread.
    pub parent_id: Option<String>,
    /// Superstep number.
    pub step: u32,
    /// Full state.
    pub state: Value,
    /// Nodes scheduled for the next superstep.
    #[serde(default)]
    pub next: Vec<String>,
    /// Writes from nodes that finished in an incomplete superstep.
    #[serde(default)]
    pub pending_writes: Vec<PendingWrite>,
    /// Open interrupts.
    #[serde(default)]
    pub interrupts: Vec<Interrupt>,
    /// Metadata: `source` (`"input"`, `"loop"`, `"update"`, `"fork"`), writes, tags...
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// Creation time.
    pub created_at: DateTime<Utc>,
}

impl Checkpoint {
    /// A new checkpoint with a fresh id and the current time.
    pub fn new(thread_id: impl Into<String>, step: u32, state: Value) -> Self {
        Checkpoint {
            v: CHECKPOINT_VERSION,
            id: uuid::Uuid::now_v7().to_string(),
            thread_id: thread_id.into(),
            namespace: String::new(),
            parent_id: None,
            step,
            state,
            next: Vec::new(),
            pending_writes: Vec::new(),
            interrupts: Vec::new(),
            metadata: Map::new(),
            created_at: Utc::now(),
        }
    }
}

/// Options for [`Checkpointer::list`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ListFilter {
    /// Namespace to list; `None` means the root graph (`""`).
    pub namespace: Option<String>,
    /// Only checkpoints created before this checkpoint id.
    pub before: Option<String>,
    /// Max number of results.
    pub limit: Option<usize>,
}

/// Persistent storage for graph checkpoints, keyed by thread id.
///
/// Implementations must be safe to share across tasks. Checkpoint ids are
/// UUID v7 strings, so sorting by id sorts by creation time.
#[async_trait]
pub trait Checkpointer: Send + Sync {
    /// Store a checkpoint.
    async fn put(&self, checkpoint: &Checkpoint) -> Result<()>;

    /// Store writes from an incomplete superstep for `checkpoint_id`.
    async fn put_writes(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: &str,
        writes: &[PendingWrite],
    ) -> Result<()>;

    /// Fetch a checkpoint; the latest one when `checkpoint_id` is `None`.
    async fn get(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint>>;

    /// List checkpoints for a thread, newest first.
    async fn list(&self, thread_id: &str, filter: ListFilter) -> Result<Vec<Checkpoint>>;

    /// Delete every checkpoint of a thread.
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip() {
        let mut c = Checkpoint::new("t", 2, json!({"a": 1}));
        c.interrupts.push(Interrupt {
            id: "i".into(),
            node: "n".into(),
            value: json!("?"),
        });
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["v"], 1);
        let back: Checkpoint = serde_json::from_value(v).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn ids_sort_by_time() {
        let a = Checkpoint::new("t", 0, json!({}));
        let b = Checkpoint::new("t", 1, json!({}));
        assert!(a.id < b.id);
    }
}
