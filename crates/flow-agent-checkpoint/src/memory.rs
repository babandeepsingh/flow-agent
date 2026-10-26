//! In-memory checkpointer.

use std::collections::BTreeMap;

use async_trait::async_trait;
use flow_agent_core::{Checkpoint, Checkpointer, ListFilter, PendingWrite, Result};
use tokio::sync::RwLock;

type Key = (String, String); // (thread_id, namespace)

/// Keeps checkpoints in process memory. For tests, prototypes and
/// single-process apps that don't need state to survive a restart.
///
/// ```
/// use std::sync::Arc;
/// use flow_agent_checkpoint::MemorySaver;
/// let saver = Arc::new(MemorySaver::new());
/// ```
#[derive(Debug, Default)]
pub struct MemorySaver {
    // Checkpoints per thread/namespace, keyed by id (UUID v7, so time-ordered).
    data: RwLock<BTreeMap<Key, BTreeMap<String, Checkpoint>>>,
}

impl MemorySaver {
    /// An empty saver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored checkpoints (all threads).
    pub async fn len(&self) -> usize {
        self.data.read().await.values().map(BTreeMap::len).sum()
    }

    /// Whether no checkpoints are stored.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

fn key(thread_id: &str, namespace: &str) -> Key {
    (thread_id.to_owned(), namespace.to_owned())
}

#[async_trait]
impl Checkpointer for MemorySaver {
    async fn put(&self, checkpoint: &Checkpoint) -> Result<()> {
        self.data
            .write()
            .await
            .entry(key(&checkpoint.thread_id, &checkpoint.namespace))
            .or_default()
            .insert(checkpoint.id.clone(), checkpoint.clone());
        Ok(())
    }

    async fn put_writes(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: &str,
        writes: &[PendingWrite],
    ) -> Result<()> {
        let mut data = self.data.write().await;
        let cp = data
            .get_mut(&key(thread_id, namespace))
            .and_then(|m| m.get_mut(checkpoint_id))
            .ok_or_else(|| {
                flow_agent_core::Error::Checkpoint(format!(
                    "checkpoint '{checkpoint_id}' not found for thread '{thread_id}'"
                ))
            })?;
        for w in writes {
            cp.pending_writes.retain(|x| x.task_id != w.task_id);
            cp.pending_writes.push(w.clone());
        }
        Ok(())
    }

    async fn get(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint>> {
        let data = self.data.read().await;
        let Some(m) = data.get(&key(thread_id, namespace)) else {
            return Ok(None);
        };
        Ok(match checkpoint_id {
            Some(id) => m.get(id).cloned(),
            None => m.values().next_back().cloned(),
        })
    }

    async fn list(&self, thread_id: &str, filter: ListFilter) -> Result<Vec<Checkpoint>> {
        let data = self.data.read().await;
        let ns = filter.namespace.as_deref().unwrap_or("");
        let Some(m) = data.get(&key(thread_id, ns)) else {
            return Ok(Vec::new());
        };
        Ok(m.values()
            .rev()
            .filter(|c| filter.before.as_deref().is_none_or(|b| c.id.as_str() < b))
            .take(filter.limit.unwrap_or(usize::MAX))
            .cloned()
            .collect())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.data.write().await.retain(|(t, _), _| t != thread_id);
        Ok(())
    }
}
