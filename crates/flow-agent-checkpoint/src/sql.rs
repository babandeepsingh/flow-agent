//! SQLite and Postgres checkpointers (sqlx).
//!
//! Each checkpoint is stored as one JSON document plus indexed columns for
//! lookup. Pending writes live in a second table and are merged on read.

use flow_agent_core::{Checkpoint, Error, PendingWrite, Result};

pub(crate) fn db_err(e: sqlx::Error) -> Error {
    Error::Checkpoint(format!("database error: {e}"))
}

pub(crate) fn decode(data: &str, writes: Vec<String>) -> Result<Checkpoint> {
    let mut cp: Checkpoint = serde_json::from_str(data)
        .map_err(|e| Error::Checkpoint(format!("corrupt checkpoint: {e}")))?;
    for w in writes {
        let w: PendingWrite = serde_json::from_str(&w)
            .map_err(|e| Error::Checkpoint(format!("corrupt pending write: {e}")))?;
        cp.pending_writes.retain(|x| x.task_id != w.task_id);
        cp.pending_writes.push(w);
    }
    Ok(cp)
}

pub(crate) fn encode(cp: &Checkpoint) -> Result<String> {
    Ok(serde_json::to_string(cp)?)
}

#[cfg(feature = "sqlite")]
pub use self::sqlite::SqliteSaver;

#[cfg(feature = "sqlite")]
mod sqlite {
    use std::str::FromStr;

    use async_trait::async_trait;
    use flow_agent_core::{Checkpoint, Checkpointer, ListFilter, PendingWrite, Result};
    use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};

    use super::{db_err, decode, encode};

    const SCHEMA: &[&str] = &[
        "CREATE TABLE IF NOT EXISTS flow_agent_checkpoints (
            thread_id  TEXT    NOT NULL,
            namespace  TEXT    NOT NULL,
            id         TEXT    NOT NULL,
            parent_id  TEXT,
            step       INTEGER NOT NULL,
            data       TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            PRIMARY KEY (thread_id, namespace, id)
        )",
        "CREATE TABLE IF NOT EXISTS flow_agent_writes (
            thread_id     TEXT NOT NULL,
            namespace     TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            task_id       TEXT NOT NULL,
            data          TEXT NOT NULL,
            PRIMARY KEY (thread_id, namespace, checkpoint_id, task_id)
        )",
    ];

    /// Stores checkpoints in SQLite. Good for single-node services, CLIs and
    /// desktop apps.
    ///
    /// ```no_run
    /// # async fn demo() -> flow_agent_core::Result<()> {
    /// use flow_agent_checkpoint::SqliteSaver;
    /// let saver = SqliteSaver::connect("sqlite://agent.db").await?;
    /// saver.migrate().await?; // explicit, never implicit
    /// # Ok(()) }
    /// ```
    #[derive(Clone, Debug)]
    pub struct SqliteSaver {
        pool: SqlitePool,
    }

    impl SqliteSaver {
        /// Open (creating the file if needed) with WAL journaling.
        /// Accepts `sqlite://path.db`, `sqlite:path.db` or `sqlite::memory:`.
        pub async fn connect(url: &str) -> Result<Self> {
            let opts = SqliteConnectOptions::from_str(url)
                .map_err(db_err)?
                .create_if_missing(true)
                .journal_mode(SqliteJournalMode::Wal);
            // An in-memory database exists per connection, so keep a single one.
            let max = if url.contains(":memory:") { 1 } else { 8 };
            let pool = SqlitePoolOptions::new()
                .max_connections(max)
                .connect_with(opts)
                .await
                .map_err(db_err)?;
            Ok(SqliteSaver { pool })
        }

        /// Use an existing pool.
        pub fn from_pool(pool: SqlitePool) -> Self {
            SqliteSaver { pool }
        }

        /// The underlying pool.
        pub fn pool(&self) -> &SqlitePool {
            &self.pool
        }

        /// Create the tables if they don't exist. Safe to run repeatedly.
        pub async fn migrate(&self) -> Result<()> {
            for stmt in SCHEMA {
                sqlx::query(stmt)
                    .execute(&self.pool)
                    .await
                    .map_err(db_err)?;
            }
            Ok(())
        }

        async fn writes(&self, thread: &str, ns: &str, id: &str) -> Result<Vec<String>> {
            sqlx::query_scalar(
                "SELECT data FROM flow_agent_writes
                 WHERE thread_id = ? AND namespace = ? AND checkpoint_id = ? ORDER BY task_id",
            )
            .bind(thread)
            .bind(ns)
            .bind(id)
            .fetch_all(&self.pool)
            .await
            .map_err(db_err)
        }
    }

    #[async_trait]
    impl Checkpointer for SqliteSaver {
        async fn put(&self, cp: &Checkpoint) -> Result<()> {
            sqlx::query(
                "INSERT OR REPLACE INTO flow_agent_checkpoints
                 (thread_id, namespace, id, parent_id, step, data, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&cp.thread_id)
            .bind(&cp.namespace)
            .bind(&cp.id)
            .bind(&cp.parent_id)
            .bind(i64::from(cp.step))
            .bind(encode(cp)?)
            .bind(cp.created_at.to_rfc3339())
            .execute(&self.pool)
            .await
            .map_err(db_err)?;
            Ok(())
        }

        async fn put_writes(
            &self,
            thread_id: &str,
            namespace: &str,
            checkpoint_id: &str,
            writes: &[PendingWrite],
        ) -> Result<()> {
            let mut tx = self.pool.begin().await.map_err(db_err)?;
            for w in writes {
                sqlx::query(
                    "INSERT OR REPLACE INTO flow_agent_writes
                     (thread_id, namespace, checkpoint_id, task_id, data) VALUES (?, ?, ?, ?, ?)",
                )
                .bind(thread_id)
                .bind(namespace)
                .bind(checkpoint_id)
                .bind(&w.task_id)
                .bind(serde_json::to_string(w)?)
                .execute(&mut *tx)
                .await
                .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)
        }

        async fn get(
            &self,
            thread_id: &str,
            namespace: &str,
            checkpoint_id: Option<&str>,
        ) -> Result<Option<Checkpoint>> {
            let row: Option<(String, String)> = match checkpoint_id {
                Some(id) => sqlx::query_as(
                    "SELECT id, data FROM flow_agent_checkpoints
                     WHERE thread_id = ? AND namespace = ? AND id = ?",
                )
                .bind(thread_id)
                .bind(namespace)
                .bind(id),
                None => sqlx::query_as(
                    "SELECT id, data FROM flow_agent_checkpoints
                     WHERE thread_id = ? AND namespace = ? ORDER BY id DESC LIMIT 1",
                )
                .bind(thread_id)
                .bind(namespace),
            }
            .fetch_optional(&self.pool)
            .await
            .map_err(db_err)?;
            match row {
                Some((id, data)) => {
                    let writes = self.writes(thread_id, namespace, &id).await?;
                    decode(&data, writes).map(Some)
                }
                None => Ok(None),
            }
        }

        async fn list(&self, thread_id: &str, filter: ListFilter) -> Result<Vec<Checkpoint>> {
            let ns = filter.namespace.as_deref().unwrap_or("");
            let limit = filter
                .limit
                .map(|l| i64::try_from(l).unwrap_or(i64::MAX))
                .unwrap_or(-1);
            let rows: Vec<(String, String)> = sqlx::query_as(
                "SELECT id, data FROM flow_agent_checkpoints
                 WHERE thread_id = ? AND namespace = ? AND (? IS NULL OR id < ?)
                 ORDER BY id DESC LIMIT ?",
            )
            .bind(thread_id)
            .bind(ns)
            .bind(&filter.before)
            .bind(&filter.before)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(db_err)?;
            let mut out = Vec::with_capacity(rows.len());
            for (id, data) in rows {
                let writes = self.writes(thread_id, ns, &id).await?;
                out.push(decode(&data, writes)?);
            }
            Ok(out)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            let mut tx = self.pool.begin().await.map_err(db_err)?;
            for table in ["flow_agent_checkpoints", "flow_agent_writes"] {
                sqlx::query(&format!("DELETE FROM {table} WHERE thread_id = ?"))
                    .bind(thread_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)
        }
    }
}

#[cfg(feature = "postgres")]
pub use self::postgres::PostgresSaver;

#[cfg(feature = "postgres")]
mod postgres {
    use async_trait::async_trait;
    use flow_agent_core::{Checkpoint, Checkpointer, ListFilter, PendingWrite, Result};
    use sqlx::postgres::{PgPool, PgPoolOptions};

    use super::{db_err, decode, encode};

    const SCHEMA: &[&str] = &[
        "CREATE TABLE IF NOT EXISTS flow_agent_checkpoints (
            thread_id  TEXT        NOT NULL,
            namespace  TEXT        NOT NULL,
            id         TEXT        NOT NULL,
            parent_id  TEXT,
            step       BIGINT      NOT NULL,
            data       JSONB       NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (thread_id, namespace, id)
        )",
        "CREATE TABLE IF NOT EXISTS flow_agent_writes (
            thread_id     TEXT  NOT NULL,
            namespace     TEXT  NOT NULL,
            checkpoint_id TEXT  NOT NULL,
            task_id       TEXT  NOT NULL,
            data          JSONB NOT NULL,
            PRIMARY KEY (thread_id, namespace, checkpoint_id, task_id)
        )",
    ];

    /// Stores checkpoints in Postgres (`JSONB`). For multi-instance services.
    ///
    /// ```no_run
    /// # async fn demo() -> flow_agent_core::Result<()> {
    /// use flow_agent_checkpoint::PostgresSaver;
    /// let saver = PostgresSaver::connect("postgres://user:pass@localhost/app").await?;
    /// saver.migrate().await?;
    /// # Ok(()) }
    /// ```
    #[derive(Clone, Debug)]
    pub struct PostgresSaver {
        pool: PgPool,
    }

    impl PostgresSaver {
        /// Connect with a default pool (10 connections).
        pub async fn connect(url: &str) -> Result<Self> {
            let pool = PgPoolOptions::new()
                .max_connections(10)
                .connect(url)
                .await
                .map_err(db_err)?;
            Ok(PostgresSaver { pool })
        }

        /// Use an existing pool.
        pub fn from_pool(pool: PgPool) -> Self {
            PostgresSaver { pool }
        }

        /// The underlying pool.
        pub fn pool(&self) -> &PgPool {
            &self.pool
        }

        /// Create the tables if they don't exist. Safe to run repeatedly.
        pub async fn migrate(&self) -> Result<()> {
            for stmt in SCHEMA {
                sqlx::query(stmt)
                    .execute(&self.pool)
                    .await
                    .map_err(db_err)?;
            }
            Ok(())
        }

        async fn writes(&self, thread: &str, ns: &str, id: &str) -> Result<Vec<String>> {
            sqlx::query_scalar(
                "SELECT data::text FROM flow_agent_writes
                 WHERE thread_id = $1 AND namespace = $2 AND checkpoint_id = $3 ORDER BY task_id",
            )
            .bind(thread)
            .bind(ns)
            .bind(id)
            .fetch_all(&self.pool)
            .await
            .map_err(db_err)
        }
    }

    #[async_trait]
    impl Checkpointer for PostgresSaver {
        async fn put(&self, cp: &Checkpoint) -> Result<()> {
            sqlx::query(
                "INSERT INTO flow_agent_checkpoints
                 (thread_id, namespace, id, parent_id, step, data, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::timestamptz)
                 ON CONFLICT (thread_id, namespace, id)
                 DO UPDATE SET data = EXCLUDED.data, step = EXCLUDED.step",
            )
            .bind(&cp.thread_id)
            .bind(&cp.namespace)
            .bind(&cp.id)
            .bind(&cp.parent_id)
            .bind(i64::from(cp.step))
            .bind(encode(cp)?)
            .bind(cp.created_at.to_rfc3339())
            .execute(&self.pool)
            .await
            .map_err(db_err)?;
            Ok(())
        }

        async fn put_writes(
            &self,
            thread_id: &str,
            namespace: &str,
            checkpoint_id: &str,
            writes: &[PendingWrite],
        ) -> Result<()> {
            let mut tx = self.pool.begin().await.map_err(db_err)?;
            for w in writes {
                sqlx::query(
                    "INSERT INTO flow_agent_writes
                     (thread_id, namespace, checkpoint_id, task_id, data)
                     VALUES ($1, $2, $3, $4, $5::jsonb)
                     ON CONFLICT (thread_id, namespace, checkpoint_id, task_id)
                     DO UPDATE SET data = EXCLUDED.data",
                )
                .bind(thread_id)
                .bind(namespace)
                .bind(checkpoint_id)
                .bind(&w.task_id)
                .bind(serde_json::to_string(w)?)
                .execute(&mut *tx)
                .await
                .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)
        }

        async fn get(
            &self,
            thread_id: &str,
            namespace: &str,
            checkpoint_id: Option<&str>,
        ) -> Result<Option<Checkpoint>> {
            let row: Option<(String, String)> = match checkpoint_id {
                Some(id) => sqlx::query_as(
                    "SELECT id, data::text FROM flow_agent_checkpoints
                     WHERE thread_id = $1 AND namespace = $2 AND id = $3",
                )
                .bind(thread_id)
                .bind(namespace)
                .bind(id),
                None => sqlx::query_as(
                    "SELECT id, data::text FROM flow_agent_checkpoints
                     WHERE thread_id = $1 AND namespace = $2 ORDER BY id DESC LIMIT 1",
                )
                .bind(thread_id)
                .bind(namespace),
            }
            .fetch_optional(&self.pool)
            .await
            .map_err(db_err)?;
            match row {
                Some((id, data)) => {
                    let writes = self.writes(thread_id, namespace, &id).await?;
                    decode(&data, writes).map(Some)
                }
                None => Ok(None),
            }
        }

        async fn list(&self, thread_id: &str, filter: ListFilter) -> Result<Vec<Checkpoint>> {
            let ns = filter.namespace.as_deref().unwrap_or("");
            let limit = filter.limit.map(|l| i64::try_from(l).unwrap_or(i64::MAX));
            let rows: Vec<(String, String)> = sqlx::query_as(
                "SELECT id, data::text FROM flow_agent_checkpoints
                 WHERE thread_id = $1 AND namespace = $2 AND ($3::text IS NULL OR id < $3)
                 ORDER BY id DESC LIMIT $4",
            )
            .bind(thread_id)
            .bind(ns)
            .bind(&filter.before)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
            .map_err(db_err)?;
            let mut out = Vec::with_capacity(rows.len());
            for (id, data) in rows {
                let writes = self.writes(thread_id, ns, &id).await?;
                out.push(decode(&data, writes)?);
            }
            Ok(out)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            let mut tx = self.pool.begin().await.map_err(db_err)?;
            for table in ["flow_agent_checkpoints", "flow_agent_writes"] {
                sqlx::query(&format!("DELETE FROM {table} WHERE thread_id = $1"))
                    .bind(thread_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)
        }
    }
}
