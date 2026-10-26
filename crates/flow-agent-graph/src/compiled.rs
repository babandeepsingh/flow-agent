//! [`CompiledGraph`]: the Pregel-style executor.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use async_trait::async_trait;
use flow_agent_core::runnable::{EventStream, stream_owned};
use flow_agent_core::{
    Checkpoint, Checkpointer, Error, Event, Interrupt, ListFilter, PendingWrite, Result, RunConfig,
    Runnable,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::Instrument;
use uuid::Uuid;

use crate::builder::{Branch, NodeEntry, WaitEdge};
use crate::node::{END, NodeContext, NodeOutput, Route, RouteTarget, START, as_interrupt};
use crate::reducer::Schema;

/// Key under which an interrupted run reports its interrupts.
pub const INTERRUPT_KEY: &str = "__interrupt__";
const META_KEY: &str = "flow_agent";

pub(crate) struct Inner {
    pub name: String,
    pub schema: Schema,
    pub nodes: BTreeMap<String, NodeEntry>,
    pub edges: BTreeMap<String, Vec<String>>,
    pub branches: BTreeMap<String, Vec<Branch>>,
    pub waits: Vec<WaitEdge>,
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    pub interrupt_before: BTreeSet<String>,
    pub interrupt_after: BTreeSet<String>,
}

/// A node execution scheduled for a superstep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TaskSpec {
    node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arg: Option<Value>,
}

/// Scheduling state persisted in checkpoint metadata.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Pending {
    tasks: Vec<TaskSpec>,
    #[serde(default)]
    barriers: BTreeMap<usize, BTreeSet<String>>,
    #[serde(default)]
    resume: BTreeMap<usize, Vec<Value>>,
    #[serde(default)]
    interrupted_before: bool,
}

/// A finished task's output.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskWrite {
    node: String,
    update: Value,
    #[serde(default)]
    goto: Route,
}

/// Result of a graph run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunOutput {
    /// Final (or, when interrupted, current) state.
    pub state: Value,
    /// Open interrupts; empty when the run completed.
    pub interrupts: Vec<Interrupt>,
    /// Latest checkpoint, when a checkpointer is configured.
    pub checkpoint_id: Option<String>,
}

impl RunOutput {
    /// Whether the run paused for human input.
    pub fn is_interrupted(&self) -> bool {
        !self.interrupts.is_empty()
    }

    /// State as JSON, with `"__interrupt__": [...]` added when interrupted.
    pub fn into_value(self) -> Value {
        let mut state = self.state;
        if !self.interrupts.is_empty() {
            if let Value::Object(m) = &mut state {
                m.insert(INTERRUPT_KEY.into(), json!(self.interrupts));
            }
        }
        state
    }
}

/// A thread's state at one checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// State values.
    pub values: Value,
    /// Nodes that will run next (empty when finished).
    pub next: Vec<String>,
    /// Open interrupts.
    pub interrupts: Vec<Interrupt>,
    /// This checkpoint's id (use with `RunConfig::checkpoint_id` to replay or fork).
    pub checkpoint_id: String,
    /// Previous checkpoint.
    pub parent_checkpoint_id: Option<String>,
    /// Superstep.
    pub step: u32,
    /// Checkpoint metadata (`source`: `input`, `loop`, `update`, `error`).
    pub metadata: Map<String, Value>,
    /// Creation time (RFC 3339).
    pub created_at: String,
}

impl StateSnapshot {
    fn from_checkpoint(cp: Checkpoint) -> Self {
        let mut metadata = cp.metadata;
        metadata.remove(META_KEY);
        StateSnapshot {
            values: cp.state,
            next: cp.next,
            interrupts: cp.interrupts,
            checkpoint_id: cp.id,
            parent_checkpoint_id: cp.parent_id,
            step: cp.step,
            metadata,
            created_at: cp.created_at.to_rfc3339(),
        }
    }
}

enum Start {
    Input(Value),
    Resume(ResumeWith),
}

enum ResumeWith {
    All(Value),
    ById(BTreeMap<String, Value>),
}

/// A compiled, runnable graph. Cheap to clone and `'static`; implements
/// [`Runnable`], so it can be used as a node in another graph.
#[derive(Clone)]
pub struct CompiledGraph {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("name", &self.inner.name)
            .field("nodes", &self.inner.nodes.keys().collect::<Vec<_>>())
            .field("checkpointer", &self.inner.checkpointer.is_some())
            .finish()
    }
}

struct Saver<'a> {
    saver: Option<(Arc<dyn Checkpointer>, String)>,
    parent_id: Option<String>,
    step: u32,
    config: &'a RunConfig,
}

impl Saver<'_> {
    async fn save(
        &mut self,
        state: &Map<String, Value>,
        pending: &Pending,
        writes: &BTreeMap<usize, TaskWrite>,
        interrupts: &[Interrupt],
        source: &str,
    ) -> Result<()> {
        let Some((saver, thread)) = &self.saver else {
            return Ok(());
        };
        let mut cp = Checkpoint::new(thread.clone(), self.step, Value::Object(state.clone()));
        cp.parent_id = self.parent_id.clone();
        cp.next = pending.tasks.iter().map(|t| t.node.clone()).collect();
        cp.interrupts = interrupts.to_vec();
        cp.pending_writes = writes
            .iter()
            .map(|(i, w)| PendingWrite {
                task_id: i.to_string(),
                node: w.node.clone(),
                update: serde_json::to_value(w).unwrap_or(Value::Null),
            })
            .collect();
        cp.metadata.insert("source".into(), json!(source));
        cp.metadata
            .insert("run_id".into(), json!(self.config.run_id.to_string()));
        cp.metadata
            .insert(META_KEY.into(), serde_json::to_value(pending)?);
        saver
            .put(&cp)
            .await
            .map_err(|e| e.context("saving checkpoint"))?;
        self.config
            .emit(Event::Checkpoint {
                checkpoint_id: cp.id.clone(),
                step: cp.step,
            })
            .await;
        self.parent_id = Some(cp.id);
        Ok(())
    }
}

fn pending_of(cp: &Checkpoint) -> Result<Pending> {
    match cp.metadata.get(META_KEY) {
        Some(v) => Ok(serde_json::from_value(v.clone())?),
        None => Ok(Pending {
            tasks: cp
                .next
                .iter()
                .map(|n| TaskSpec {
                    node: n.clone(),
                    arg: None,
                })
                .collect(),
            ..Default::default()
        }),
    }
}

fn writes_of(cp: &Checkpoint) -> Result<BTreeMap<usize, TaskWrite>> {
    cp.pending_writes
        .iter()
        .map(|w| {
            let i = w
                .task_id
                .parse()
                .map_err(|_| Error::Checkpoint(format!("bad pending write id '{}'", w.task_id)))?;
            Ok((i, serde_json::from_value(w.update.clone())?))
        })
        .collect()
}

fn static_interrupt(when: &str, node: &str) -> Interrupt {
    Interrupt {
        id: format!("{when}:{node}"),
        node: node.to_owned(),
        value: json!({ "when": when }),
    }
}

/// Task index encoded in a dynamic interrupt id (`"{step}.{task}.{n}"`).
fn task_of_interrupt(id: &str) -> Option<usize> {
    id.split('.').nth(1)?.parse().ok()
}

impl CompiledGraph {
    pub(crate) fn new(inner: Inner) -> Self {
        CompiledGraph {
            inner: Arc::new(inner),
        }
    }

    /// Graph name.
    pub fn graph_name(&self) -> &str {
        &self.inner.name
    }

    /// Node names.
    pub fn node_names(&self) -> Vec<&str> {
        self.inner.nodes.keys().map(String::as_str).collect()
    }

    /// Run and return a typed result (state, interrupts, checkpoint id).
    pub async fn run(&self, input: Value, config: &RunConfig) -> Result<RunOutput> {
        self.execute_guarded(Start::Input(input), config).await
    }

    /// Resume an interrupted thread: every open interrupt receives `value`.
    /// Returns the state like `invoke` (with `__interrupt__` if it pauses again).
    pub async fn resume(&self, value: Value, config: &RunConfig) -> Result<Value> {
        self.resume_run(value, config)
            .await
            .map(RunOutput::into_value)
    }

    /// Like [`resume`](Self::resume), returning a [`RunOutput`].
    pub async fn resume_run(&self, value: Value, config: &RunConfig) -> Result<RunOutput> {
        self.execute_guarded(Start::Resume(ResumeWith::All(value)), config)
            .await
    }

    /// Resume with a different value per interrupt id (when several nodes
    /// interrupted in the same step).
    pub async fn resume_each(
        &self,
        values: BTreeMap<String, Value>,
        config: &RunConfig,
    ) -> Result<RunOutput> {
        self.execute_guarded(Start::Resume(ResumeWith::ById(values)), config)
            .await
    }

    /// Stream events with an owned (`'static`) stream, e.g. for an SSE response.
    pub fn stream_owned(&self, input: Value, config: &RunConfig) -> EventStream<'static> {
        stream_owned(Arc::new(self.clone()), input, config)
    }

    fn saver_for(&self, config: &RunConfig) -> Result<(Arc<dyn Checkpointer>, String)> {
        let saver = self.inner.checkpointer.clone().ok_or_else(|| {
            Error::Config(
                "this operation needs a checkpointer: CompileOptions::checkpointer".into(),
            )
        })?;
        let thread = config
            .thread_id
            .clone()
            .ok_or_else(|| Error::Config("RunConfig::thread_id is required".into()))?;
        Ok((saver, thread))
    }

    /// Current state of a thread (or of `config.checkpoint_id`).
    pub async fn get_state(&self, config: &RunConfig) -> Result<Option<StateSnapshot>> {
        let (saver, thread) = self.saver_for(config)?;
        Ok(saver
            .get(&thread, "", config.checkpoint_id.as_deref())
            .await?
            .map(StateSnapshot::from_checkpoint))
    }

    /// All checkpoints of a thread, newest first.
    pub async fn get_state_history(&self, config: &RunConfig) -> Result<Vec<StateSnapshot>> {
        let (saver, thread) = self.saver_for(config)?;
        Ok(saver
            .list(&thread, ListFilter::default())
            .await?
            .into_iter()
            .map(StateSnapshot::from_checkpoint)
            .collect())
    }

    /// Edit a thread's state as if `as_node` had returned `values` (its edges
    /// then decide what runs next). With `as_node = None` the pending work is
    /// kept. Returns the new checkpoint id.
    pub async fn update_state(
        &self,
        config: &RunConfig,
        values: Value,
        as_node: Option<&str>,
    ) -> Result<String> {
        let (saver, thread) = self.saver_for(config)?;
        let g = &*self.inner;
        let cp = saver
            .get(&thread, "", config.checkpoint_id.as_deref())
            .await?;
        let (mut state, mut pending, step, parent, interrupts, writes) = match &cp {
            Some(cp) => (
                cp.state.as_object().cloned().unwrap_or_default(),
                pending_of(cp)?,
                cp.step + 1,
                Some(cp.id.clone()),
                cp.interrupts.clone(),
                writes_of(cp)?,
            ),
            None => (
                Map::new(),
                Pending::default(),
                0,
                None,
                vec![],
                BTreeMap::new(),
            ),
        };
        let writer = as_node.unwrap_or("__update__");
        g.schema.apply(&mut state, &[(writer.to_owned(), values)])?;
        let (interrupts, writes) = match as_node {
            Some(node) => {
                if !g.nodes.contains_key(node) {
                    return Err(Error::Config(format!(
                        "update_state: unknown node '{node}'"
                    )));
                }
                let sv = Value::Object(state.clone());
                pending.tasks = self.next_after(node, &Route::default(), &sv)?;
                pending.resume.clear();
                pending.interrupted_before = false;
                (vec![], BTreeMap::new())
            }
            None => (interrupts, writes),
        };
        let mut s = Saver {
            saver: Some((saver, thread)),
            parent_id: parent,
            step,
            config,
        };
        s.save(&state, &pending, &writes, &interrupts, "update")
            .await?;
        s.parent_id
            .ok_or_else(|| Error::Checkpoint("checkpoint was not saved".into()))
    }

    /// Mermaid flowchart of the graph.
    pub fn to_mermaid(&self) -> String {
        let g = &*self.inner;
        let id = |n: &str| match n {
            START => "__start__".to_owned(),
            END => "__end__".to_owned(),
            other => other.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
        };
        let mut out = String::from("graph TD\n    __start__([start])\n    __end__([end])\n");
        for n in g.nodes.keys() {
            out.push_str(&format!("    {}[\"{n}\"]\n", id(n)));
        }
        for (from, tos) in &g.edges {
            for to in tos {
                out.push_str(&format!("    {} --> {}\n", id(from), id(to)));
            }
        }
        for (from, branches) in &g.branches {
            for b in branches {
                match &b.path_map {
                    Some(m) => {
                        for (k, to) in m {
                            out.push_str(&format!("    {} -.->|{k}| {}\n", id(from), id(to)));
                        }
                    }
                    None => out.push_str(&format!("    {} -.-> ?\n", id(from))),
                }
            }
        }
        for w in &g.waits {
            for s in &w.sources {
                out.push_str(&format!("    {} ==> {}\n", id(s), id(&w.target)));
            }
        }
        for (name, e) in &g.nodes {
            for to in &e.opts.routes_to {
                out.push_str(&format!("    {} -.-> {}\n", id(name), id(to)));
            }
        }
        out
    }

    /// Targets after `node` finished with `goto`, evaluated on `state`.
    fn next_after(&self, node: &str, goto: &Route, state: &Value) -> Result<Vec<TaskSpec>> {
        let g = &*self.inner;
        let mut out = Vec::new();
        let push = |target: RouteTarget, out: &mut Vec<TaskSpec>| -> Result<()> {
            let (name, arg) = match target {
                RouteTarget::Node(n) => (n, None),
                RouteTarget::Send(s) => (s.node, Some(s.arg)),
            };
            if name == END {
                return Ok(());
            }
            if !g.nodes.contains_key(&name) {
                return Err(Error::InvalidGraph(vec![format!(
                    "'{node}' routed to unknown node '{name}'"
                )]));
            }
            out.push(TaskSpec { node: name, arg });
            Ok(())
        };
        for t in g.edges.get(node).into_iter().flatten() {
            push(RouteTarget::Node(t.clone()), &mut out)?;
        }
        for b in g.branches.get(node).into_iter().flatten() {
            for t in (b.router)(state).0 {
                let t = match (t, &b.path_map) {
                    (RouteTarget::Node(key), Some(map)) => {
                        RouteTarget::Node(map.get(&key).cloned().ok_or_else(|| {
                            Error::InvalidGraph(vec![format!(
                                "router of '{node}' returned '{key}', which is not in its path map"
                            )])
                        })?)
                    }
                    (t, _) => t,
                };
                push(t, &mut out)?;
            }
        }
        for t in &goto.0 {
            push(t.clone(), &mut out)?;
        }
        Ok(out)
    }

    async fn execute_guarded(&self, start: Start, config: &RunConfig) -> Result<RunOutput> {
        let span = tracing::info_span!(
            "flow_agent.graph",
            graph = %self.inner.name,
            thread_id = config.thread_id.as_deref().unwrap_or(""),
            run_id = %config.run_id,
        );
        let fut = self.execute(start, config).instrument(span);
        match config.timeout {
            Some(t) => tokio::time::timeout(t, fut)
                .await
                .map_err(|_| Error::Timeout {
                    what: format!("graph '{}'", self.inner.name),
                    after: t,
                })?,
            None => fut.await,
        }
    }

    async fn execute(&self, start: Start, config: &RunConfig) -> Result<RunOutput> {
        let g = &*self.inner;
        let saver = match (&g.checkpointer, &config.thread_id) {
            (Some(s), Some(t)) => Some((s.clone(), t.clone())),
            (Some(_), None) => {
                return Err(Error::Config(
                    "this graph has a checkpointer: set RunConfig::thread_id".into(),
                ));
            }
            (None, _) => None,
        };
        let loaded = match &saver {
            Some((s, t)) => s.get(t, "", config.checkpoint_id.as_deref()).await?,
            None => None,
        };
        if config.checkpoint_id.is_some() && loaded.is_none() {
            return Err(Error::Checkpoint(format!(
                "checkpoint '{}' not found",
                config.checkpoint_id.as_deref().unwrap_or_default()
            )));
        }
        let mut saver = Saver {
            saver,
            parent_id: loaded.as_ref().map(|c| c.id.clone()),
            step: loaded.as_ref().map(|c| c.step).unwrap_or(0),
            config,
        };

        let (mut state, mut pending, mut done) = match (start, loaded) {
            (Start::Input(input), Some(cp)) if input.is_null() => {
                // `invoke(null)` on a saved thread continues (or replays) from the checkpoint.
                let mut pending = pending_of(&cp)?;
                if !cp.interrupts.iter().any(|i| i.id.starts_with("before:")) {
                    pending.interrupted_before = false;
                }
                let done = writes_of(&cp)?;
                (
                    cp.state.as_object().cloned().unwrap_or_default(),
                    pending,
                    done,
                )
            }
            (Start::Input(input), loaded) => {
                let mut state = loaded
                    .as_ref()
                    .and_then(|c| c.state.as_object().cloned())
                    .unwrap_or_default();
                let input = match input {
                    // Shorthand: `invoke("hi")` on a graph with a `messages` key.
                    Value::String(_) | Value::Array(_)
                        if g.schema.reducers.contains_key("messages") =>
                    {
                        json!({ "messages": input })
                    }
                    other => other,
                };
                g.schema
                    .apply(&mut state, &[(START.to_owned(), input)])
                    .map_err(|e| e.context("applying graph input"))?;
                let pending = Pending {
                    tasks: self.next_after(
                        START,
                        &Route::default(),
                        &Value::Object(state.clone()),
                    )?,
                    ..Default::default()
                };
                if loaded.is_some() {
                    saver.step += 1;
                }
                saver
                    .save(&state, &pending, &BTreeMap::new(), &[], "input")
                    .await?;
                (state, pending, BTreeMap::new())
            }
            (Start::Resume(with), loaded) => {
                let cp = loaded.ok_or_else(|| {
                    Error::Config("nothing to resume: no checkpoint for this thread".into())
                })?;
                if cp.interrupts.is_empty() {
                    return Err(Error::Config(
                        "nothing to resume: the thread has no open interrupt".into(),
                    ));
                }
                let mut pending = pending_of(&cp)?;
                for i in &cp.interrupts {
                    let Some(task) = task_of_interrupt(&i.id) else {
                        continue; // static breakpoint: just continue
                    };
                    let value = match &with {
                        ResumeWith::All(v) => Some(v.clone()),
                        ResumeWith::ById(m) => m.get(&i.id).cloned(),
                    };
                    if let Some(v) = value {
                        pending.resume.entry(task).or_default().push(v);
                    }
                }
                let done = writes_of(&cp)?;
                (
                    cp.state.as_object().cloned().unwrap_or_default(),
                    pending,
                    done,
                )
            }
        };

        let mut steps_this_run = 0u32;
        loop {
            if pending.tasks.is_empty() {
                break;
            }
            if steps_this_run >= config.recursion_limit {
                return Err(Error::RecursionLimit {
                    limit: config.recursion_limit,
                });
            }
            config.check_cancelled()?;

            // Static breakpoints before nodes.
            let skip_before = std::mem::take(&mut pending.interrupted_before);
            if !skip_before && done.is_empty() {
                let mut hits: Vec<Interrupt> = Vec::new();
                for t in &pending.tasks {
                    if g.interrupt_before.contains(&t.node)
                        && !hits.iter().any(|h| h.node == t.node)
                    {
                        hits.push(static_interrupt("before", &t.node));
                    }
                }
                if !hits.is_empty() {
                    pending.interrupted_before = true;
                    return self.pause(&mut saver, state, &pending, &done, hits).await;
                }
            }

            // Run the superstep.
            let snapshot = Value::Object(state.clone());
            let to_run: Vec<(usize, TaskSpec)> = pending
                .tasks
                .iter()
                .cloned()
                .enumerate()
                .filter(|(i, _)| !done.contains_key(i))
                .collect();
            let mut interrupts = Vec::new();
            let mut failure = None;
            {
                let step = saver.step;
                let pending_ref = &pending;
                let snapshot = &snapshot;
                let mut results = futures::stream::iter(to_run)
                    .map(|(i, t)| async move {
                        let node = t.node.clone();
                        (
                            i,
                            node,
                            self.run_task(i, t, snapshot, pending_ref, step, config)
                                .await,
                        )
                    })
                    .buffer_unordered(config.max_concurrency.max(1));
                while let Some((i, node, res)) = results.next().await {
                    match res {
                        Ok(w) => {
                            done.insert(i, w);
                        }
                        Err(e) => match as_interrupt(&e) {
                            Some(int) => interrupts.push(Interrupt {
                                id: int.id.clone(),
                                node,
                                value: int.value.clone(),
                            }),
                            None => {
                                failure = Some(e);
                                break;
                            }
                        },
                    }
                }
            }
            if let Some(e) = failure {
                // Keep finished work so a later run can continue from here.
                saver.save(&state, &pending, &done, &[], "error").await.ok();
                return Err(e);
            }
            if !interrupts.is_empty() {
                interrupts.sort_by(|a, b| a.id.cmp(&b.id));
                return self
                    .pause(&mut saver, state, &pending, &done, interrupts)
                    .await;
            }

            // Apply writes deterministically (by node name, then task order).
            let mut ordered: Vec<(&usize, &TaskWrite)> = done.iter().collect();
            ordered.sort_by(|a, b| a.1.node.cmp(&b.1.node).then(a.0.cmp(b.0)));
            let updates: Vec<(String, Value)> = ordered
                .iter()
                .map(|(_, w)| (w.node.clone(), w.update.clone()))
                .collect();
            g.schema
                .apply(&mut state, &updates)
                .map_err(|e| e.context(format!("applying updates of step {}", saver.step)))?;
            let ran: Vec<String> = ordered.iter().map(|(_, w)| w.node.clone()).collect();
            let snapshot = Value::Object(state.clone());
            config
                .emit(Event::StateUpdate {
                    step: saver.step,
                    nodes: ran.clone(),
                    values: snapshot.clone(),
                })
                .await;

            // Schedule the next superstep.
            let mut next: Vec<TaskSpec> = Vec::new();
            for (_, w) in &ordered {
                next.extend(self.next_after(&w.node, &w.goto, &snapshot)?);
            }
            for (wi, wait) in g.waits.iter().enumerate() {
                let finished = pending.barriers.entry(wi).or_default();
                for n in &ran {
                    if wait.sources.contains(n) {
                        finished.insert(n.clone());
                    }
                }
                if wait.sources.iter().all(|s| finished.contains(s)) {
                    pending.barriers.remove(&wi);
                    if wait.target != END {
                        next.push(TaskSpec {
                            node: wait.target.clone(),
                            arg: None,
                        });
                    }
                }
            }
            pending.barriers.retain(|_, s| !s.is_empty());
            let mut seen = BTreeSet::new();
            next.retain(|t| t.arg.is_some() || seen.insert(t.node.clone()));
            pending.tasks = next;
            pending.resume.clear();
            done.clear();
            saver.step += 1;
            steps_this_run += 1;

            // Static breakpoints after nodes.
            let mut after: Vec<Interrupt> = Vec::new();
            if !pending.tasks.is_empty() {
                for n in &ran {
                    if g.interrupt_after.contains(n) && !after.iter().any(|h| &h.node == n) {
                        after.push(static_interrupt("after", n));
                    }
                }
            }
            if !after.is_empty() {
                return self.pause(&mut saver, state, &pending, &done, after).await;
            }
            saver.save(&state, &pending, &done, &[], "loop").await?;
        }

        Ok(RunOutput {
            state: Value::Object(state),
            interrupts: Vec::new(),
            checkpoint_id: saver.parent_id,
        })
    }

    async fn pause(
        &self,
        saver: &mut Saver<'_>,
        state: Map<String, Value>,
        pending: &Pending,
        done: &BTreeMap<usize, TaskWrite>,
        interrupts: Vec<Interrupt>,
    ) -> Result<RunOutput> {
        saver
            .save(&state, pending, done, &interrupts, "loop")
            .await?;
        for i in &interrupts {
            saver
                .config
                .emit(Event::Interrupt {
                    step: saver.step,
                    node: i.node.clone(),
                    interrupt_id: i.id.clone(),
                    value: i.value.clone(),
                })
                .await;
        }
        Ok(RunOutput {
            state: Value::Object(state),
            interrupts,
            checkpoint_id: saver.parent_id.clone(),
        })
    }

    async fn run_task(
        &self,
        index: usize,
        spec: TaskSpec,
        state: &Value,
        pending: &Pending,
        step: u32,
        config: &RunConfig,
    ) -> Result<TaskWrite> {
        let entry =
            self.inner.nodes.get(&spec.node).ok_or_else(|| {
                Error::InvalidGraph(vec![format!("unknown node '{}'", spec.node)])
            })?;
        let input = spec.arg.clone().unwrap_or_else(|| state.clone());
        let task_id = Uuid::now_v7();
        let node_cfg = config.child().with_node(spec.node.clone());
        let span = tracing::info_span!(
            "flow_agent.node",
            node = %spec.node,
            step,
            task_id = %task_id,
            attempt = tracing::field::Empty,
        );
        config
            .emit(Event::NodeStart {
                step,
                node: spec.node.clone(),
                task_id,
            })
            .await;
        let started = Instant::now();
        let resume = Arc::new(pending.resume.get(&index).cloned().unwrap_or_default());
        let mut attempt = 1u32;
        let result = loop {
            span.record("attempt", attempt);
            let ctx = NodeContext {
                config: node_cfg.clone(),
                node: spec.node.clone(),
                step,
                task_id,
                interrupt_prefix: format!("{step}.{index}"),
                resume: resume.clone(),
                counter: Arc::new(AtomicUsize::new(0)),
            };
            let run = entry.node.run(input.clone(), ctx);
            let run = async {
                match entry.opts.timeout {
                    Some(t) => tokio::time::timeout(t, run)
                        .await
                        .map_err(|_| Error::Timeout {
                            what: format!("node '{}'", spec.node),
                            after: t,
                        })?,
                    None => run.await,
                }
            };
            let res = tokio::select! {
                r = run.instrument(span.clone()) => r,
                _ = config.cancellation_token().cancelled() => Err(Error::Cancelled),
            };
            match (&res, &entry.opts.retry) {
                (Err(e), Some(p))
                    if as_interrupt(e).is_none()
                        && attempt < p.max_attempts
                        && p.should_retry(e) =>
                {
                    tracing::warn!(parent: &span, error = %e, attempt, "node failed, retrying");
                    tokio::time::sleep(p.delay_for(attempt)).await;
                    attempt += 1;
                }
                _ => break res,
            }
        };
        match result {
            Ok(out) => {
                let (update, goto) = match out {
                    NodeOutput::Update(v) => (v, Route::default()),
                    NodeOutput::Command(c) => (c.update.unwrap_or(Value::Null), c.goto),
                };
                config
                    .emit(Event::NodeEnd {
                        step,
                        node: spec.node.clone(),
                        task_id,
                        update: update.clone(),
                        duration_ms: started.elapsed().as_millis() as u64,
                    })
                    .await;
                Ok(TaskWrite {
                    node: spec.node,
                    update,
                    goto,
                })
            }
            Err(e) if as_interrupt(&e).is_some() => Err(e),
            Err(e) if matches!(e.root(), Error::Cancelled) => Err(Error::Cancelled),
            Err(e) => {
                tracing::warn!(parent: &span, error = %e, "node failed");
                config.emit_error(&e).await;
                Err(Error::NodeFailed {
                    node: spec.node,
                    step,
                    source: Box::new(e),
                })
            }
        }
    }
}

#[async_trait]
impl Runnable for CompiledGraph {
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Runs the graph. The result is the final state; when the run pauses for
    /// human input it also contains `"__interrupt__": [...]`.
    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value> {
        self.run(input, config).await.map(RunOutput::into_value)
    }
}
