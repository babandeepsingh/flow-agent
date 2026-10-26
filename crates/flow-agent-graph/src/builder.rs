//! [`StateGraph`]: declare nodes, edges and reducers, then compile.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use flow_agent_core::{Checkpointer, Error, Result};
use serde_json::Value;

use crate::compiled::{CompiledGraph, Inner};
use crate::node::{END, IntoNode, Node, NodeOptions, Route, START};
use crate::reducer::{Reducer, Schema};

pub(crate) type RouterFn = Arc<dyn Fn(&Value) -> Route + Send + Sync>;

pub(crate) struct NodeEntry {
    pub node: Arc<dyn Node>,
    pub opts: NodeOptions,
}

pub(crate) struct Branch {
    pub router: RouterFn,
    pub path_map: Option<BTreeMap<String, String>>,
}

pub(crate) struct WaitEdge {
    pub sources: Vec<String>,
    pub target: String,
}

/// Builds a stateful graph. State is a JSON object; each node returns a
/// partial update that is merged with the key's [`Reducer`].
///
/// ```
/// use flow_agent_graph::{StateGraph, Reducer, NodeContext, START, END};
/// use flow_agent_core::{Runnable, json, Value};
///
/// # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
/// let graph = StateGraph::new()
///     .key("log", Reducer::Append)
///     .add_node("a", |_s: Value, _c: NodeContext| async { Ok(json!({ "log": "a" })) })
///     .add_node("b", |_s: Value, _c: NodeContext| async { Ok(json!({ "log": "b" })) })
///     .add_edge(START, "a")
///     .add_edge("a", "b")
///     .add_edge("b", END)
///     .compile()
///     .unwrap();
///
/// let out = graph.invoke(json!({})).await.unwrap();
/// assert_eq!(out["log"], json!(["a", "b"]));
/// # });
/// ```
pub struct StateGraph {
    name: String,
    schema: Schema,
    nodes: BTreeMap<String, NodeEntry>,
    edges: BTreeMap<String, Vec<String>>,
    branches: BTreeMap<String, Vec<Branch>>,
    waits: Vec<WaitEdge>,
    errors: Vec<String>,
}

impl Default for StateGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StateGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateGraph")
            .field("name", &self.name)
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .finish()
    }
}

impl StateGraph {
    /// An empty graph.
    pub fn new() -> Self {
        StateGraph {
            name: "graph".into(),
            schema: Schema::default(),
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            branches: BTreeMap::new(),
            waits: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Name used in traces and events.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Declare how updates to `key` are merged.
    pub fn key(mut self, key: impl Into<String>, reducer: Reducer) -> Self {
        let key = key.into();
        if self.schema.reducers.insert(key.clone(), reducer).is_some() {
            self.errors
                .push(format!("reducer for key '{key}' declared twice"));
        }
        self
    }

    /// Add a node: an async closure `|state: Value, ctx: NodeContext| async { ... }`
    /// returning a partial update (`Value`) or a [`Command`](crate::Command), or any
    /// [`Runnable`](flow_agent_core::Runnable) (models, chains, subgraphs).
    pub fn add_node<M>(self, name: impl Into<String>, node: impl IntoNode<M>) -> Self {
        self.add_node_with(name, node, NodeOptions::default())
    }

    /// Add a node with a timeout, retry policy or declared `Command` targets.
    pub fn add_node_with<M>(
        mut self,
        name: impl Into<String>,
        node: impl IntoNode<M>,
        options: NodeOptions,
    ) -> Self {
        let name = name.into();
        if name.is_empty() || name == START || name == END || name.contains(':') {
            self.errors.push(format!(
                "invalid node name '{name}' (reserved, empty or contains ':')"
            ));
            return self;
        }
        if self.nodes.contains_key(&name) {
            self.errors.push(format!("node '{name}' added twice"));
            return self;
        }
        self.nodes.insert(
            name,
            NodeEntry {
                node: node.into_node(),
                opts: options,
            },
        );
        self
    }

    /// Always go from `from` to `to`. Several edges from one node run their
    /// targets in parallel.
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.entry(from.into()).or_default().push(to.into());
        self
    }

    /// Run `to` once **all** of `sources` have finished (fan-in).
    pub fn add_edge_from<I, S>(mut self, sources: I, to: impl Into<String>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.waits.push(WaitEdge {
            sources: sources.into_iter().map(Into::into).collect(),
            target: to.into(),
        });
        self
    }

    /// Choose the next node(s) after `source` from the state. The router returns
    /// a node name, [`END`], several names (run in parallel) or [`Send`](crate::Send)s.
    pub fn add_conditional_edges<F, R>(mut self, source: impl Into<String>, router: F) -> Self
    where
        F: Fn(&Value) -> R + Send + Sync + 'static,
        R: Into<Route>,
    {
        self.branches
            .entry(source.into())
            .or_default()
            .push(Branch {
                router: Arc::new(move |s| router(s).into()),
                path_map: None,
            });
        self
    }

    /// Like [`add_conditional_edges`](Self::add_conditional_edges), but the router
    /// returns keys that `path_map` maps to node names. Targets are checked at
    /// compile time.
    pub fn add_conditional_edges_with<F, R, I, K, V>(
        mut self,
        source: impl Into<String>,
        router: F,
        path_map: I,
    ) -> Self
    where
        F: Fn(&Value) -> R + Send + Sync + 'static,
        R: Into<Route>,
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.branches
            .entry(source.into())
            .or_default()
            .push(Branch {
                router: Arc::new(move |s| router(s).into()),
                path_map: Some(
                    path_map
                        .into_iter()
                        .map(|(k, v)| (k.into(), v.into()))
                        .collect(),
                ),
            });
        self
    }

    /// Validate and compile without a checkpointer.
    pub fn compile(self) -> Result<CompiledGraph> {
        self.compile_with(CompileOptions::default())
    }

    /// Validate and compile. All problems are reported at once in
    /// [`Error::InvalidGraph`].
    pub fn compile_with(self, options: CompileOptions) -> Result<CompiledGraph> {
        let mut errors = self.errors.clone();
        let exists = |n: &str| self.nodes.contains_key(n);
        let target_ok = |n: &str| n == END || exists(n);

        if !self.edges.contains_key(START) && !self.branches.contains_key(START) {
            errors.push("no entry point: add_edge(START, \"<node>\")".into());
        }
        for (from, tos) in &self.edges {
            if from == END {
                errors.push("edges cannot start at END".into());
            } else if from != START && !exists(from) {
                errors.push(format!("edge from unknown node '{from}'"));
            }
            for to in tos {
                if to == START {
                    errors.push(format!("edge from '{from}' into START"));
                } else if !target_ok(to) {
                    errors.push(format!("edge from '{from}' to unknown node '{to}'"));
                }
            }
        }
        for (from, branches) in &self.branches {
            if from != START && !exists(from) {
                errors.push(format!("conditional edge from unknown node '{from}'"));
            }
            for b in branches {
                for to in b.path_map.iter().flat_map(|m| m.values()) {
                    if !target_ok(to) {
                        errors.push(format!(
                            "conditional edge from '{from}' maps to unknown node '{to}'"
                        ));
                    }
                }
            }
        }
        for w in &self.waits {
            if w.sources.is_empty() {
                errors.push(format!("add_edge_from into '{}' has no sources", w.target));
            }
            for s in &w.sources {
                if !exists(s) {
                    errors.push(format!("add_edge_from: unknown source node '{s}'"));
                }
            }
            if !target_ok(&w.target) {
                errors.push(format!("add_edge_from: unknown target node '{}'", w.target));
            }
        }
        for (name, entry) in &self.nodes {
            for t in &entry.opts.routes_to {
                if !target_ok(t) {
                    errors.push(format!("node '{name}' routes_to unknown node '{t}'"));
                }
            }
            let has_out = self.edges.contains_key(name)
                || self.branches.contains_key(name)
                || self.waits.iter().any(|w| w.sources.contains(name))
                || !entry.opts.routes_to.is_empty();
            if !has_out {
                errors.push(format!(
                    "node '{name}' has no outgoing edge; add_edge(\"{name}\", END) if it should finish the run"
                ));
            }
        }
        for n in options
            .interrupt_before
            .iter()
            .chain(options.interrupt_after.iter())
        {
            if !exists(n) {
                errors.push(format!("interrupt on unknown node '{n}'"));
            }
        }

        // Reachability (skipped if a router without a path map can go anywhere).
        let dynamic = self
            .branches
            .values()
            .flatten()
            .any(|b| b.path_map.is_none());
        if !dynamic && errors.is_empty() {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            let mut queue: VecDeque<&str> = VecDeque::from([START]);
            while let Some(n) = queue.pop_front() {
                if !seen.insert(n) {
                    continue;
                }
                let mut next: Vec<&str> = Vec::new();
                next.extend(self.edges.get(n).into_iter().flatten().map(String::as_str));
                for b in self.branches.get(n).into_iter().flatten() {
                    next.extend(
                        b.path_map
                            .iter()
                            .flat_map(|m| m.values())
                            .map(String::as_str),
                    );
                }
                if let Some(e) = self.nodes.get(n) {
                    next.extend(e.opts.routes_to.iter().map(String::as_str));
                }
                for w in &self.waits {
                    if w.sources.iter().any(|s| s == n) {
                        next.push(&w.target);
                    }
                }
                queue.extend(next);
            }
            for n in self.nodes.keys() {
                if !seen.contains(n.as_str()) {
                    errors.push(format!("node '{n}' is unreachable from START"));
                }
            }
        }

        if !errors.is_empty() {
            return Err(Error::InvalidGraph(errors));
        }

        Ok(CompiledGraph::new(Inner {
            name: self.name,
            schema: self.schema,
            nodes: self.nodes,
            edges: self.edges,
            branches: self.branches,
            waits: self.waits,
            checkpointer: options.checkpointer,
            interrupt_before: options.interrupt_before,
            interrupt_after: options.interrupt_after,
        }))
    }
}

/// Options for [`StateGraph::compile_with`].
///
/// ```ignore
/// let app = graph.compile_with(
///     CompileOptions::new()
///         .checkpointer(Arc::new(MemorySaver::new()))
///         .interrupt_before(["tools"]),
/// )?;
/// ```
#[derive(Clone, Default)]
pub struct CompileOptions {
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) interrupt_before: BTreeSet<String>,
    pub(crate) interrupt_after: BTreeSet<String>,
}

impl std::fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompileOptions")
            .field("checkpointer", &self.checkpointer.is_some())
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .finish()
    }
}

impl CompileOptions {
    /// No checkpointer, no breakpoints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save a checkpoint after every superstep (enables resume, interrupts,
    /// state inspection and replay). Runs then need `RunConfig::thread_id`.
    pub fn checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Pause before these nodes run.
    pub fn interrupt_before<I, S>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.interrupt_before
            .extend(nodes.into_iter().map(Into::into));
        self
    }

    /// Pause after these nodes run.
    pub fn interrupt_after<I, S>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.interrupt_after
            .extend(nodes.into_iter().map(Into::into));
        self
    }
}
