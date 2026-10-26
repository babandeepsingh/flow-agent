//! Nodes, routing targets and the context a node runs with.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::{Error, Event, Result, RetryPolicy, RunConfig, Runnable};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

/// Name of the virtual entry node.
pub const START: &str = "__start__";
/// Name of the virtual exit node.
pub const END: &str = "__end__";

/// Run one node again with its own input (map-reduce / fan-out with payloads).
///
/// ```
/// use flow_agent_graph::Send;
/// use serde_json::json;
/// // In a conditional edge: one "summarize" task per document.
/// let route = |state: &serde_json::Value| -> Vec<Send> {
///     state["docs"].as_array().into_iter().flatten()
///         .map(|d| Send::new("summarize", json!({ "doc": d })))
///         .collect()
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Send {
    /// Target node.
    pub node: String,
    /// Input passed to the node instead of the state.
    pub arg: Value,
}

impl Send {
    /// Send `arg` to `node`.
    pub fn new(node: impl Into<String>, arg: Value) -> Self {
        Send {
            node: node.into(),
            arg,
        }
    }
}

/// One routing target: a node name or a [`Send`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RouteTarget {
    /// Run the node on the current state.
    Node(String),
    /// Run the node with a custom input.
    Send(Send),
}

/// Where to go next. Built from `&str`, `String`, `Send` or vectors of them;
/// returning [`END`] (or nothing) stops that path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Route(pub Vec<RouteTarget>);

impl From<&str> for Route {
    fn from(s: &str) -> Self {
        Route(vec![RouteTarget::Node(s.to_owned())])
    }
}
impl From<String> for Route {
    fn from(s: String) -> Self {
        Route(vec![RouteTarget::Node(s)])
    }
}
impl From<&String> for Route {
    fn from(s: &String) -> Self {
        Route(vec![RouteTarget::Node(s.clone())])
    }
}
impl From<Send> for Route {
    fn from(s: Send) -> Self {
        Route(vec![RouteTarget::Send(s)])
    }
}
impl<T: Into<Route>> From<Vec<T>> for Route {
    fn from(v: Vec<T>) -> Self {
        Route(v.into_iter().flat_map(|t| t.into().0).collect())
    }
}
impl<T: Into<Route>> From<Option<T>> for Route {
    fn from(v: Option<T>) -> Self {
        v.map(Into::into).unwrap_or_default()
    }
}

/// Update state and choose the next nodes in one return value.
///
/// ```
/// use flow_agent_graph::Command;
/// use serde_json::json;
/// let cmd = Command::goto("review").update(json!({ "draft": "..." }));
/// ```
///
/// Nodes that route with `Command` must declare their possible targets with
/// [`NodeOptions::routes_to`] so the graph can be validated.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Command {
    /// Partial state update.
    pub update: Option<Value>,
    /// Next nodes (in addition to the node's edges).
    pub goto: Route,
}

impl Command {
    /// Route to `target`.
    pub fn goto(target: impl Into<Route>) -> Self {
        Command {
            update: None,
            goto: target.into(),
        }
    }

    /// Only update state.
    pub fn update_only(update: Value) -> Self {
        Command {
            update: Some(update),
            goto: Route::default(),
        }
    }

    /// Set the state update.
    pub fn update(mut self, update: Value) -> Self {
        self.update = Some(update);
        self
    }
}

/// What a node returns: a partial state update or a [`Command`].
#[derive(Clone, Debug, PartialEq)]
pub enum NodeOutput {
    /// A partial update (`{"key": value}`), or `null` for no change.
    Update(Value),
    /// Update and route.
    Command(Command),
}

impl From<Value> for NodeOutput {
    fn from(v: Value) -> Self {
        NodeOutput::Update(v)
    }
}

impl From<Command> for NodeOutput {
    fn from(c: Command) -> Self {
        NodeOutput::Command(c)
    }
}

impl From<()> for NodeOutput {
    fn from(_: ()) -> Self {
        NodeOutput::Update(Value::Null)
    }
}

/// Raised by [`NodeContext::interrupt`]; carried inside [`Error::Other`] so it
/// travels through `?`. The executor turns it into a paused run.
#[derive(Debug)]
pub(crate) struct GraphInterrupt {
    pub id: String,
    pub value: Value,
}

impl std::fmt::Display for GraphInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "graph interrupted ({})", self.id)
    }
}

impl std::error::Error for GraphInterrupt {}

pub(crate) fn as_interrupt(e: &Error) -> Option<&GraphInterrupt> {
    match e.root() {
        Error::Other(inner) => inner.downcast_ref::<GraphInterrupt>(),
        _ => None,
    }
}

/// Access to the run from inside a node.
#[derive(Clone)]
pub struct NodeContext {
    pub(crate) config: RunConfig,
    pub(crate) node: String,
    pub(crate) step: u32,
    pub(crate) task_id: Uuid,
    pub(crate) interrupt_prefix: String,
    pub(crate) resume: Arc<Vec<Value>>,
    pub(crate) counter: Arc<AtomicUsize>,
}

impl std::fmt::Debug for NodeContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeContext")
            .field("node", &self.node)
            .field("step", &self.step)
            .field("task_id", &self.task_id)
            .finish()
    }
}

impl NodeContext {
    /// A context for calling a node outside a graph (tests).
    pub fn detached(node: impl Into<String>, config: RunConfig) -> Self {
        let node = node.into();
        NodeContext {
            config: config.with_node(node.clone()),
            interrupt_prefix: format!("0.0.{node}"),
            node,
            step: 0,
            task_id: Uuid::now_v7(),
            resume: Arc::new(Vec::new()),
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// The run config (thread id, metadata, callbacks...). Pass it to models
    /// and tools so their events and cancellation are tied to this run.
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// This node's name.
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Current superstep.
    pub fn step(&self) -> u32 {
        self.step
    }

    /// Unique id of this node execution.
    pub fn task_id(&self) -> Uuid {
        self.task_id
    }

    /// Whether the run was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.config.is_cancelled()
    }

    /// Emit an application-defined [`Event::Custom`] (progress, partial results...).
    pub async fn emit(&self, name: impl Into<String>, data: Value) {
        self.config
            .emit(Event::Custom {
                node: Some(self.node.clone()),
                name: name.into(),
                data,
            })
            .await;
    }

    /// Pause the run and ask a human. The first time, this returns an error that
    /// you propagate with `?`; the run stops and returns the interrupt. When
    /// the run is resumed with `CompiledGraph::resume(value, ..)`, the node
    /// runs again **from the start** and this call returns `value`.
    ///
    /// Code before the interrupt therefore runs twice: keep it idempotent.
    /// Several interrupts in one node are matched by order.
    pub fn interrupt(&self, value: Value) -> Result<Value> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        match self.resume.get(n) {
            Some(v) => Ok(v.clone()),
            None => Err(Error::Other(Box::new(GraphInterrupt {
                id: format!("{}.{n}", self.interrupt_prefix),
                value,
            }))),
        }
    }
}

/// A unit of work in a graph.
#[async_trait]
pub trait Node: std::marker::Send + Sync {
    /// Run with the current state (or the [`Send`] argument).
    async fn run(&self, input: Value, ctx: NodeContext) -> Result<NodeOutput>;
}

/// Converts closures and runnables into nodes. You don't implement this.
pub trait IntoNode<Marker> {
    /// Convert.
    fn into_node(self) -> Arc<dyn Node>;
}

#[doc(hidden)]
pub struct FnMarker<O>(PhantomData<O>);
#[doc(hidden)]
pub struct RunnableMarker;

struct FnNode<F>(F);

#[async_trait]
impl<F, Fut, O> Node for FnNode<F>
where
    F: Fn(Value, NodeContext) -> Fut + std::marker::Send + Sync,
    Fut: Future<Output = Result<O>> + std::marker::Send,
    O: Into<NodeOutput>,
{
    async fn run(&self, input: Value, ctx: NodeContext) -> Result<NodeOutput> {
        (self.0)(input, ctx).await.map(Into::into)
    }
}

impl<F, Fut, O> IntoNode<FnMarker<O>> for F
where
    F: Fn(Value, NodeContext) -> Fut + std::marker::Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + std::marker::Send + 'static,
    O: Into<NodeOutput> + 'static,
{
    fn into_node(self) -> Arc<dyn Node> {
        Arc::new(FnNode(self))
    }
}

struct RunnableNode<R>(R);

#[async_trait]
impl<R: Runnable> Node for RunnableNode<R> {
    async fn run(&self, input: Value, ctx: NodeContext) -> Result<NodeOutput> {
        let out = self.0.invoke_with(input, &ctx.config).await?;
        Ok(NodeOutput::Update(runnable_output_to_update(out)?))
    }
}

/// A message output becomes `{"messages": [msg]}`; objects are used as-is.
fn runnable_output_to_update(out: Value) -> Result<Value> {
    match out {
        Value::Object(ref o) if o.contains_key("role") => Ok(json!({ "messages": [out] })),
        Value::Object(_) | Value::Null => Ok(out),
        Value::Array(ref a) if a.iter().all(|m| m.get("role").is_some()) => {
            Ok(json!({ "messages": out }))
        }
        other => Err(Error::InvalidUpdate {
            key: "*".into(),
            message: format!(
                "a runnable used as a node must return an object or messages, got {other}"
            ),
        }),
    }
}

impl<R: Runnable + 'static> IntoNode<RunnableMarker> for R {
    fn into_node(self) -> Arc<dyn Node> {
        Arc::new(RunnableNode(self))
    }
}

/// Per-node execution options.
///
/// ```
/// use flow_agent_graph::NodeOptions;
/// use flow_agent_core::RetryPolicy;
/// use std::time::Duration;
/// let opts = NodeOptions::new()
///     .timeout(Duration::from_secs(10))
///     .retry(RetryPolicy::exponential(3));
/// ```
#[derive(Clone, Debug, Default)]
pub struct NodeOptions {
    pub(crate) timeout: Option<Duration>,
    pub(crate) retry: Option<RetryPolicy>,
    pub(crate) routes_to: Vec<String>,
}

impl NodeOptions {
    /// No timeout, no retries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fail the node with [`Error::Timeout`] after `timeout` (per attempt).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Retry failures with this policy.
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// Declare the nodes this node may route to with [`Command::goto`].
    pub fn routes_to<I, S>(mut self, targets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.routes_to.extend(targets.into_iter().map(Into::into));
        self
    }
}

/// `"tools"` if the last message has tool calls, otherwise [`END`]. Use as a
/// conditional edge after a model node.
pub fn tools_condition(state: &Value) -> &'static str {
    let has_calls = state["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .is_some_and(|c| !c.is_empty());
    if has_calls { "tools" } else { END }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_from_many_shapes() {
        assert_eq!(Route::from("a").0.len(), 1);
        assert_eq!(Route::from(vec!["a", "b"]).0.len(), 2);
        assert_eq!(Route::from(vec![Send::new("w", json!(1))]).0.len(), 1);
        assert!(Route::from(None::<&str>).0.is_empty());
    }

    #[test]
    fn runnable_output_mapping() {
        assert_eq!(
            runnable_output_to_update(json!({"role": "assistant", "content": "x"})).unwrap()["messages"]
                [0]["content"],
            "x"
        );
        assert_eq!(
            runnable_output_to_update(json!({"a": 1})).unwrap(),
            json!({"a": 1})
        );
        assert!(runnable_output_to_update(json!("text")).is_err());
    }

    #[test]
    fn interrupt_then_resume_values() {
        let mut ctx = NodeContext::detached("n", RunConfig::new());
        let err = ctx.interrupt(json!("q")).unwrap_err();
        assert_eq!(as_interrupt(&err).unwrap().value, json!("q"));
        ctx.resume = Arc::new(vec![json!("yes")]);
        ctx.counter = Arc::new(AtomicUsize::new(0));
        assert_eq!(ctx.interrupt(json!("q")).unwrap(), json!("yes"));
        assert!(ctx.interrupt(json!("q2")).is_err());
    }

    #[test]
    fn tools_condition_works() {
        assert_eq!(
            tools_condition(
                &json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "1", "name": "t", "args": {}}]}]})
            ),
            "tools"
        );
        assert_eq!(tools_condition(&json!({"messages": []})), END);
    }
}
