# flow-agent — Design

> Status: **draft for review**. Nothing here is implemented yet. Once approved,
> this document is the contract each crate PR is checked against.

flow-agent is a Rust library for building LLM applications and agents,
inspired by LangChain (composable runnables) and LangGraph (stateful graphs
with checkpoints and human-in-the-loop). It is **a library only**: no server
and no CLI in the core crates. You embed it in your own Axum API, CLI or
background worker.

---

## 1. Goals and non-goals

### Goals

| Goal | What it means in practice |
|---|---|
| Quick to use | Feels like LangChain/LangGraph in Python: structs, methods, builder chaining. A ReAct agent is one call. |
| JSON in, JSON out | `serde_json::Value` flows through every component. The user-facing API has **no generics**. |
| Async on tokio | Every I/O call is `async`. Streams are `futures::Stream`. |
| No surprises | No global state, no panics in library code, **no env-var magic** (API keys are passed explicitly). |
| Production-ready | `tracing` spans everywhere, typed errors with context, timeouts, retries, cancellation, everything serializable. |
| Lean | Minimal default dependencies; providers and databases behind feature flags. |

### Non-goals (for 0.1)

- Typed/generic state structs (users can `serde_json::from_value` at the edges).
- Document loaders, vector stores, retrievers, embeddings (possible later crates).
- A hosted server, LangSmith-style tracing UI, or a CLI.
- Python interop.

---

## 2. Workspace layout

```
flow-agent/
├── Cargo.toml                 # [workspace], shared [workspace.package] metadata
├── README.md                  # used by the umbrella crate on crates.io
├── DESIGN.md
├── LICENSE-MIT, LICENSE-APACHE
├── crates/
│   ├── flow-agent-core/       # Runnable, messages, prompts, parsers, tools, config, events, errors, Checkpointer trait
│   ├── flow-agent-models/     # ChatModel, providers (openai, anthropic), FakeChatModel
│   ├── flow-agent-graph/      # StateGraph, reducers, CompiledGraph (Pregel-style executor)
│   ├── flow-agent-checkpoint/ # MemorySaver, SqliteSaver, PostgresSaver (feature)
│   ├── flow-agent-prebuilt/   # create_react_agent / ReactAgent builder
│   └── flow-agent/            # umbrella: re-exports + prelude
├── examples/
│   ├── axum-api/              # separate package (publish = false)
│   ├── cli/
│   └── worker/
└── .github/workflows/{ci.yml, release.yml}
```

### Dependency graph

```
                 flow-agent-core
            ┌────────┼──────────────┐
            ▼        ▼              ▼
   flow-agent-models  flow-agent-graph  flow-agent-checkpoint
            └────────┬──────┘
                     ▼
           flow-agent-prebuilt
                     ▼
               flow-agent (umbrella; depends on all)
```

The `Checkpointer` **trait and `Checkpoint` types live in core**; the
`flow-agent-checkpoint` crate only provides implementations. This keeps graph
independent of any storage backend and lets the PRs land in the requested
order (graph before checkpoint) without a dependency cycle.

Publish order (dependency order): `core → models → graph → checkpoint → prebuilt → flow-agent`.

---

## 3. flow-agent-core

### 3.1 `Runnable`

Everything executable is a `Runnable`: prompts, models, parsers, tools,
lambdas, chains and compiled graphs.

```rust
pub type Value = serde_json::Value;
pub type Result<T, E = Error> = std::result::Result<T, E>;
pub type EventStream = BoxStream<'static, Result<Event>>;

#[async_trait]
pub trait Runnable: Send + Sync {
    /// Human-readable name used in traces and events.
    fn name(&self) -> &str;

    /// The one required method.
    async fn invoke_with(&self, input: Value, config: &RunConfig) -> Result<Value>;

    /// Convenience: `invoke_with(input, &RunConfig::default())`.
    async fn invoke(&self, input: Value) -> Result<Value> { … }

    /// Stream events (tokens, tool calls, node events…) followed by a final
    /// `Event::Output`. Default: run `invoke_with` and emit one `Output`.
    fn stream_with(&self, input: Value, config: &RunConfig) -> EventStream { … }
    fn stream(&self, input: Value) -> EventStream { … }

    /// Run many inputs concurrently, bounded by `config.max_concurrency`.
    /// Results are returned in input order; one failure doesn't abort others.
    async fn batch_with(&self, inputs: Vec<Value>, config: &RunConfig) -> Vec<Result<Value>> { … }
    async fn batch(&self, inputs: Vec<Value>) -> Vec<Result<Value>> { … }
}

/// Cheap-to-clone, type-erased runnable.
pub type BoxRunnable = Arc<dyn Runnable>;
```

`invoke` vs `invoke_with` mirrors Python's optional `config=` argument
without generics. The trait is object safe, so `Arc<dyn Runnable>` works
everywhere (graph nodes, chain steps, tools).

### 3.2 Composition (pipe-style chaining)

`RunnableExt` is implemented for every `Runnable` and returns concrete
structs that are themselves `Runnable`:

```rust
let chain = prompt
    .pipe(model)                 // RunnableSequence
    .pipe(StrOutputParser::new());

let answer = chain.invoke(json!({ "topic": "ownership in Rust" })).await?;
```

| Combinator | Python equivalent | Behaviour |
|---|---|---|
| `a.pipe(b)` | `a \| b` | Output of `a` is input of `b`. |
| `RunnableLambda::new("name", \|v\| async move { … })` | `RunnableLambda` | Wrap an async closure. |
| `RunnableParallel::new().branch("a", r1).branch("b", r2)` | `RunnableParallel` / dict | Same input to all; output `{"a": …, "b": …}`. Runs concurrently. |
| `RunnablePassthrough::new()` / `.assign("k", r)` | `RunnablePassthrough` | Identity / add keys to an object input. |
| `r.with_retry(RetryPolicy)` | `.with_retry()` | Exponential backoff + jitter, retries only `err.is_retryable()` by default. |
| `r.with_fallbacks([r2, r3])` | `.with_fallbacks()` | Try next on error. |
| `r.with_timeout(Duration)` | — | Fails with `Error::Timeout`. |
| `r.with_config(RunConfig)` | `.with_config()` | Merge default tags/metadata. |

> Trade-off: Rust can't implement `|` for arbitrary `Arc<dyn Runnable>`
> ergonomically (orphan rules + ownership), so `.pipe()` is the primary API.
> We may add `BitOr` on the concrete combinator types later as sugar.

### 3.3 Messages

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,                      // system | user | assistant | tool
    pub content: Content,                // String or Vec<ContentPart> (text, image_url, …)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,       // assistant only
    pub tool_call_id: Option<String>,    // tool only
    pub name: Option<String>,
    pub id: Option<String>,              // used by the Messages reducer for replace-by-id
    #[serde(default)]
    pub usage: Option<Usage>,            // token usage on assistant messages
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

Message::system("You are terse.");
Message::user("Hi!");                    // alias: Message::human
Message::assistant("Hello.");            // alias: Message::ai
Message::tool("call_123", json!({ "temp": 21 }));

let msgs: Vec<Message> = Message::from_value(value)?;   // accepts many shapes
let text: Option<&str> = message.text();
```

Input coercion (`Message::from_value`) accepts: a plain string (→ user
message), a single message object, an array of message objects, an array of
`["role", "content"]` tuples, or an object with a `"messages"` key. This makes
`model.invoke(json!("hi"))` and `agent.invoke(json!({"messages": […]}))` both
work.

### 3.4 Prompt templates

```rust
// String template, f-string syntax; `{{` / `}}` escape braces.
let p = PromptTemplate::new("Tell me a joke about {topic}");
p.invoke(json!({ "topic": "crabs" })).await?;          // -> "Tell me a joke about crabs"

// Chat template -> array of messages.
let chat = ChatPromptTemplate::from_messages([
        ("system", "You are a {style} assistant."),
        ("user", "{question}"),
    ])
    .placeholder("history");                           // inserts a message array from input["history"]

let messages = chat.invoke(json!({
    "style": "helpful", "question": "What is a lifetime?", "history": []
})).await?;
```

Variables are extracted at construction; missing variables fail with
`Error::Template { missing: [...] }`. `.partial("style", "terse")`
pre-fills variables.

### 3.5 Output parsers

All parsers are `Runnable`s and accept a message, string or message chunk.

| Parser | Output |
|---|---|
| `StrOutputParser` | Message text as a JSON string. |
| `JsonOutputParser` | Parsed JSON; tolerates ```` ```json ```` fences and leading prose. |
| `JsonOutputParser::with_schema(schema)` | Also validates required keys / types (lightweight, no full JSON-Schema engine by default). |
| `ToolCallsParser` | `[{ "name", "args", "id" }]` from an assistant message. |

### 3.6 Tools

```rust
let weather = Tool::new("get_weather", "Get the current weather for a city")
    .param("city", "string", "City name, e.g. 'Paris'")      // required by default
    .optional_param("unit", "string", "celsius | fahrenheit")
    .handler(|args: Value| async move {
        let city = args["city"].as_str().unwrap_or_default();
        Ok(json!({ "city": city, "forecast": "sunny", "temp_c": 21 }))
    });

// Or give a full JSON Schema:
let search = Tool::new("search", "Web search")
    .input_schema(json!({ "type": "object", "properties": { "q": { "type": "string" } }, "required": ["q"] }))
    .handler(search_fn);

// With the `schemars` feature: typed args, schema derived automatically.
let add = Tool::typed("add", "Add two numbers", |a: AddArgs| async move { Ok(a.x + a.y) });

let tools = ToolRegistry::new().with(weather).with(search);
let result: ToolResult = tools.call(&tool_call, &config).await;   // never panics; errors become ToolResult::Error
```

- `Tool` is a `Runnable` (input = args object, output = result).
- `ToolRegistry` holds tools by name, exposes `definitions()` (name,
  description, JSON Schema) for models, and executes many calls concurrently
  with `call_many`.
- Arguments are validated against `required` keys before the handler runs.
- `ToolResult` → `Message::tool(...)` conversion is provided; tool errors can be
  returned to the model as text (default in the prebuilt agent) or propagated.

### 3.7 `RunConfig`

Passed by reference through every call. Cheap to clone (internals are `Arc`).

```rust
let config = RunConfig::new()
    .thread_id("user-42")                 // required for checkpointing
    .run_name("support-bot")
    .tag("prod")
    .metadata("tenant", json!("acme"))
    .callback(MyHandler::new())           // Arc<dyn CallbackHandler>
    .max_concurrency(8)                   // for batch / parallel nodes
    .recursion_limit(50)                  // graph supersteps
    .timeout(Duration::from_secs(60))     // whole run
    .cancellation(token.clone())          // tokio_util::sync::CancellationToken
    .configurable("model", json!("anthropic:claude-sonnet-5-5")); // free-form runtime options
```

| Field | Serialized? | Notes |
|---|---|---|
| `thread_id`, `checkpoint_id`, `run_id`, `run_name`, `tags`, `metadata`, `configurable`, `recursion_limit`, `max_concurrency` | yes | `run_id` is a UUID v7, generated if absent. |
| `callbacks`, `cancellation`, internal event sink | no (`#[serde(skip)]`) | Runtime-only handles. |

Child runs inherit the parent config with a new `run_id` and `parent_run_id`.

### 3.8 Events and callbacks

One serializable event enum is shared by models, tools, chains and graphs, so
the same stream can be forwarded as SSE/WebSocket JSON with no mapping.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Event {
    Token        { run_id: Uuid, node: Option<String>, text: String },
    ToolCallChunk{ run_id: Uuid, node: Option<String>, index: usize, id: Option<String>, name: Option<String>, args_delta: String },
    ModelStart   { run_id: Uuid, node: Option<String>, model: String },
    ModelEnd     { run_id: Uuid, node: Option<String>, message: Message },
    ToolStart    { run_id: Uuid, node: Option<String>, tool: String, call_id: String, args: Value },
    ToolEnd      { run_id: Uuid, node: Option<String>, tool: String, call_id: String, output: Value, is_error: bool },
    NodeStart    { step: u32, node: String, task_id: Uuid },
    NodeEnd      { step: u32, node: String, task_id: Uuid, update: Value, duration_ms: u64 },
    StateUpdate  { step: u32, nodes: Vec<String>, values: Value },
    Interrupt    { step: u32, node: String, interrupt_id: String, value: Value },
    Checkpoint   { checkpoint_id: String, step: u32 },
    Custom       { node: Option<String>, name: String, data: Value },
    Output       { value: Value },
}
// JSON: {"type":"token","run_id":"…","node":"agent","text":"Hel"}
```

```rust
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    async fn on_event(&self, event: &Event, config: &RunConfig);
    async fn on_error(&self, error: &Error, config: &RunConfig) {}
}
```

How tokens reach a graph stream: `RunConfig` carries an internal event sink.
When a model is invoked inside a node that is being streamed, it emits
`Token` events into that sink even though the node itself called
`invoke`. This is how LangGraph's `stream_mode="messages"` works and means
node code never has to deal with streaming explicitly.

### 3.9 Errors

One `#[non_exhaustive]` enum (via `thiserror`), with context chaining:

```rust
pub enum Error {
    Model     { provider: String, message: String, status: Option<u16>, retryable: bool },
    Tool      { tool: String, message: String },
    ToolNotFound(String),
    InvalidToolArgs { tool: String, message: String },
    Template  { message: String, missing: Vec<String> },
    Parse     { message: String, input: String },
    InvalidInput(String),
    InvalidGraph(Vec<String>),          // all compile-time validation problems at once
    InvalidUpdate { key: String, message: String },
    NodeFailed { node: String, step: u32, source: Box<Error> },
    RecursionLimit { limit: u32 },
    Timeout   { what: String, after: Duration },
    Cancelled,
    Checkpoint(String),
    Config(String),
    Serde(serde_json::Error),
    Http(String),
    Context   { context: String, source: Box<Error> },
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn is_retryable(&self) -> bool;         // 429, 5xx, timeouts, network errors
    pub fn context(self, ctx: impl Into<String>) -> Self;
    pub fn to_value(&self) -> Value;            // JSON for APIs / events
}

pub trait ResultExt<T> { fn context(self, ctx: impl Into<String>) -> Result<T>; }
```

Interrupts are **not** errors at the public API: a graph run that pauses
returns successfully with the interrupt in its output (see §5.6).

### 3.10 Checkpointer trait (implemented in `flow-agent-checkpoint`)

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub v: u32,                            // format version (1)
    pub id: String,                        // UUID v7: sortable by time
    pub thread_id: String,
    pub namespace: String,                 // "" for root graph, "parent:child" for subgraphs
    pub parent_id: Option<String>,
    pub step: u32,
    pub state: Value,                      // full state after the superstep
    pub next: Vec<String>,                 // nodes scheduled for the next superstep
    pub pending_writes: Vec<PendingWrite>, // writes from tasks that finished in an interrupted step
    pub interrupts: Vec<Interrupt>,
    pub metadata: Map<String, Value>,      // source ("input" | "loop" | "update" | "fork"), writes, tags…
    pub created_at: DateTime<Utc>,
}

#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn put(&self, checkpoint: &Checkpoint) -> Result<()>;
    async fn put_writes(&self, thread_id: &str, checkpoint_id: &str, writes: &[PendingWrite]) -> Result<()>;
    /// Latest checkpoint when `checkpoint_id` is None.
    async fn get(&self, thread_id: &str, namespace: &str, checkpoint_id: Option<&str>) -> Result<Option<Checkpoint>>;
    /// Newest first.
    async fn list(&self, thread_id: &str, filter: ListFilter) -> Result<Vec<Checkpoint>>;
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}
```

### 3.11 Core dependencies

`serde`, `serde_json`, `thiserror`, `async-trait`, `futures`, `tokio`
(`rt`, `time`, `sync`, `macros`), `tokio-util` (cancellation), `tracing`,
`uuid` (v7, serde), `chrono` (serde). Optional: `schemars` (feature
`schemars`).

---

## 4. flow-agent-models

### 4.1 `ChatModel`

A concrete, cheap-to-clone struct (wraps `Arc<dyn ChatProvider>` + settings)
that implements `Runnable`. Input: anything `Message::from_value` accepts.
Output: the assistant `Message` as JSON.

```rust
// "provider:model" spec. API keys are explicit: no env-var lookups in the library.
let model = ChatModel::builder("anthropic:claude-sonnet-5-5")
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)   // the *application* reads env, not the library
    .temperature(0.2)
    .max_tokens(1024)
    .timeout(Duration::from_secs(60))
    .max_retries(2)
    .build()?;

let reply = model.invoke(json!("Explain Rc vs Arc in one sentence")).await?;

// OpenAI-compatible servers (vLLM, Ollama, LM Studio, Azure-style gateways):
let local = ChatModel::builder("openai:llama3.1")
    .base_url("http://localhost:11434/v1")
    .build()?;

// Bind tools -> returns a new ChatModel; the original is unchanged.
let with_tools = model.bind_tools(&tools);

// Structured output: forces a tool call / JSON schema and returns the parsed object.
let extractor = model.with_structured_output(json!({
    "type": "object",
    "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
    "required": ["name", "age"]
}));
```

- `ChatModel::from_spec(spec, ProviderConfig)` is the non-builder form.
- Unknown provider prefix or a provider whose feature is disabled →
  `Error::Config("provider 'openai' requires the `openai` feature")`.
- `ModelRegistry` lets users register custom providers
  (`registry.register("ollama", factory)`) — an explicit value you own,
  **not** a global.

### 4.2 Provider trait and normalization

```rust
#[async_trait]
pub trait ChatProvider: Send + Sync {
    fn provider(&self) -> &str;          // "anthropic"
    fn model(&self) -> &str;             // "claude-sonnet-5-5"
    async fn generate(&self, req: ChatRequest) -> Result<Message>;
    fn stream(&self, req: ChatRequest) -> BoxStream<'static, Result<ChatChunk>>;
}

pub struct ChatRequest { pub messages: Vec<Message>, pub tools: Vec<ToolDefinition>, pub tool_choice: ToolChoice, pub options: ModelOptions }
pub struct ChatChunk   { pub text: Option<String>, pub tool_call_chunks: Vec<ToolCallChunk>, pub usage: Option<Usage>, pub finish_reason: Option<FinishReason> }
```

Every provider maps its wire format to the same `Message`, `ToolCall`
(`{ id, name, args: Value }`), `Usage` and `FinishReason`. Streaming
tool-call argument fragments are accumulated by a shared `ChunkAggregator`
so `stream` and `invoke` produce identical final messages. Anthropic's
separate `system` field and OpenAI's `tool` role are handled inside the
providers.

### 4.3 Features

| Feature | Default | Pulls in |
|---|---|---|
| `openai` | off | `reqwest` (rustls, json, stream); SSE parsing is built in |
| `anthropic` | off | same as above |

No provider is enabled in `flow-agent-models` by default; the umbrella crate
enables both (see §8). TLS is `rustls` to avoid a system OpenSSL dependency.

### 4.4 `FakeChatModel` (always available, no network)

```rust
let fake = FakeChatModel::new([
    Message::assistant("").with_tool_call("get_weather", json!({ "city": "Paris" })),
    Message::assistant("It's sunny in Paris."),
]);
// Responses are returned in order (error once exhausted, or `.cycle()`).
// stream() splits text into word tokens. `fake.calls()` returns recorded requests for assertions.
let model: ChatModel = fake.clone().into();
```

Also `FakeChatModel::echo()` and `FakeChatModel::failing(err)` / `.fail_times(n)` for retry tests.

---

## 5. flow-agent-graph

### 5.1 State and reducers

State is a JSON object. Each key has a reducer that decides how a node's
partial update is merged. Keys without a declared reducer use `Overwrite`.

```rust
let graph = StateGraph::new()
    .key("messages", Reducer::Messages)  // append; replace when ids match (like add_messages)
    .key("docs",     Reducer::Append)    // array concat (scalars are pushed)
    .key("meta",     Reducer::Merge)     // shallow object merge
    .key("count",    Reducer::custom(|old, new| Ok(json!(old.as_i64().unwrap_or(0) + new.as_i64().unwrap_or(0)))));
```

| Reducer | Semantics |
|---|---|
| `Overwrite` (default) | New value replaces old. Two nodes writing the same key in the same superstep → `Error::InvalidUpdate`. |
| `Append` | Arrays concatenate; non-arrays are appended as one element. |
| `Merge` | Shallow object merge (`DeepMerge` variant for recursive). |
| `Messages` | Coerces to messages, appends, replaces by `id`, supports `{"remove": id}`. |
| `custom(f)` | `Fn(Value, Value) -> Result<Value>`. |

### 5.2 Nodes

A node is any `Runnable`, or a closure receiving the current state and a
`NodeContext`. A node returns a **partial update** (`Value` object) or a
`Command`.

```rust
.add_node("greet", |state: Value, _ctx: NodeContext| async move {
    Ok(json!({ "messages": [Message::assistant("Hello!")] }))
})
.add_node("agent", model.clone())                      // any Runnable works as a node
.add_node("tools", ToolNode::new(tools.clone()))       // executes tool calls from the last message
.add_node_with("search", search_fn, NodeOptions::new()
    .timeout(Duration::from_secs(10))
    .retry(RetryPolicy::exponential(3).initial(Duration::from_millis(200)).max(Duration::from_secs(5))))
```

When a `Runnable` (not a closure) is used as a node, its output is applied
as-is if it's an object of state keys; an assistant message output is wrapped
as `{"messages": [msg]}`.

`NodeContext` gives node code access to the run:

```rust
ctx.config()                     // &RunConfig (thread_id, metadata…)
ctx.step(); ctx.node();
ctx.emit("progress", json!({ "pct": 40 }));     // Event::Custom
ctx.interrupt(json!({ "question": "Approve?" }))?   // see §5.6
ctx.is_cancelled()
```

`Command` lets a node update state **and** route in one return:

```rust
Ok(Command::goto("human_review").update(json!({ "draft": draft })).into())
```

### 5.3 Edges

```rust
use flow_agent_graph::{START, END};

.add_edge(START, "agent")
.add_conditional_edges("agent", |state: &Value| {
    if last_message_has_tool_calls(state) { "tools" } else { END }
})
.add_edge("tools", "agent")

// With an explicit path map (lets compile() validate targets and draw diagrams):
.add_conditional_edges_with("classify", route_fn, [("billing", "billing_agent"), ("tech", "tech_agent")])

// Fan-out: a router may return several targets (Vec<String>) → they run in parallel.
// Fan-in: add_edge_from(["a", "b"], "join") waits for both a and b.
// Map-reduce: return `Send::new("worker", json!({ "item": x }))` items to run one node many times with custom input.
```

### 5.4 Compilation and validation

```rust
let app: CompiledGraph = graph.compile()?;            // validation only
let app = graph.compile_with(
    CompileOptions::new()
        .checkpointer(Arc::new(MemorySaver::new()))
        .interrupt_before(["tools"])
        .interrupt_after(["draft"]),
)?;
```

`compile()` collects **all** problems and returns `Error::InvalidGraph(Vec<String>)`:

- no edge from `START`; edges into `START` or out of `END`;
- edge/path-map/interrupt targets that don't exist; duplicate node names; reserved names (`__start__`, `__end__`, names containing `:`);
- nodes unreachable from `START`;
- nodes with no outgoing edge and no way to reach `END` (dead ends);
- `Overwrite` vs `Append` conflicts declared twice for one key.

Conditional edges without a path map are validated at runtime (unknown target
→ `Error::InvalidGraph`). `app.to_mermaid()` renders the graph.

### 5.5 Execution model (Pregel / BSP supersteps)

1. Input is applied to state via reducers (as if written by `START`).
2. Each **superstep** runs all scheduled nodes **concurrently** on a
   `JoinSet`, bounded by `max_concurrency`. Every node sees the same
   state snapshot from the start of the step.
3. When all finish, updates are applied in deterministic order (node name
   order), so results don't depend on scheduling.
4. Edges are evaluated against the new state to schedule the next step.
5. A checkpoint is saved after every step (when a checkpointer is set).
6. Stop when no nodes are scheduled (all paths reached `END`).

Guards: `recursion_limit` (default 25 supersteps) → `Error::RecursionLimit`;
per-node `timeout`; per-node `RetryPolicy` (exponential backoff with jitter,
retries `is_retryable()` errors by default, customizable predicate);
`RunConfig::cancellation` is checked between and during steps (in-flight
nodes are aborted, state stays at the last checkpoint) → `Error::Cancelled`.
If one node in a step fails after retries, the others in that step are
cancelled, successful writes are stored as `pending_writes`, and the error is
returned with node and step context.

### 5.6 Invoke, stream, resume, state

`CompiledGraph` is cheap to clone (`Arc` inside), `Send + Sync + 'static`,
and implements `Runnable`, so a compiled graph can be a node in another graph
(subgraph).

```rust
let config = RunConfig::new().thread_id("t-1");

// invoke → final state
let state = app.invoke_with(json!({ "messages": [{ "role": "user", "content": "hi" }] }), &config).await?;

// stream → typed events (serializable, `"type"`-tagged)
let mut events = app.stream_with(input, &config);
while let Some(event) = events.next().await {
    match event? {
        Event::Token { text, .. } => print!("{text}"),
        Event::NodeStart { node, .. } => eprintln!("→ {node}"),
        Event::StateUpdate { values, .. } => { /* … */ }
        Event::Interrupt { value, .. } => { /* ask the human */ }
        _ => {}
    }
}
// Filter: app.stream_with_options(input, &config, StreamOptions::only([EventKind::Token, EventKind::Interrupt]))
```

**Human-in-the-loop.** Inside a node, `ctx.interrupt(value)?` pauses the run
(the `?` propagates an internal signal). The run returns normally:

```rust
let out = app.invoke_with(input, &config).await?;
// out["__interrupt__"] == [{ "id": "…", "node": "review", "value": { "question": "Approve?" } }]
// Typed alternative: app.run(input, &config).await? -> RunOutput { state, interrupts, checkpoint_id }

let out = app.resume(json!({ "approved": true }), &config).await?;
```

On resume the interrupted node **re-runs from its start** and
`ctx.interrupt(..)` now returns the resume value (same semantics as
LangGraph; documented: code before an interrupt must be idempotent).
Multiple interrupts in one node are matched by order. `interrupt_before` /
`interrupt_after` give static breakpoints; `resume(Value::Null, ..)` continues.

**State inspection and time travel** (require a checkpointer):

```rust
let snap: StateSnapshot = app.get_state(&config).await?;      // values, next, interrupts, checkpoint_id, metadata
let history = app.get_state_history(&config).await?;          // newest first
app.update_state(&config, json!({ "approved": false }), Some("review")).await?; // edit as if written by a node

// Replay / fork from an earlier checkpoint:
let past = RunConfig::new().thread_id("t-1").checkpoint_id(&history[3].checkpoint_id);
app.invoke_with(Value::Null, &past).await?;   // re-executes from that point, creating a new branch
```

Running without a checkpointer is fully supported; only resume/state APIs
return `Error::Config("checkpointer required")`.

### 5.7 Tracing

Spans (all at `INFO` with structured fields, children nested correctly across
`tokio` tasks via `Instrument`):

| Span | Fields |
|---|---|
| `flow_agent.graph` | `graph`, `thread_id`, `run_id` |
| `flow_agent.node` | `node`, `step`, `task_id`, `attempt` |
| `flow_agent.model` | `provider`, `model`, `stream`, `input_tokens`, `output_tokens`, `finish_reason` |
| `flow_agent.tool` | `tool`, `call_id`, `is_error` |

The library never installs a subscriber; that's the application's job.
Message contents are **not** recorded on spans by default (PII); opt in with
`RunConfig::trace_content(true)`.

---

## 6. flow-agent-checkpoint

| Saver | Feature | Storage |
|---|---|---|
| `MemorySaver` | always | `tokio::sync::RwLock<HashMap<…>>`, for tests and single-process apps |
| `SqliteSaver` | `sqlite` | `sqlx` (sqlite, runtime-tokio, no macros), WAL mode |
| `PostgresSaver` | `postgres` | `sqlx` (postgres), `JSONB` state |

```rust
let saver = SqliteSaver::connect("sqlite://agent.db?mode=rwc").await?;
saver.migrate().await?;                          // explicit; never implicit on connect
let app = graph.compile_with(CompileOptions::new().checkpointer(Arc::new(saver)))?;
```

- Schema: `checkpoints(thread_id, namespace, id, parent_id, step, state JSON, next, interrupts, metadata, created_at)` + `checkpoint_writes(...)`, primary key `(thread_id, namespace, id)`. Migrations embedded and versioned.
- All savers pass one shared **conformance test suite** (`flow_agent_checkpoint::testing::run_conformance(saver)`) so custom backends (Redis, DynamoDB…) can reuse it.
- Integration tests in this crate exercise graph resume, interrupts and replay end-to-end (`flow-agent-graph` is a dev-dependency here, so no cycle).

> Trade-off: `sqlx` over `rusqlite` + `tokio-postgres` — one async API and one
> migration story for both databases. We disable sqlx default features and
> its compile-time macros to keep build times down.

---

## 7. flow-agent-prebuilt

```rust
// One call:
let agent = create_react_agent(model, [weather, search])?;
let out = agent.invoke(json!("What's the weather in Paris?")).await?;

// Builder for more control:
let agent = ReactAgent::builder(model)
    .tools([weather, search])
    .system_prompt("You are a concise travel assistant.")
    .max_iterations(10)                  // maps to recursion_limit
    .parallel_tool_calls(true)
    .tool_errors(ToolErrorMode::ReturnToModel)   // or Propagate
    .interrupt_before_tools(true)        // human approval of every tool call
    .checkpointer(Arc::new(MemorySaver::new()))
    .build()?;                           // -> CompiledGraph
```

Graph: `START → agent → (tool_calls? tools : END)`, `tools → agent`. State
keys: `messages` (Messages reducer) plus anything the user adds via
`.state_key(..)`. Because the result is a plain `CompiledGraph`, streaming,
interrupts, resume and checkpointing all work unchanged. `ToolNode` is
exported for use in custom graphs.

---

## 8. flow-agent (umbrella)

```toml
[dependencies]
flow-agent = "0.1"
```

```rust
use flow_agent::prelude::*;   // Runnable, RunnableExt, RunConfig, Message, PromptTemplate,
                              // ChatPromptTemplate, StrOutputParser, JsonOutputParser, Tool,
                              // ToolRegistry, ChatModel, FakeChatModel, StateGraph, Reducer,
                              // START, END, Command, MemorySaver, create_react_agent, Event, Error, Result
```

Modules mirror crates: `flow_agent::{core, models, graph, checkpoint, prebuilt}`.

| Umbrella feature | Default | Enables |
|---|---|---|
| `anthropic` | **on** | `flow-agent-models/anthropic` |
| `openai` | **on** | `flow-agent-models/openai` |
| `sqlite` | off | `flow-agent-checkpoint/sqlite` |
| `postgres` | off | `flow-agent-checkpoint/postgres` |
| `schemars` | off | typed tools |
| `full` | off | everything |

> Trade-off: providers on by default in the umbrella so `cargo add flow-agent`
> "just works"; `default-features = false` gives a provider-free build
> (e.g. fake-model-only tests, or custom providers). The individual crates
> keep everything off by default.

---

## 9. Serialization guarantees

| Type | Format |
|---|---|
| State | Plain JSON object. |
| `Message`, `ToolCall`, `Usage` | Stable JSON (OpenAI-like role names). |
| `Event` | `{"type": "...", ...}` internally tagged — SSE/WebSocket ready. |
| `Checkpoint` | Versioned (`"v": 1`) JSON; DB backends store the same JSON. |
| `Error` | `Error::to_value()` → `{"kind": "...", "message": "...", ...}`. |
| `RunConfig` | Serializable minus runtime handles. |

Round-trip tests (`to_value` → `from_value`) exist for each.

---

## 10. Production practices

- **No panics**: `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` in library crates (allowed in tests/examples).
- `#![deny(missing_docs)]` and `#![forbid(unsafe_code)]` in every crate.
- MSRV **1.85** (edition 2024), checked in CI.
- `#[non_exhaustive]` on public enums that may grow (`Event`, `Error`, `Reducer`, …).
- SemVer: all crates share one version (`[workspace.package] version`) and are released together.

---

## 11. Testing

- Unit tests next to code; integration tests per crate under `tests/`.
- All tests use `FakeChatModel` — **CI never hits the network**.
- Provider wire-format tests use recorded JSON fixtures (request building + response/stream parsing); a few `#[ignore]` live tests run manually with real keys.
- Graph tests: parallel supersteps, reducer conflicts, timeouts (`tokio::time::pause`), retries, recursion limit, cancellation, compile-time validation, interrupts/resume.
- Doc examples compile (`cargo test --doc`).

## 12. Examples (`examples/`, `publish = false`)

| Example | Shows |
|---|---|
| `axum-api` | `POST /chat` (invoke), `POST /chat/stream` (SSE of `Event`s), `POST /threads/{id}/resume`, SQLite checkpoints, per-request `thread_id`. |
| `cli` | Interactive REPL with token streaming and tool-approval prompts. |
| `worker` | Tokio worker consuming jobs from a channel, bounded concurrency, graceful shutdown via `CancellationToken`, tracing to stdout. |

Each runs with the fake model when no API key is set, so they work offline.

---

## 13. CI and release

**`ci.yml`** (push + PR): `cargo fmt --check`; `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo test` with `--no-default-features`; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`; MSRV build; `cargo package --workspace` dry-run to catch packaging errors early. Uses `Swatinem/rust-cache`.

**`release.yml`** (on tag `v*`):

1. Verify the tag matches `workspace.package.version`.
2. Run the CI checks.
3. Publish in dependency order with `cargo publish -p <crate>` for
   `core → models → graph → checkpoint → prebuilt → flow-agent`, skipping
   crates whose version is already on crates.io (makes a failed release safely re-runnable).
4. Auth via `CARGO_REGISTRY_TOKEN` secret today. Switching to **crates.io
   Trusted Publishing** later = add `permissions: id-token: write`, add a
   `rust-lang/crates-io-auth-action@v1` step, and use its token output
   instead of the secret. The workflow isolates auth in one step to make
   this a ~5-line change.

### crates.io metadata (shared via `[workspace.package]`)

```toml
[workspace.package]
version      = "0.1.0"
edition      = "2024"
rust-version = "1.85"
license      = "MIT OR Apache-2.0"
repository   = "https://github.com/babandeepsingh/flow-agent"
homepage     = "https://github.com/babandeepsingh/flow-agent"
authors      = ["Babandeep Singh"]
keywords     = ["llm", "agent", "langchain", "langgraph", "ai"]   # max 5
categories   = ["asynchronous", "api-bindings", "science", "development-tools"]
```

Each crate sets its own `description` and `readme` (umbrella uses the root
`README.md`; sub-crates get a short `README.md` pointing to it), and internal
dependencies specify both `path` and `version` so they publish correctly.
Crate names `flow-agent` and `flow-agent-{core,models,graph,checkpoint,prebuilt}`
were checked and are **available** on crates.io as of 2026-10-26.

---

## 14. Delivery plan

| PR | Crate | Contents |
|---|---|---|
| 0 | — | This `DESIGN.md` + `README.md` (review only) |
| 1 | `flow-agent-core` | Workspace skeleton, licenses, CI workflow, name-reservation workflow, everything in §3 |
| 2 | `flow-agent-models` | §4, FakeChatModel, OpenAI + Anthropic behind features |
| 3 | `flow-agent-graph` | §5 |
| 4 | `flow-agent-checkpoint` | §6, conformance suite, resume/replay integration tests |
| 5 | `flow-agent-prebuilt` | §7 |
| 6 | `flow-agent` | Umbrella, prelude, examples, release workflow |

---

## 15. Decisions

Confirmed by the project owner:

| Question | Decision |
|---|---|
| Providers working out of the box | **Claude and OpenAI** both on by default in the umbrella crate; sub-crates keep everything off. |
| A node with no outgoing edge | **Compile error** (stricter than LangGraph); write `add_edge(x, END)` explicitly. |
| When to go live on crates.io | **Reserve the names now** with `0.0.1` placeholders, real `0.1.0` when all crates are done. |
| Review flow | **One PR per crate**; the next starts only after the previous is approved. |

Technical defaults chosen by the implementer (change on request):

1. `invoke(input)` + `invoke_with(input, &config)` instead of one method that always takes a config.
2. `async-trait` for object-safe async traits (native `async fn` in traits isn't `dyn`-compatible yet).
3. `Checkpointer` trait in core, implementations in `flow-agent-checkpoint`.
4. Interrupted runs return state with an `__interrupt__` key (LangGraph-compatible) plus a typed `run()`.
5. `sqlx` (no macros) for SQLite and Postgres.
6. Crates live under `crates/` with a root virtual manifest.

### Reserving crate names

PR 1 adds a manually triggered `reserve-names.yml` workflow that publishes a
minimal `0.0.1` of each of the six crates (real description, links to this
repository, no code) using the `CARGO_REGISTRY_TOKEN` secret. The owner runs it
once from the GitHub *Actions* tab after adding the secret. crates.io forbids
mass name-squatting, but reserving the names of a single, actively developed
project is normal practice; the placeholders are superseded by `0.1.0`.
