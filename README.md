# flow-agent

**LangChain + LangGraph ideas, in Rust.** Build LLM chains, tool-using agents and
stateful, resumable graphs. It's a library: you embed it in your own Axum API, CLI or
background worker.

[![CI](https://github.com/babandeepsingh/flow-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/babandeepsingh/flow-agent/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/flow-agent.svg)](https://crates.io/crates/flow-agent)
[![docs.rs](https://docs.rs/flow-agent/badge.svg)](https://docs.rs/flow-agent)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> **Status: pre-release, under construction.** The API is specified in
> [DESIGN.md](DESIGN.md) and the crates are being built one by one. The first
> crates.io release will be `0.1.0`. The examples below show the planned API.

- **Simple**: structs, methods and builder chaining, like LangChain in Python.
- **JSON in, JSON out**: `serde_json::Value` everywhere, so no generics in your code.
- **Async**: runs on tokio, streams tokens and events.
- **Ready for production**: `tracing` spans, typed errors, timeouts, retries, cancellation,
  checkpoints (memory, SQLite, Postgres), human-in-the-loop interrupts.
- **No surprises**: no global state, no panics, no hidden env-var reads.

---

## Install

```sh
cargo add flow-agent
cargo add tokio --features full
cargo add serde_json
```

Or in `Cargo.toml`:

```toml
[dependencies]
flow-agent = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

Requires Rust **1.85+**.

### Feature flags

| Feature | Default | What it adds |
|---|:---:|---|
| `anthropic` | ✅ | Claude models (`"anthropic:..."`) |
| `openai` | ✅ | OpenAI and OpenAI-compatible servers (`"openai:..."`) |
| `sqlite` | | SQLite checkpointer |
| `postgres` | | Postgres checkpointer |
| `schemars` | | Typed tool arguments with an auto-generated JSON Schema |
| `full` | | All of the above |

For the smallest build (for example, tests that only use the fake model):

```toml
flow-agent = { version = "0.1", default-features = false }
```

---

## Quick start

### 1. Call a model

```rust
use flow_agent::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> flow_agent::Result<()> {
    // The library never reads env vars. Your app passes the key in.
    let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();

    let model = ChatModel::builder("anthropic:claude-sonnet-5-5")
        .api_key(key)
        .temperature(0.2)
        .build()?;

    let reply = model.invoke(json!("Explain Rust ownership in one sentence.")).await?;
    println!("{}", reply["content"]);
    Ok(())
}
```

### 2. Chain prompt → model → parser

```rust
let prompt = ChatPromptTemplate::from_messages([
    ("system", "You are a {style} assistant."),
    ("user", "{question}"),
]);

let chain = prompt.pipe(model).pipe(StrOutputParser::new());

let answer = chain
    .invoke(json!({ "style": "concise", "question": "What is a lifetime?" }))
    .await?;
```

### 3. Agent with tools, in one call

```rust
let weather = Tool::new("get_weather", "Get the weather for a city")
    .param("city", "string", "City name")
    .handler(|args| async move {
        Ok(json!({ "city": args["city"], "forecast": "sunny", "temp_c": 21 }))
    });

let agent = create_react_agent(model, [weather])?;
let state = agent.invoke(json!("What's the weather in Paris?")).await?;
```

### 4. Build your own graph

```rust
let graph = StateGraph::new()
    .key("messages", Reducer::Messages)
    .add_node("agent", model.bind_tools(&tools))
    .add_node("tools", ToolNode::new(tools))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", |state: &Value| {
        if has_tool_calls(state) { "tools" } else { END }
    })
    .add_edge("tools", "agent");

let app = graph.compile_with(
    CompileOptions::new().checkpointer(Arc::new(MemorySaver::new())),
)?;
```

### 5. Stream tokens and events

```rust
use futures::StreamExt;

let config = RunConfig::new().thread_id("user-42");
let mut events = app.stream_with(json!("Hi!"), &config);

while let Some(event) = events.next().await {
    match event? {
        Event::Token { text, .. } => print!("{text}"),
        Event::NodeStart { node, .. } => eprintln!("\n→ {node}"),
        Event::Interrupt { value, .. } => eprintln!("needs input: {value}"),
        _ => {}
    }
}
```

Every `Event` serializes to JSON like `{"type":"token","text":"Hel",...}`, so you can
forward it straight to Server-Sent Events or a WebSocket.

### 6. Human-in-the-loop

```rust
// Inside a node:
let answer = ctx.interrupt(json!({ "question": "Send this email?" }))?;

// In your app:
let out = app.invoke_with(input, &config).await?;   // pauses; out["__interrupt__"] has the question
let out = app.resume(json!({ "approved": true }), &config).await?;   // continues
```

### Testing without network

```rust
let model: ChatModel = FakeChatModel::new([
    Message::assistant("").with_tool_call("get_weather", json!({ "city": "Paris" })),
    Message::assistant("It's sunny in Paris."),
]).into();
```

---

## Crates

Most people only need `flow-agent`. The pieces are also published separately:

| Crate | What it is |
|---|---|
| [`flow-agent`](https://crates.io/crates/flow-agent) | Umbrella crate: re-exports everything plus a `prelude` |
| [`flow-agent-core`](https://crates.io/crates/flow-agent-core) | `Runnable`, chaining, messages, prompts, parsers, tools, `RunConfig`, events, errors |
| [`flow-agent-models`](https://crates.io/crates/flow-agent-models) | `ChatModel`, OpenAI and Anthropic providers, `FakeChatModel` |
| [`flow-agent-graph`](https://crates.io/crates/flow-agent-graph) | `StateGraph`, reducers, parallel execution, streaming, interrupts |
| [`flow-agent-checkpoint`](https://crates.io/crates/flow-agent-checkpoint) | Memory, SQLite and Postgres checkpointers |
| [`flow-agent-prebuilt`](https://crates.io/crates/flow-agent-prebuilt) | Ready-made ReAct agent |

## Examples

| Example | Run |
|---|---|
| Axum HTTP API with SSE streaming | `cargo run -p axum-api` |
| Interactive CLI chat | `cargo run -p cli` |
| Background worker | `cargo run -p worker` |

They fall back to the fake model when no API key is set, so they run offline.

## Observability

flow-agent emits `tracing` spans for every graph run, node, model call and tool call
(`flow_agent.graph`, `flow_agent.node`, `flow_agent.model`, `flow_agent.tool`).
Add any subscriber, for example `tracing_subscriber::fmt::init()`, to see them.
Message contents are not recorded unless you opt in.

---

## Roadmap

- [x] Design ([DESIGN.md](DESIGN.md))
- [x] Reserve crate names on crates.io (`0.0.1` placeholders)
- [x] `flow-agent-core`
- [ ] `flow-agent-models`
- [ ] `flow-agent-graph`
- [ ] `flow-agent-checkpoint`
- [ ] `flow-agent-prebuilt`
- [ ] `flow-agent` umbrella + examples
- [ ] Publish `0.1.0` to crates.io

## Contributing

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

CI runs the same checks on every pull request. Tests never call real APIs.

## Releasing (maintainers)

All crates share one version and are published together to crates.io.

1. Bump `version` in the root `Cargo.toml` (`[workspace.package]`) and update the changelog.
2. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. The `release` workflow runs the checks and publishes the crates in dependency order:
   `core → models → graph → checkpoint → prebuilt → flow-agent`.

One-time setup: create an API token at <https://crates.io/settings/tokens>
(scope `publish-new` and `publish-update`) and save it as the repository secret
`CARGO_REGISTRY_TOKEN`. The workflow can later switch to crates.io
[Trusted Publishing](https://crates.io/docs/trusted-publishing) so no long-lived token is needed.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
