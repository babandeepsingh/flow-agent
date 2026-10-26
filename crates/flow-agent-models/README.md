# flow-agent-models

Chat models for [flow-agent](https://github.com/babandeepsingh/flow-agent), a Rust
library for LLM chains, agents and stateful graphs inspired by LangChain and LangGraph.

- One `ChatModel` API, built from a `"provider:model"` string
- Providers behind feature flags: `openai` (plus OpenAI-compatible servers through
  `base_url`) and `anthropic`
- Token streaming and tool calls normalized across providers
- `bind_tools`, `with_structured_output`, retries, timeouts and cancellation
- `FakeChatModel` for offline, deterministic tests

**Most users should depend on [`flow-agent`](https://crates.io/crates/flow-agent) instead**,
which enables both providers by default and re-exports everything.

```toml
flow-agent-models = { version = "0.1", features = ["anthropic", "openai"] }
```

```rust
use flow_agent_core::{Runnable, json};
use flow_agent_models::ChatModel;

// API keys are passed explicitly; the library never reads environment variables.
let model = ChatModel::builder("anthropic:claude-sonnet-5-5")
    .api_key(my_key)
    .temperature(0.2)
    .build()?;

let reply = model.invoke(json!("Explain Rust lifetimes in one sentence.")).await?;
```

Local or self-hosted models through an OpenAI-compatible server:

```rust
let local = ChatModel::builder("openai:llama3.1")
    .base_url("http://localhost:11434/v1")
    .build()?;
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
