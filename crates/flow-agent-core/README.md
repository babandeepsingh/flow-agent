# flow-agent-core

Core building blocks for [flow-agent](https://github.com/babandeepsingh/flow-agent),
a Rust library for LLM chains, agents and stateful graphs inspired by LangChain and LangGraph.

This crate provides:

- `Runnable`: the JSON-in, JSON-out trait with `invoke`, `stream` and `batch`
- Chaining with `.pipe()`, plus `RunnableParallel`, `RunnablePassthrough`, retries, fallbacks and timeouts
- `Message`, `PromptTemplate`, `ChatPromptTemplate`
- Output parsers: `StrOutputParser`, `JsonOutputParser`, `ToolCallsParser`
- `Tool` and `ToolRegistry`
- `RunConfig` (thread id, tags, metadata, callbacks, cancellation), `Event`, and the `Error` type
- The `Checkpointer` trait (implementations live in `flow-agent-checkpoint`)

**Most users should depend on [`flow-agent`](https://crates.io/crates/flow-agent) instead**,
which re-exports this crate together with models, graphs and checkpointers.

```rust
use flow_agent_core::prelude::*;

let chain = PromptTemplate::new("Hello, {name}!")
    .pipe(RunnableLambda::new("shout", |v| async move {
        Ok(json!(v.as_str().unwrap_or_default().to_uppercase()))
    }));

let out = chain.invoke(json!({ "name": "crab" })).await?; // "HELLO, CRAB!"
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
