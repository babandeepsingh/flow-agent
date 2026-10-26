# flow-agent-checkpoint

Checkpointers for [flow-agent](https://github.com/babandeepsingh/flow-agent) graphs.
They save graph state after every step, keyed by thread id, enabling
multi-turn conversations, resume after restarts, human-in-the-loop interrupts and replay.

| Saver | Feature | Use for |
|---|---|---|
| `MemorySaver` | always | tests, prototypes, single process |
| `SqliteSaver` | `sqlite` | single-node services, CLIs, desktop apps |
| `PostgresSaver` | `postgres` | multi-instance production services |

**Most users should depend on [`flow-agent`](https://crates.io/crates/flow-agent) instead**
(`features = ["sqlite"]` or `["postgres"]`).

```rust
use std::sync::Arc;
use flow_agent_checkpoint::SqliteSaver;
use flow_agent_graph::CompileOptions;

let saver = SqliteSaver::connect("sqlite://agent.db").await?;
saver.migrate().await?; // creates tables; explicit, never implicit
let app = graph.compile_with(CompileOptions::new().checkpointer(Arc::new(saver)))?;
```

Writing your own backend? Run `flow_agent_checkpoint::testing::run_conformance(&saver)`
in your tests.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
