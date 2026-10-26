# flow-agent-graph

LangGraph-style stateful graphs for [flow-agent](https://github.com/babandeepsingh/flow-agent).

- `StateGraph` builder: `add_node`, `add_edge`, `add_conditional_edges`, `START` / `END`
- JSON state with per-key reducers: overwrite, append, merge, messages, custom
- Parallel supersteps, fan-out / fan-in, map-reduce with `Send`, `Command` routing
- Per-node timeouts and retries, recursion limit, cancellation
- Graph validated at compile time (all problems reported at once)
- `invoke`, typed event `stream`, human-in-the-loop interrupts, `resume`,
  `get_state`, `update_state`, history and replay (with a checkpointer)

**Most users should depend on [`flow-agent`](https://crates.io/crates/flow-agent) instead.**

```rust
use flow_agent_graph::{StateGraph, Reducer, ToolNode, START, tools_condition};

let app = StateGraph::new()
    .key("messages", Reducer::Messages)
    .add_node("agent", model.bind_tools(&tools))
    .add_node("tools", ToolNode::new(tools))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", tools_condition)
    .add_edge("tools", "agent")
    .compile()?;

let state = app.invoke(json!({ "messages": ["What's the weather in Paris?"] })).await?;
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
