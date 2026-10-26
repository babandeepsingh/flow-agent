# flow-agent-prebuilt

Ready-made agents for [flow-agent](https://github.com/babandeepsingh/flow-agent).

`create_react_agent(model, tools)` builds a ReAct-style tool-calling agent in one call.
`ReactAgent::builder(model)` adds a system prompt, an iteration limit, sequential or
parallel tool calls, tool error handling, human approval before tools, and persistence.

**Most users should depend on [`flow-agent`](https://crates.io/crates/flow-agent) instead.**

```rust
use flow_agent_prebuilt::create_react_agent;

let agent = create_react_agent(model, [weather_tool])?;
let state = agent.invoke(json!("What's the weather in Paris?")).await?;
```

The agent is a normal `CompiledGraph`, so `stream`, checkpoints, interrupts and
`resume` all work.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
