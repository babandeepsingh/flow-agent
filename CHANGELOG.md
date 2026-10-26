# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/). All crates share one version.

## [Unreleased]

### Added

- `flow-agent-core`: `Runnable` (invoke, stream, batch), pipe chaining, messages,
  prompt templates, output parsers, tools and registry, `RunConfig`, events, errors,
  retry policy and the `Checkpointer` trait.
- `flow-agent-models`: `ChatModel` from `"provider:model"`, OpenAI and Anthropic
  providers, streaming and tool calls, structured output, `FakeChatModel`.
- `flow-agent-graph`: `StateGraph`, reducers, parallel supersteps, `Send` and
  `Command`, compile-time validation, interrupts, resume, state history and replay.
- `flow-agent-checkpoint`: memory, SQLite and Postgres checkpointers and a
  conformance suite.
- `flow-agent-prebuilt`: `create_react_agent` and `ReactAgent::builder`.
- `flow-agent`: umbrella crate with a prelude; Axum, CLI and worker examples.
