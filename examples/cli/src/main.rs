//! Interactive chat in the terminal: tokens stream as they arrive, and every
//! tool call needs your approval.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p cli   # or OPENAI_API_KEY, or nothing for a fake model
//! ```

use std::io::Write;

use flow_agent::prelude::*;
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader, Lines, Stdin};

fn model() -> Result<ChatModel> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        return ChatModel::builder("anthropic:claude-sonnet-5-5")
            .api_key(key)
            .build();
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        return ChatModel::builder("openai:gpt-4.1-mini")
            .api_key(key)
            .build();
    }
    println!("(no API key set: using a fake model)");
    Ok(FakeChatModel::new([
        Message::assistant("").with_tool_call("calculator", json!({ "a": 6, "b": 7 })),
        Message::assistant("6 times 7 is 42."),
    ])
    .cycle()
    .into())
}

fn calculator() -> Tool {
    Tool::new("calculator", "Multiply two numbers")
        .param("a", "number", "First number")
        .param("b", "number", "Second number")
        .handler(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a * b))
        })
}

/// Stream one run, printing tokens. Returns the interrupts, if any.
async fn stream_run(agent: &CompiledGraph, input: Value, config: &RunConfig) -> Result<Vec<Value>> {
    let mut interrupts = Vec::new();
    let mut events = agent.stream_with(input, config);
    while let Some(event) = events.next().await {
        match event? {
            Event::Token { text, .. } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            Event::ToolEnd { tool, output, .. } => println!("  [{tool} -> {output}]"),
            Event::Interrupt { value, .. } => interrupts.push(value),
            Event::Output { .. } => println!(),
            _ => {}
        }
    }
    Ok(interrupts)
}

async fn ask(lines: &mut Lines<BufReader<Stdin>>, prompt: &str) -> Option<String> {
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    lines.next_line().await.ok().flatten()
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = ReactAgent::builder(model()?)
        .tool(calculator())
        .interrupt_before_tools(true)
        .checkpointer(Arc::new(MemorySaver::new()))
        .build()?;
    let config = RunConfig::new().thread_id("cli");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    println!("Chat with the agent. Try \"what is 6 times 7?\". Ctrl-D to quit.");

    while let Some(line) = ask(&mut lines, "\nyou> ").await {
        if line.trim().is_empty() {
            continue;
        }
        print!("bot> ");
        let mut interrupts = stream_run(&agent, json!(line), &config).await?;

        // Approve or deny pending tool calls until the turn completes.
        while !interrupts.is_empty() {
            let state = agent.get_state(&config).await?;
            let calls = state
                .as_ref()
                .and_then(|s| flow_agent::core::message::last_message(&s.values))
                .map(|m| m.tool_calls)
                .unwrap_or_default();
            for c in &calls {
                println!("\n  tool request: {}({})", c.name, c.args);
            }
            let answer = ask(&mut lines, "  approve? [y/N] ")
                .await
                .unwrap_or_default();
            if !answer.trim().eq_ignore_ascii_case("y") {
                // Answer the calls ourselves, as if the tools node ran.
                let denied: Vec<Value> = calls
                    .iter()
                    .map(|c| Message::tool(c.id.clone(), "Denied by the user.").to_value())
                    .collect();
                agent
                    .update_state(&config, json!({ "messages": denied }), Some("tools"))
                    .await?;
            }
            print!("bot> ");
            interrupts = stream_run(&agent, Value::Null, &config).await?;
        }
    }
    Ok(())
}
