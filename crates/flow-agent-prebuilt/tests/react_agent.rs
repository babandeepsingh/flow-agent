//! ReAct agent behaviour with the fake model.

use std::sync::Arc;

use flow_agent_checkpoint::MemorySaver;
use flow_agent_core::prelude::*;
use flow_agent_models::{ChatModel, FakeChatModel, last_text};
use flow_agent_prebuilt::{ITERATION_LIMIT_MESSAGE, ReactAgent, ToolErrorMode, create_react_agent};
use futures::StreamExt;

fn weather() -> Tool {
    Tool::new("get_weather", "Weather for a city")
        .param("city", "string", "City")
        .handler(|a| async move { Ok(json!({ "city": a["city"], "forecast": "sunny" })) })
}

fn failing() -> Tool {
    Tool::new("flaky", "Always fails")
        .handler(|_| async { Err(Error::tool("flaky", "service down")) })
}

#[tokio::test]
async fn tool_loop_with_parallel_calls_and_system_prompt() {
    let fake = FakeChatModel::new([
        Message::assistant("")
            .with_tool_call("get_weather", json!({"city": "Paris"}))
            .with_tool_call("get_weather", json!({"city": "Oslo"})),
        Message::assistant("Paris and Oslo are sunny."),
    ]);
    let agent = ReactAgent::builder(fake.clone().into())
        .tools([weather()])
        .system_prompt("Be brief.")
        .build()
        .unwrap();
    let out = agent
        .invoke(json!("Weather in Paris and Oslo?"))
        .await
        .unwrap();
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 5); // user, ai(tool calls), tool, tool, ai
    assert_eq!(msgs[2]["role"], "tool");
    assert!(msgs[3]["content"].as_str().unwrap().contains("Oslo"));
    assert_eq!(last_text(&out), "Paris and Oslo are sunny.");
    assert!(
        msgs.iter().all(|m| m["role"] != "system"),
        "system prompt is not stored in state"
    );

    let calls = fake.calls();
    assert_eq!(calls[0].messages[0].role, Role::System);
    assert_eq!(calls[0].tools[0].name, "get_weather");
    assert_eq!(calls[1].messages.len(), 5); // system + 4
}

#[tokio::test]
async fn tool_errors_go_back_to_model_or_propagate() {
    let fake = FakeChatModel::new([
        Message::assistant("").with_tool_call("flaky", json!({})),
        Message::assistant("The service is down, sorry."),
    ]);
    let agent = create_react_agent(fake.clone().into(), [failing()]).unwrap();
    let out = agent.invoke(json!("go")).await.unwrap();
    assert!(
        out["messages"][2]["content"]
            .as_str()
            .unwrap()
            .starts_with("Error:")
    );
    assert_eq!(last_text(&out), "The service is down, sorry.");

    let fake = FakeChatModel::new([Message::assistant("").with_tool_call("flaky", json!({}))]);
    let agent = ReactAgent::builder(fake.into())
        .tools([failing()])
        .tool_errors(ToolErrorMode::Propagate)
        .build()
        .unwrap();
    let err = agent.invoke(json!("go")).await.unwrap_err();
    assert!(err.to_string().contains("service down"), "{err}");
}

#[tokio::test]
async fn max_iterations_stops_endless_tool_use() {
    let looping: ChatModel = FakeChatModel::new([
        Message::assistant("").with_tool_call("get_weather", json!({"city": "X"}))
    ])
    .cycle()
    .into();
    let agent = ReactAgent::builder(looping)
        .tools([weather()])
        .max_iterations(3)
        .build()
        .unwrap();
    let out = agent.invoke(json!("loop")).await.unwrap();
    assert_eq!(last_text(&out), ITERATION_LIMIT_MESSAGE);
}

#[tokio::test]
async fn human_approval_before_tools_and_multi_turn() {
    let fake = FakeChatModel::new([
        Message::assistant("").with_tool_call("get_weather", json!({"city": "Rome"})),
        Message::assistant("Rome is sunny."),
        Message::assistant("You're welcome!"),
    ]);
    let agent = ReactAgent::builder(fake.into())
        .tools([weather()])
        .interrupt_before_tools(true)
        .checkpointer(Arc::new(MemorySaver::new()))
        .build()
        .unwrap();
    let cfg = RunConfig::new().thread_id("trip");

    let paused = agent.run(json!("Weather in Rome?"), &cfg).await.unwrap();
    assert_eq!(paused.interrupts[0].node, "tools");
    let pending = agent.get_state(&cfg).await.unwrap().unwrap();
    assert_eq!(
        pending.values["messages"][1]["tool_calls"][0]["args"]["city"],
        "Rome"
    );

    let done = agent.resume(Value::Null, &cfg).await.unwrap();
    assert_eq!(last_text(&done), "Rome is sunny.");

    let next = agent.invoke_with(json!("Thanks!"), &cfg).await.unwrap();
    assert_eq!(last_text(&next), "You're welcome!");
    assert_eq!(next["messages"].as_array().unwrap().len(), 6);

    assert!(
        ReactAgent::builder(FakeChatModel::echo().into())
            .interrupt_before_tools(true)
            .build()
            .is_err(),
        "approval without a checkpointer is rejected"
    );
}

#[tokio::test]
async fn streams_tokens_and_tool_events() {
    let model: ChatModel = FakeChatModel::new([
        Message::assistant("").with_tool_call("get_weather", json!({"city": "Lima"})),
        Message::assistant("Lima is sunny"),
    ])
    .into();
    let agent = create_react_agent(model, [weather()]).unwrap();
    let kinds: Vec<EventKind> = agent
        .stream(json!("Weather in Lima?"))
        .map(|e| e.unwrap().kind())
        .collect()
        .await;
    for k in [
        EventKind::ToolStart,
        EventKind::ToolEnd,
        EventKind::Token,
        EventKind::Output,
    ] {
        assert!(kinds.contains(&k), "missing {k:?} in {kinds:?}");
    }
}

#[tokio::test]
async fn sequential_tool_calls_keep_order() {
    let fake = FakeChatModel::new([
        Message::assistant("")
            .with_tool_call("get_weather", json!({"city": "A"}))
            .with_tool_call("get_weather", json!({"city": "B"})),
        Message::assistant("done"),
    ]);
    let agent = ReactAgent::builder(fake.into())
        .tools([weather()])
        .parallel_tool_calls(false)
        .build()
        .unwrap();
    let out = agent
        .invoke(json!({"messages": [{"role": "user", "content": "x"}]}))
        .await
        .unwrap();
    assert!(
        out["messages"][2]["content"]
            .as_str()
            .unwrap()
            .contains("\"A\"")
    );
    assert!(
        out["messages"][3]["content"]
            .as_str()
            .unwrap()
            .contains("\"B\"")
    );
}
