//! ChatModel behaviour with the fake provider: invoke, streaming, tools,
//! structured output, retries, timeouts, cancellation and events.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::prelude::*;
use flow_agent_core::runnable::collect_output;
use flow_agent_models::{ChatModel, FakeChatModel, ModelRegistry, ToolChoice, last_text};
use futures::StreamExt;

#[tokio::test]
async fn invoke_returns_assistant_message_with_id_and_usage() {
    let fake = FakeChatModel::new([Message::assistant("Hello there")]);
    let model: ChatModel = fake.clone().into();
    let out = model.invoke(json!("hi")).await.unwrap();
    assert_eq!(out["role"], "assistant");
    assert_eq!(out["content"], "Hello there");
    assert!(out["id"].is_string());
    assert!(out["usage"]["total_tokens"].is_number());
    assert_eq!(fake.calls()[0].messages[0].text(), "hi");
}

#[tokio::test]
async fn chain_prompt_model_parser() {
    let model: ChatModel = FakeChatModel::echo().into();
    let chain = ChatPromptTemplate::new()
        .system("be brief")
        .user("{q}")
        .pipe(model)
        .pipe(StrOutputParser::new());
    assert_eq!(
        chain.invoke(json!({"q": "ping"})).await.unwrap(),
        json!("ping")
    );
}

#[tokio::test]
async fn stream_emits_tokens_then_output() {
    let model: ChatModel = FakeChatModel::new([Message::assistant("It is sunny")]).into();
    let events: Vec<Event> = model
        .stream(json!("weather?"))
        .map(|e| e.unwrap())
        .collect()
        .await;
    let tokens: String = events
        .iter()
        .filter_map(|e| match e {
            Event::Token { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tokens, "It is sunny");
    assert!(matches!(events.first(), Some(Event::ModelStart { .. })));
    assert!(matches!(events[events.len() - 2], Event::ModelEnd { .. }));
    match events.last() {
        Some(Event::Output { value }) => assert_eq!(value["content"], "It is sunny"),
        other => panic!("unexpected last event {other:?}"),
    }
}

#[tokio::test]
async fn tokens_reach_outer_stream_through_a_chain() {
    let model: ChatModel = FakeChatModel::new([Message::assistant("a b c")]).into();
    let chain = model.pipe(StrOutputParser::new());
    let n = chain
        .stream(json!("x"))
        .filter(|e| futures::future::ready(matches!(e, Ok(Event::Token { .. }))))
        .count()
        .await;
    assert_eq!(n, 3);
    let out = collect_output(chain.stream(json!("x"))).await;
    // Script is exhausted on the second run.
    assert!(out.is_err());
}

#[tokio::test]
async fn bind_tools_and_streamed_tool_calls() {
    let fake = FakeChatModel::new([
        Message::assistant("").with_tool_call("get_weather", json!({"city": "Paris"}))
    ]);
    let tools = ToolRegistry::new().with(
        Tool::new("get_weather", "Weather")
            .param("city", "string", "City")
            .handler(|a| async move { Ok(a) }),
    );
    let model = ChatModel::from(fake.clone()).bind_tools(&tools);
    let events: Vec<Event> = model
        .stream(json!("weather in Paris?"))
        .map(|e| e.unwrap())
        .collect()
        .await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::ToolCallChunk { name: Some(n), .. } if n == "get_weather"))
    );
    let Some(Event::Output { value }) = events.last() else {
        panic!("no output")
    };
    assert_eq!(value["tool_calls"][0]["args"]["city"], "Paris");

    let req = &fake.calls()[0];
    assert_eq!(req.tools[0].name, "get_weather");
    assert_eq!(req.tool_choice, ToolChoice::Auto);
}

#[tokio::test]
async fn structured_output() {
    let schema = json!({
        "title": "person",
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]
    });
    let fake = FakeChatModel::new([
        Message::assistant("").with_tool_call("person", json!({"name": "Ada", "age": 36})),
        Message::assistant("```json\n{\"name\": \"Bob\", \"age\": 5}\n```"),
        Message::assistant("{\"name\": \"Cy\"}"),
    ]);
    let model = ChatModel::from(fake.clone()).with_structured_output(schema);
    assert_eq!(
        model.invoke(json!("Ada is 36")).await.unwrap(),
        json!({"name": "Ada", "age": 36})
    );
    assert_eq!(
        fake.calls()[0].tool_choice,
        ToolChoice::Tool("person".into())
    );
    assert_eq!(model.invoke(json!("x")).await.unwrap()["name"], "Bob");
    assert!(matches!(
        model.invoke(json!("x")).await.unwrap_err(),
        Error::Parse { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn retries_retryable_errors() {
    let fake = FakeChatModel::new([Message::assistant("ok")]).fail_times(2);
    let model = ChatModel::from(fake.clone());
    assert_eq!(model.invoke(json!("x")).await.unwrap()["content"], "ok");
    assert_eq!(fake.calls().len(), 3);

    let fake = FakeChatModel::new([Message::assistant("ok")]).fail_times(2);
    let model = ChatModel::from(fake.clone()).with_max_retries(1);
    assert!(model.invoke(json!("x")).await.unwrap_err().is_retryable());
}

#[tokio::test(start_paused = true)]
async fn timeout_and_cancellation() {
    let slow = FakeChatModel::echo().with_delay(Duration::from_secs(30));
    let model = ChatModel::from(slow)
        .with_timeout(Duration::from_secs(1))
        .with_max_retries(0);
    assert!(matches!(
        model.invoke(json!("x")).await.unwrap_err(),
        Error::Timeout { .. }
    ));

    let slow = FakeChatModel::echo().with_delay(Duration::from_secs(30));
    let model = ChatModel::from(slow);
    let token = tokio_util::sync::CancellationToken::new();
    let config = RunConfig::new().cancellation(token.clone());
    let handle = tokio::spawn(async move { model.invoke_with(json!("x"), &config).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    token.cancel();
    assert!(matches!(
        handle.await.unwrap().unwrap_err(),
        Error::Cancelled
    ));
}

#[derive(Default)]
struct Kinds(Mutex<Vec<EventKind>>);

#[async_trait]
impl CallbackHandler for Kinds {
    async fn on_event(&self, event: &Event, _config: &RunConfig) {
        self.0.lock().unwrap().push(event.kind());
    }
}

#[tokio::test]
async fn callbacks_see_model_events_with_node() {
    let kinds = Arc::new(Kinds::default());
    let model: ChatModel = FakeChatModel::new([Message::assistant("hi you")]).into();
    let config = RunConfig::new()
        .callback_arc(kinds.clone())
        .with_node("agent");
    model.invoke_with(json!("x"), &config).await.unwrap();
    let seen = kinds.0.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            EventKind::ModelStart,
            EventKind::Token,
            EventKind::Token,
            EventKind::ModelEnd
        ]
    );
}

#[tokio::test]
async fn empty_input_is_rejected() {
    let model: ChatModel = FakeChatModel::echo().into();
    assert!(matches!(
        model.invoke(json!([])).await.unwrap_err(),
        Error::InvalidInput(_)
    ));
}

#[test]
fn builder_errors_are_helpful() {
    let err = ChatModel::builder("claude").build().unwrap_err();
    assert!(err.to_string().contains("provider:model"), "{err}");
    let err = ChatModel::builder("nope:model").build().unwrap_err();
    assert!(err.to_string().contains("unknown model provider"), "{err}");

    let registry = ModelRegistry::empty().register("mock", |cfg| {
        Ok(Arc::new(FakeChatModel::echo().with_model(cfg.model)))
    });
    let m = ChatModel::builder("mock:tiny")
        .registry(&registry)
        .temperature(0.1)
        .build()
        .unwrap();
    assert_eq!(m.spec(), "fake:tiny");
    assert_eq!(m.options().temperature, Some(0.1));
}

#[test]
fn last_text_helper() {
    assert_eq!(
        last_text(&json!({"messages": [{"role": "assistant", "content": "done"}]})),
        "done"
    );
    assert_eq!(last_text(&json!("plain")), "plain");
}
