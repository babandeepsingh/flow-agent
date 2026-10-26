//! Integration tests for runnables, chaining, streaming, callbacks and cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::prelude::*;
use flow_agent_core::runnable::{collect_output, stream_owned};
use flow_agent_core::{BoxRunnable, ToolCall};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

fn upper() -> RunnableLambda {
    RunnableLambda::new("upper", |v| async move {
        Ok(json!(v.as_str().unwrap_or_default().to_uppercase()))
    })
}

/// Emits two tokens through the config, like a streaming model would.
fn fake_streamer() -> RunnableLambda {
    RunnableLambda::with_config("streamer", |_v, cfg: RunConfig| async move {
        for t in ["Hel", "lo"] {
            cfg.emit(Event::Token {
                run_id: cfg.run_id,
                node: cfg.node.clone(),
                text: t.into(),
            })
            .await;
        }
        Ok(json!("Hello"))
    })
}

#[tokio::test]
async fn prompt_pipe_lambda_pipe_parser() {
    let chain = ChatPromptTemplate::new()
        .system("Be {style}.")
        .user("{q}")
        .pipe(RunnableLambda::new("fake_model", |msgs| async move {
            let last = Message::list_from_value(msgs)?.pop().map(|m| m.text());
            Ok(Message::assistant(format!("echo: {}", last.unwrap_or_default())).to_value())
        }))
        .pipe(StrOutputParser::new());

    assert_eq!(chain.steps().len(), 3, "pipe on a sequence stays flat");
    let out = chain
        .invoke(json!({"style": "brief", "q": "hi"}))
        .await
        .unwrap();
    assert_eq!(out, json!("echo: hi"));
}

#[tokio::test]
async fn sequence_error_has_step_context() {
    let chain = PromptTemplate::new("{a}").pipe(upper());
    let err = chain.invoke(json!({})).await.unwrap_err();
    assert!(err.to_string().contains("step 0"), "{err}");
    assert!(matches!(err.root(), Error::Template { .. }));
}

#[tokio::test]
async fn parallel_and_passthrough() {
    let par = RunnableParallel::new()
        .branch("up", upper())
        .branch("same", RunnablePassthrough::new());
    assert_eq!(
        par.invoke(json!("x")).await.unwrap(),
        json!({"up": "X", "same": "x"})
    );

    let assign = RunnablePassthrough::new().assign(
        "n",
        RunnableLambda::new("len", |v| async move {
            Ok(json!(v["s"].as_str().map(str::len).unwrap_or(0)))
        }),
    );
    assert_eq!(
        assign.invoke(json!({"s": "abc"})).await.unwrap(),
        json!({"s": "abc", "n": 3})
    );
    assert!(assign.invoke(json!("not an object")).await.is_err());
}

#[tokio::test(start_paused = true)]
async fn retry_fallbacks_timeout() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let flaky = RunnableLambda::new("flaky", move |v| {
        let c = c.clone();
        async move {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(Error::Http("reset".into()))
            } else {
                Ok(v)
            }
        }
    });
    let r = flaky.with_retry(RetryPolicy::exponential(3));
    assert_eq!(r.invoke(json!(1)).await.unwrap(), json!(1));
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let broken = RunnableLambda::new("broken", |_| async { Err(Error::Http("down".into())) });
    let fb = broken.with_fallbacks([upper()]);
    assert_eq!(fb.invoke(json!("a")).await.unwrap(), json!("A"));

    let slow = RunnableLambda::new("slow", |v| async move {
        tokio::time::sleep(Duration::from_secs(10)).await;
        Ok(v)
    })
    .with_timeout(Duration::from_secs(1));
    assert!(matches!(
        slow.invoke(json!(null)).await.unwrap_err(),
        Error::Timeout { .. }
    ));
}

#[tokio::test]
async fn batch_keeps_order_and_isolates_errors() {
    let r = RunnableLambda::new("maybe", |v| async move {
        if v == json!(2) {
            Err(Error::InvalidInput("two".into()))
        } else {
            Ok(json!(v.as_i64().unwrap_or(0) * 10))
        }
    });
    let out = r
        .batch_with(
            vec![json!(1), json!(2), json!(3)],
            &RunConfig::new().max_concurrency(2),
        )
        .await;
    assert_eq!(out[0].as_ref().unwrap(), &json!(10));
    assert!(out[1].is_err());
    assert_eq!(out[2].as_ref().unwrap(), &json!(30));
}

#[tokio::test]
async fn stream_forwards_nested_events_then_output() {
    let chain = fake_streamer().pipe(upper());
    let events: Vec<Event> = chain
        .stream(json!(null))
        .map(|e| e.unwrap())
        .collect()
        .await;
    let kinds: Vec<EventKind> = events.iter().map(Event::kind).collect();
    assert_eq!(
        kinds,
        vec![EventKind::Token, EventKind::Token, EventKind::Output]
    );
    assert_eq!(
        events[2],
        Event::Output {
            value: json!("HELLO")
        }
    );
}

#[tokio::test]
async fn stream_owned_is_static_and_reports_errors() {
    let r: BoxRunnable = Arc::new(fake_streamer());
    let s = stream_owned(r, json!(null), &RunConfig::new());
    let handle = tokio::spawn(async move { collect_output(s).await });
    assert_eq!(handle.await.unwrap().unwrap(), Some(json!("Hello")));

    let failing = RunnableLambda::new("f", |_| async { Err(Error::Cancelled) });
    let items: Vec<_> = failing.stream(json!(null)).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_err());
}

#[derive(Default)]
struct Recorder {
    events: std::sync::Mutex<Vec<EventKind>>,
    errors: AtomicUsize,
}

#[async_trait]
impl CallbackHandler for Recorder {
    async fn on_event(&self, event: &Event, _config: &RunConfig) {
        self.events.lock().unwrap().push(event.kind());
    }
    async fn on_error(&self, _error: &Error, _config: &RunConfig) {
        self.errors.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn callbacks_receive_tool_events() {
    let rec = Arc::new(Recorder::default());
    let config = RunConfig::new().callback_arc(rec.clone());
    let tools = ToolRegistry::new().with(
        Tool::new("add", "Add")
            .param("a", "integer", "")
            .param("b", "integer", "")
            .handler(|args| async move {
                Ok(json!(
                    args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0)
                ))
            }),
    );
    let res = tools
        .call(&ToolCall::new("add", json!({"a": 2, "b": 3})), &config)
        .await;
    assert_eq!(res.output, json!(5));
    let _ = tools.call(&ToolCall::new("nope", json!({})), &config).await;

    let events = rec.events.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            EventKind::ToolStart,
            EventKind::ToolEnd,
            EventKind::ToolStart,
            EventKind::ToolEnd
        ]
    );
    assert_eq!(rec.errors.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_stops_chain() {
    let token = CancellationToken::new();
    let t = token.clone();
    let chain = RunnableLambda::new("cancel", move |v| {
        let t = t.clone();
        async move {
            t.cancel();
            Ok(v)
        }
    })
    .pipe(upper());
    let err = chain
        .invoke_with(json!("x"), &RunConfig::new().cancellation(token))
        .await
        .unwrap_err();
    assert!(matches!(err.root(), Error::Cancelled));
}

#[tokio::test]
async fn with_config_merges_tags() {
    let seen =
        RunnableLambda::with_config(
            "seen",
            |_v, cfg: RunConfig| async move { Ok(json!(cfg.tags)) },
        )
        .with_config(RunConfig::new().tag("default"));
    let out = seen
        .invoke_with(json!(null), &RunConfig::new().tag("call"))
        .await
        .unwrap();
    assert_eq!(out, json!(["call", "default"]));
}
