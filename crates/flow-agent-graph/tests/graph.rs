//! Graph execution: edges, reducers, parallelism, validation, limits,
//! streaming, interrupts, resume, state editing and replay.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use flow_agent_core::prelude::*;
use flow_agent_core::{Checkpoint, Checkpointer, ListFilter, PendingWrite};
use flow_agent_graph::{
    Command, CompileOptions, END, NodeContext, NodeOptions, Reducer, START, Send, StateGraph,
};
use flow_agent_models::{ChatModel, FakeChatModel};
use futures::StreamExt;

/// Minimal checkpointer for these tests (real ones live in flow-agent-checkpoint).
#[derive(Default)]
struct TestSaver(Mutex<Vec<Checkpoint>>);

#[async_trait]
impl Checkpointer for TestSaver {
    async fn put(&self, cp: &Checkpoint) -> Result<()> {
        self.0.lock().unwrap().push(cp.clone());
        Ok(())
    }
    async fn put_writes(&self, _: &str, _: &str, _: &str, _: &[PendingWrite]) -> Result<()> {
        Ok(())
    }
    async fn get(&self, thread: &str, _ns: &str, id: Option<&str>) -> Result<Option<Checkpoint>> {
        let all = self.0.lock().unwrap();
        Ok(all
            .iter()
            .rev()
            .find(|c| c.thread_id == thread && id.is_none_or(|id| c.id == id))
            .cloned())
    }
    async fn list(&self, thread: &str, _f: ListFilter) -> Result<Vec<Checkpoint>> {
        let all = self.0.lock().unwrap();
        Ok(all
            .iter()
            .rev()
            .filter(|c| c.thread_id == thread)
            .cloned()
            .collect())
    }
    async fn delete_thread(&self, thread: &str) -> Result<()> {
        self.0.lock().unwrap().retain(|c| c.thread_id != thread);
        Ok(())
    }
}

fn log(
    name: &'static str,
) -> impl Fn(Value, NodeContext) -> futures::future::BoxFuture<'static, Result<Value>> {
    move |_s, _c| Box::pin(async move { Ok(json!({ "log": [name] })) })
}

fn thread(id: &str) -> RunConfig {
    RunConfig::new().thread_id(id)
}

#[tokio::test]
async fn parallel_fan_out_and_fan_in() {
    let app = StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("a", log("a"))
        .add_node("b", log("b"))
        .add_node("c", log("c"))
        .add_node("join", log("join"))
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("a", "c")
        .add_edge_from(["b", "c"], "join")
        .add_edge("join", END)
        .compile()
        .unwrap();
    let out = app.invoke(json!({})).await.unwrap();
    // b and c run in the same step; updates are applied in node-name order.
    assert_eq!(out["log"], json!(["a", "b", "c", "join"]));
}

#[tokio::test]
async fn fan_in_waits_for_sources_in_different_steps() {
    let app = StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("fast", log("fast"))
        .add_node("slow1", log("slow1"))
        .add_node("slow2", log("slow2"))
        .add_node("join", log("join"))
        .add_edge(START, "fast")
        .add_edge(START, "slow1")
        .add_edge("slow1", "slow2")
        .add_edge_from(["fast", "slow2"], "join")
        .add_edge("join", END)
        .compile()
        .unwrap();
    let out = app.invoke(json!({})).await.unwrap();
    assert_eq!(out["log"], json!(["fast", "slow1", "slow2", "join"]));
}

#[tokio::test]
async fn overwrite_conflict_in_same_step_is_an_error() {
    let app = StateGraph::new()
        .add_node("a", |_s: Value, _c: NodeContext| async {
            Ok(json!({"x": 1}))
        })
        .add_node("b", |_s: Value, _c: NodeContext| async {
            Ok(json!({"x": 2}))
        })
        .add_edge(START, "a")
        .add_edge(START, "b")
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
        .unwrap();
    let err = app.invoke(json!({})).await.unwrap_err();
    assert!(matches!(err.root(), Error::InvalidUpdate { .. }), "{err}");
}

#[tokio::test]
async fn conditional_loop_and_recursion_limit() {
    let build = || {
        StateGraph::new()
            .add_node("inc", |s: Value, _c: NodeContext| async move {
                Ok(json!({"n": s["n"].as_i64().unwrap_or(0) + 1}))
            })
            .add_edge(START, "inc")
            .add_conditional_edges_with(
                "inc",
                |s: &Value| {
                    if s["n"].as_i64() < Some(3) {
                        "again"
                    } else {
                        "done"
                    }
                },
                [("again", "inc"), ("done", END)],
            )
            .compile()
            .unwrap()
    };
    assert_eq!(build().invoke(json!({"n": 0})).await.unwrap()["n"], 3);
    let err = build()
        .invoke_with(json!({"n": -100}), &RunConfig::new().recursion_limit(5))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::RecursionLimit { limit: 5 }));
}

#[tokio::test]
async fn send_map_reduce() {
    let app = StateGraph::new()
        .key("results", Reducer::Append)
        .add_node("split", |_s: Value, _c: NodeContext| async {
            Ok(json!(null))
        })
        .add_node("square", |arg: Value, _c: NodeContext| async move {
            let n = arg["n"].as_i64().unwrap_or(0);
            Ok(json!({ "results": [n * n] }))
        })
        .add_edge(START, "split")
        .add_conditional_edges("split", |s: &Value| {
            s["items"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|n| Send::new("square", json!({ "n": n })))
                .collect::<Vec<_>>()
        })
        .add_edge("square", END)
        .compile()
        .unwrap();
    let out = app.invoke(json!({"items": [1, 2, 3]})).await.unwrap();
    assert_eq!(out["results"], json!([1, 4, 9]));
}

#[tokio::test]
async fn command_goto() {
    let app = StateGraph::new()
        .add_node_with(
            "decide",
            |s: Value, _c: NodeContext| async move {
                let target = if s["vip"] == true { "vip" } else { "normal" };
                Ok(Command::goto(target).update(json!({"decided": true})))
            },
            NodeOptions::new().routes_to(["vip", "normal"]),
        )
        .add_node("vip", |_s: Value, _c: NodeContext| async {
            Ok(json!({"lane": "vip"}))
        })
        .add_node("normal", |_s: Value, _c: NodeContext| async {
            Ok(json!({"lane": "normal"}))
        })
        .add_edge(START, "decide")
        .add_edge("vip", END)
        .add_edge("normal", END)
        .compile()
        .unwrap();
    let out = app.invoke(json!({"vip": true})).await.unwrap();
    assert_eq!(out["lane"], "vip");
    assert_eq!(out["decided"], true);
}

#[test]
fn compile_reports_all_problems() {
    let err = StateGraph::new()
        .add_node("a", log("a"))
        .add_node("a", log("a"))
        .add_node("orphan", log("orphan"))
        .add_node(END, log("x"))
        .add_edge("a", "missing")
        .add_edge("orphan", END)
        .compile_with(CompileOptions::new().interrupt_before(["ghost"]))
        .unwrap_err();
    let Error::InvalidGraph(problems) = err else {
        panic!("expected InvalidGraph")
    };
    let all = problems.join("\n");
    for needle in [
        "added twice",
        "invalid node name",
        "no entry point",
        "unknown node 'missing'",
        "ghost",
    ] {
        assert!(all.contains(needle), "missing '{needle}' in:\n{all}");
    }

    let err = StateGraph::new()
        .add_node("a", log("a"))
        .add_node("dead", log("dead"))
        .add_node("island", log("island"))
        .add_edge(START, "a")
        .add_edge("a", "dead")
        .add_edge("island", END)
        .compile()
        .unwrap_err();
    let all = err.to_string();
    assert!(all.contains("'dead' has no outgoing edge"), "{all}");

    let err = StateGraph::new()
        .add_node("a", log("a"))
        .add_node("island", log("island"))
        .add_edge(START, "a")
        .add_edge("a", END)
        .add_edge("island", END)
        .compile()
        .unwrap_err();
    assert!(err.to_string().contains("unreachable"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn node_timeout_retry_and_cancellation() {
    let calls = Arc::new(AtomicU32::new(0));
    let c = calls.clone();
    let app = StateGraph::new()
        .add_node_with(
            "flaky",
            move |_s: Value, _ctx: NodeContext| {
                let c = c.clone();
                async move {
                    if c.fetch_add(1, Ordering::SeqCst) < 2 {
                        Err(Error::Http("blip".into()))
                    } else {
                        Ok(json!({"ok": true}))
                    }
                }
            },
            NodeOptions::new().retry(RetryPolicy::exponential(3)),
        )
        .add_edge(START, "flaky")
        .add_edge("flaky", END)
        .compile()
        .unwrap();
    assert_eq!(app.invoke(json!({})).await.unwrap()["ok"], true);
    assert_eq!(calls.load(Ordering::SeqCst), 3);

    let slow = StateGraph::new()
        .add_node_with(
            "slow",
            |_s: Value, _c: NodeContext| async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(json!({}))
            },
            NodeOptions::new().timeout(Duration::from_secs(1)),
        )
        .add_edge(START, "slow")
        .add_edge("slow", END)
        .compile()
        .unwrap();
    let err = slow.invoke(json!({})).await.unwrap_err();
    match &err {
        Error::NodeFailed { node, source, .. } => {
            assert_eq!(node, "slow");
            assert!(matches!(**source, Error::Timeout { .. }));
        }
        other => panic!("unexpected {other:?}"),
    }

    let token = tokio_util::sync::CancellationToken::new();
    let cfg = RunConfig::new()
        .cancellation(token.clone())
        .timeout(Duration::from_secs(600));
    let app2 = StateGraph::new()
        .add_node("wait", |_s: Value, _c: NodeContext| async {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(json!({}))
        })
        .add_edge(START, "wait")
        .add_edge("wait", END)
        .compile()
        .unwrap();
    let h = tokio::spawn(async move { app2.invoke_with(json!({}), &cfg).await });
    tokio::time::sleep(Duration::from_millis(5)).await;
    token.cancel();
    assert!(matches!(h.await.unwrap().unwrap_err(), Error::Cancelled));
}

#[tokio::test]
async fn stream_events_with_model_tokens() {
    let model: ChatModel = FakeChatModel::new([Message::assistant("Hello there")]).into();
    let app = StateGraph::new()
        .key("messages", Reducer::Messages)
        .add_node("agent", model)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile()
        .unwrap();
    let events: Vec<Event> = app
        .stream(json!({"messages": ["hi"]}))
        .map(|e| e.unwrap())
        .collect()
        .await;
    let kinds: Vec<EventKind> = events.iter().map(Event::kind).collect();
    assert_eq!(kinds.first(), Some(&EventKind::NodeStart));
    assert!(kinds.contains(&EventKind::Token));
    let token_node = events.iter().find_map(|e| match e {
        Event::Token { node, .. } => node.clone(),
        _ => None,
    });
    assert_eq!(token_node.as_deref(), Some("agent"));
    let pos = |k| kinds.iter().position(|x| *x == k).unwrap();
    assert!(pos(EventKind::Token) < pos(EventKind::NodeEnd));
    assert!(pos(EventKind::NodeEnd) < pos(EventKind::StateUpdate));
    assert_eq!(kinds.last(), Some(&EventKind::Output));
    // Events serialize with a "type" tag.
    assert_eq!(events[0].to_value()["type"], "node_start");
}

#[tokio::test]
async fn subgraph_as_node() {
    let inner = StateGraph::new()
        .name("inner")
        .add_node("double", |s: Value, _c: NodeContext| async move {
            Ok(json!({"x": s["x"].as_i64().unwrap_or(0) * 2}))
        })
        .add_edge(START, "double")
        .add_edge("double", END)
        .compile()
        .unwrap();
    let outer = StateGraph::new()
        .add_node("sub", inner)
        .add_edge(START, "sub")
        .add_edge("sub", END)
        .compile()
        .unwrap();
    assert_eq!(outer.invoke(json!({"x": 21})).await.unwrap()["x"], 42);
}

fn approval_graph(saver: Arc<TestSaver>) -> flow_agent_graph::CompiledGraph {
    StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("draft", |_s: Value, _c: NodeContext| async {
            Ok(json!({"draft": "Dear customer...", "log": ["draft"]}))
        })
        .add_node("review", |s: Value, ctx: NodeContext| async move {
            let answer = ctx.interrupt(json!({"question": "Send?", "draft": s["draft"]}))?;
            Ok(json!({"approved": answer["approved"], "log": ["review"]}))
        })
        .add_node("send", log("send"))
        .add_edge(START, "draft")
        .add_edge("draft", "review")
        .add_conditional_edges_with(
            "review",
            |s: &Value| if s["approved"] == true { "yes" } else { "no" },
            [("yes", "send"), ("no", END)],
        )
        .add_edge("send", END)
        .compile_with(CompileOptions::new().checkpointer(saver))
        .unwrap()
}

#[tokio::test]
async fn interrupt_and_resume() {
    let saver = Arc::new(TestSaver::default());
    let app = approval_graph(saver.clone());
    let cfg = thread("t1");

    let out = app.run(json!({}), &cfg).await.unwrap();
    assert!(out.is_interrupted());
    assert_eq!(out.interrupts[0].node, "review");
    assert_eq!(out.interrupts[0].value["question"], "Send?");
    let snap = app.get_state(&cfg).await.unwrap().unwrap();
    assert_eq!(snap.next, vec!["review"]);
    assert_eq!(snap.interrupts.len(), 1);

    let v = app.invoke_with(json!(null), &cfg).await.unwrap();
    assert!(
        v.get("__interrupt__").is_some(),
        "continuing without a value interrupts again"
    );

    let done = app.resume(json!({"approved": true}), &cfg).await.unwrap();
    assert!(done.get("__interrupt__").is_none());
    assert_eq!(done["log"], json!(["draft", "review", "send"]));

    // Nothing left to resume.
    assert!(app.resume(json!(1), &cfg).await.is_err());
    // Runs without thread_id are rejected when a checkpointer is configured.
    assert!(app.invoke(json!({})).await.is_err());
}

#[tokio::test]
async fn interrupt_before_after_and_update_state() {
    let saver = Arc::new(TestSaver::default());
    let app = StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("plan", log("plan"))
        .add_node("act", log("act"))
        .add_node("report", log("report"))
        .add_edge(START, "plan")
        .add_edge("plan", "act")
        .add_edge("act", "report")
        .add_edge("report", END)
        .compile_with(
            CompileOptions::new()
                .checkpointer(saver)
                .interrupt_before(["act"])
                .interrupt_after(["act"]),
        )
        .unwrap();
    let cfg = thread("t2");

    let out = app.run(json!({}), &cfg).await.unwrap();
    assert_eq!(out.interrupts[0].id, "before:act");
    assert_eq!(out.state["log"], json!(["plan"]));

    // Edit state while paused, then continue.
    app.update_state(&cfg, json!({"note": "approved by ops"}), None)
        .await
        .unwrap();
    let out = app.run(json!(null), &cfg).await.unwrap();
    assert_eq!(out.interrupts[0].id, "after:act");
    assert_eq!(out.state["note"], "approved by ops");

    let out = app.run(json!(null), &cfg).await.unwrap();
    assert!(!out.is_interrupted());
    assert_eq!(out.state["log"], json!(["plan", "act", "report"]));

    let history = app.get_state_history(&cfg).await.unwrap();
    assert!(history.len() >= 5);
    assert!(history.windows(2).all(|w| w[0].step >= w[1].step));
    assert_eq!(history.last().unwrap().metadata["source"], "input");
}

#[tokio::test]
async fn replay_from_checkpoint_and_multi_turn_threads() {
    let saver = Arc::new(TestSaver::default());
    let model: ChatModel = FakeChatModel::echo().into();
    let app = StateGraph::new()
        .key("messages", Reducer::Messages)
        .add_node("bot", model)
        .add_edge(START, "bot")
        .add_edge("bot", END)
        .compile_with(CompileOptions::new().checkpointer(saver))
        .unwrap();
    let cfg = thread("chat");
    app.invoke_with(json!({"messages": ["one"]}), &cfg)
        .await
        .unwrap();
    let out = app
        .invoke_with(json!({"messages": ["two"]}), &cfg)
        .await
        .unwrap();
    assert_eq!(
        out["messages"].as_array().unwrap().len(),
        4,
        "second turn appends"
    );

    // Fork from the checkpoint taken right after the second input.
    let history = app.get_state_history(&cfg).await.unwrap();
    let fork_point = history
        .iter()
        .find(|s| {
            s.metadata["source"] == "input" && s.values["messages"].as_array().unwrap().len() == 3
        })
        .unwrap();
    let replay_cfg = thread("chat").checkpoint_id(&fork_point.checkpoint_id);
    let replayed = app.invoke_with(json!(null), &replay_cfg).await.unwrap();
    assert_eq!(replayed["messages"].as_array().unwrap().len(), 4);
    assert_eq!(replayed["messages"][3]["content"], "two");
}

#[tokio::test]
async fn failing_node_keeps_finished_writes() {
    let saver = Arc::new(TestSaver::default());
    let app = StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("good", log("good"))
        .add_node("bad", |_s: Value, _c: NodeContext| async {
            Err::<Value, _>(Error::InvalidInput("boom".into()))
        })
        .add_edge(START, "good")
        .add_edge(START, "bad")
        .add_edge("good", END)
        .add_edge("bad", END)
        .compile_with(CompileOptions::new().checkpointer(saver.clone()))
        .unwrap();
    let err = app.invoke_with(json!({}), &thread("t3")).await.unwrap_err();
    assert!(err.to_string().contains("node 'bad' failed"), "{err}");
    let last = saver.get("t3", "", None).await.unwrap().unwrap();
    assert_eq!(last.metadata["source"], "error");
}

#[tokio::test]
async fn resume_each_targets_interrupts_by_id() {
    let saver = Arc::new(TestSaver::default());
    let ask = |key: &'static str| {
        move |_s: Value, ctx: NodeContext| async move {
            let v = ctx.interrupt(json!(key))?;
            Ok(json!({ key: v }))
        }
    };
    let app = StateGraph::new()
        .add_node("q1", ask("a"))
        .add_node("q2", ask("b"))
        .add_edge(START, "q1")
        .add_edge(START, "q2")
        .add_edge("q1", END)
        .add_edge("q2", END)
        .compile_with(CompileOptions::new().checkpointer(saver))
        .unwrap();
    let cfg = thread("t4");
    let out = app.run(json!({}), &cfg).await.unwrap();
    assert_eq!(out.interrupts.len(), 2);
    let answers: BTreeMap<String, Value> = out
        .interrupts
        .iter()
        .map(|i| (i.id.clone(), json!(format!("answer-{}", i.node))))
        .collect();
    let done = app.resume_each(answers, &cfg).await.unwrap();
    assert_eq!(done.state["a"], "answer-q1");
    assert_eq!(done.state["b"], "answer-q2");
}

#[test]
fn mermaid_output() {
    let app = StateGraph::new()
        .add_node("a", log("a"))
        .add_edge(START, "a")
        .add_edge("a", END)
        .compile()
        .unwrap();
    let m = app.to_mermaid();
    assert!(m.starts_with("graph TD"));
    assert!(m.contains("__start__ --> a"));
}
