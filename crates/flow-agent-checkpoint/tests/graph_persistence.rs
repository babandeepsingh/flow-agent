//! Graphs + real savers: multi-turn threads, interrupts that survive a
//! restart, and replay.

use std::sync::Arc;

use flow_agent_checkpoint::MemorySaver;
use flow_agent_core::Checkpointer;
use flow_agent_core::prelude::*;
use flow_agent_graph::{CompileOptions, CompiledGraph, NodeContext, Reducer, START, StateGraph};

fn approval_app(saver: Arc<dyn Checkpointer>) -> CompiledGraph {
    StateGraph::new()
        .key("log", Reducer::Append)
        .add_node("draft", |_s: Value, _c: NodeContext| async {
            Ok(json!({"draft": "Hello!", "log": ["draft"]}))
        })
        .add_node("review", |s: Value, ctx: NodeContext| async move {
            let ok = ctx.interrupt(json!({"approve": s["draft"]}))?;
            Ok(json!({"approved": ok, "log": ["review"]}))
        })
        .add_edge(START, "draft")
        .add_edge("draft", "review")
        .add_edge("review", flow_agent_graph::END)
        .compile_with(CompileOptions::new().checkpointer(saver))
        .unwrap()
}

#[tokio::test]
async fn memory_saver_interrupt_resume_and_history() {
    let saver = Arc::new(MemorySaver::new());
    let app = approval_app(saver.clone());
    let cfg = RunConfig::new().thread_id("m1");

    let out = app.run(json!({}), &cfg).await.unwrap();
    assert!(out.is_interrupted());
    let done = app.resume(json!(true), &cfg).await.unwrap();
    assert_eq!(done["approved"], true);
    assert_eq!(done["log"], json!(["draft", "review"]));

    let history = app.get_state_history(&cfg).await.unwrap();
    assert!(history.len() >= 3);
    assert!(!saver.is_empty().await);
    saver.delete_thread("m1").await.unwrap();
    assert!(app.get_state(&cfg).await.unwrap().is_none());
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_interrupt_survives_restart() {
    use flow_agent_checkpoint::SqliteSaver;

    let path = std::env::temp_dir().join(format!("flow-agent-{}.db", uuid::Uuid::now_v7()));
    let url = format!("sqlite://{}", path.display());
    let cfg = RunConfig::new().thread_id("ticket-7");

    {
        let saver = SqliteSaver::connect(&url).await.unwrap();
        saver.migrate().await.unwrap();
        let app = approval_app(Arc::new(saver));
        let out = app.run(json!({}), &cfg).await.unwrap();
        assert_eq!(out.interrupts[0].value["approve"], "Hello!");
    } // process "restarts": everything dropped

    let saver = SqliteSaver::connect(&url).await.unwrap();
    let app = approval_app(Arc::new(saver));
    let snap = app.get_state(&cfg).await.unwrap().unwrap();
    assert_eq!(snap.next, vec!["review"]);
    let done = app.resume(json!("yes"), &cfg).await.unwrap();
    assert_eq!(done["approved"], "yes");
    assert_eq!(done["log"], json!(["draft", "review"]));
    let _ = std::fs::remove_file(&path);
}
