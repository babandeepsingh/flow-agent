//! A conformance suite every [`Checkpointer`] should pass. Use it to test
//! your own backend (Redis, DynamoDB...):
//!
//! ```
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use flow_agent_checkpoint::{MemorySaver, testing::run_conformance};
//! run_conformance(&MemorySaver::new()).await.unwrap();
//! # });
//! ```

use flow_agent_core::{
    Checkpoint, Checkpointer, Error, Interrupt, ListFilter, PendingWrite, Result,
};
use serde_json::json;

fn check(cond: bool, what: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(Error::Checkpoint(format!("conformance failed: {what}")))
    }
}

/// Run every check against `saver`. Uses thread ids prefixed with
/// `conformance-`, and deletes them at the end.
pub async fn run_conformance(saver: &dyn Checkpointer) -> Result<()> {
    let t = format!("conformance-{}", uuid_like());
    let other = format!("{t}-other");

    check(
        saver.get(&t, "", None).await?.is_none(),
        "empty thread returns None",
    )?;

    let mut first = Checkpoint::new(&t, 0, json!({"n": 0}));
    first.metadata.insert("source".into(), json!("input"));
    saver.put(&first).await?;
    let mut second = Checkpoint::new(&t, 1, json!({"n": 1, "nested": {"a": [1, 2]}}));
    second.parent_id = Some(first.id.clone());
    second.next = vec!["b".into()];
    second.interrupts.push(Interrupt {
        id: "1.0.0".into(),
        node: "b".into(),
        value: json!({"q": "ok?"}),
    });
    saver.put(&second).await?;
    saver.put(&Checkpoint::new(&other, 0, json!({}))).await?;

    let latest = saver.get(&t, "", None).await?;
    check(
        latest.as_ref() == Some(&second),
        "latest is the newest checkpoint, round-tripped exactly",
    )?;
    let by_id = saver.get(&t, "", Some(&first.id)).await?;
    check(by_id.as_ref() == Some(&first), "get by id")?;
    check(
        saver.get(&t, "", Some("missing")).await?.is_none(),
        "unknown id returns None",
    )?;
    check(
        saver.get(&t, "sub", None).await?.is_none(),
        "namespaces are separate",
    )?;

    let all = saver.list(&t, ListFilter::default()).await?;
    check(all.len() == 2, "list returns the thread's checkpoints only")?;
    check(
        all[0].id == second.id && all[1].id == first.id,
        "list is newest first",
    )?;
    let limited = saver
        .list(
            &t,
            ListFilter {
                limit: Some(1),
                ..Default::default()
            },
        )
        .await?;
    check(
        limited.len() == 1 && limited[0].id == second.id,
        "list limit",
    )?;
    let before = saver
        .list(
            &t,
            ListFilter {
                before: Some(second.id.clone()),
                ..Default::default()
            },
        )
        .await?;
    check(before.len() == 1 && before[0].id == first.id, "list before")?;

    let w = PendingWrite {
        task_id: "0".into(),
        node: "a".into(),
        update: json!({"x": 1}),
    };
    saver
        .put_writes(&t, "", &second.id, std::slice::from_ref(&w))
        .await?;
    saver
        .put_writes(&t, "", &second.id, std::slice::from_ref(&w))
        .await?;
    let with_writes = saver.get(&t, "", None).await?;
    check(
        with_writes.map(|c| c.pending_writes) == Some(vec![w]),
        "pending writes are stored once per task and returned with the checkpoint",
    )?;

    saver.put(&second).await?;
    check(
        saver.list(&t, ListFilter::default()).await?.len() == 2,
        "re-putting an id doesn't duplicate",
    )?;

    saver.delete_thread(&t).await?;
    check(
        saver.get(&t, "", None).await?.is_none(),
        "delete_thread removes the thread",
    )?;
    check(
        saver.get(&other, "", None).await?.is_some(),
        "delete_thread leaves other threads",
    )?;
    saver.delete_thread(&other).await?;
    Ok(())
}

fn uuid_like() -> String {
    Checkpoint::new("", 0, json!(null)).id
}
