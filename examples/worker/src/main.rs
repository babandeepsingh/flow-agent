//! A background worker: jobs arrive on a queue, each runs a small graph
//! (classify a support ticket, then draft a reply), at most N at a time.
//! Ctrl-C cancels in-flight work and shuts down cleanly.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p worker   # or nothing for a fake model
//! ```

use std::sync::Arc;
use std::time::Duration;

use flow_agent::prelude::*;
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

const MAX_CONCURRENT_JOBS: usize = 4;

/// (classifier, writer): one real model when an API key is set, otherwise two fakes.
fn models() -> Result<(ChatModel, ChatModel)> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let m = ChatModel::builder("anthropic:claude-sonnet-5-5")
            .api_key(key)
            .build()?;
        return Ok((m.clone(), m));
    }
    tracing::warn!("no API key set; using fake models");
    let classifier = FakeChatModel::new([Message::assistant("billing")]).cycle();
    let writer = FakeChatModel::new([Message::assistant(
        "Thanks for reaching out! We're looking into it.",
    )])
    .cycle();
    Ok((classifier.into(), writer.into()))
}

/// classify -> draft, with a retry policy and timeout on the model nodes.
fn ticket_graph(classify_model: ChatModel, model: ChatModel) -> Result<CompiledGraph> {
    let opts = NodeOptions::new()
        .timeout(Duration::from_secs(60))
        .retry(RetryPolicy::exponential(3));
    StateGraph::new()
        .name("ticket_worker")
        .add_node_with(
            "classify",
            move |state: Value, ctx: NodeContext| {
                let model = classify_model.clone();
                async move {
                    let prompt = format!(
                        "Classify this support ticket as billing, bug or other. Answer with one word.\n\n{}",
                        state["ticket"]
                    );
                    let reply = model.generate(vec![Message::user(prompt)], ctx.config()).await?;
                    Ok(json!({ "category": reply.text().trim().to_lowercase() }))
                }
            },
            opts.clone(),
        )
        .add_node_with(
            "draft",
            move |state: Value, ctx: NodeContext| {
                let model = model.clone();
                async move {
                    let prompt = format!(
                        "Write a short, friendly reply to this {} ticket:\n\n{}",
                        state["category"], state["ticket"]
                    );
                    let reply = model.generate(vec![Message::user(prompt)], ctx.config()).await?;
                    Ok(json!({ "reply": reply.text() }))
                }
            },
            opts,
        )
        .add_edge(START, "classify")
        .add_edge("classify", "draft")
        .add_edge("draft", END)
        .compile()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let (classifier, writer) = models()?;
    let graph = ticket_graph(classifier, writer)?;
    let shutdown = CancellationToken::new();
    let (tx, mut rx) = mpsc::channel::<(u32, String)>(100);

    // Producer: in a real worker this would be a queue consumer (SQS, Kafka, Redis...).
    tokio::spawn(async move {
        for (id, ticket) in [
            "I was charged twice this month.",
            "The export button crashes the app.",
            "Can I change my username?",
        ]
        .into_iter()
        .enumerate()
        {
            if tx.send((id as u32 + 1, ticket.to_owned())).await.is_err() {
                break;
            }
        }
    });

    let stop = shutdown.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
        stop.cancel();
    });

    let permits = Arc::new(Semaphore::new(MAX_CONCURRENT_JOBS));
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        let job = tokio::select! {
            job = rx.recv() => job,
            _ = shutdown.cancelled() => None,
        };
        let Some((id, ticket)) = job else { break };
        let Ok(permit) = permits.clone().acquire_owned().await else {
            break;
        };
        let graph = graph.clone();
        let config = RunConfig::new()
            .run_name(format!("ticket-{id}"))
            .metadata("ticket_id", json!(id))
            .cancellation(shutdown.child_token());
        tasks.spawn(async move {
            let _permit = permit;
            match graph
                .invoke_with(json!({ "ticket": ticket }), &config)
                .await
            {
                Ok(out) => {
                    tracing::info!(id, category = %out["category"], reply = %out["reply"], "done")
                }
                Err(e) => tracing::error!(id, error = %e, "job failed"),
            }
        });
    }
    while tasks.join_next().await.is_some() {}
    Ok(())
}
