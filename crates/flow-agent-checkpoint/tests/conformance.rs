//! Every saver passes the shared conformance suite.

use flow_agent_checkpoint::MemorySaver;
use flow_agent_checkpoint::testing::run_conformance;

#[tokio::test]
async fn memory_saver() {
    run_conformance(&MemorySaver::new()).await.unwrap();
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_saver_file_and_memory() {
    use flow_agent_checkpoint::SqliteSaver;

    let path = std::env::temp_dir().join(format!("flow-agent-{}.db", uuid::Uuid::now_v7()));
    let saver = SqliteSaver::connect(&format!("sqlite://{}", path.display()))
        .await
        .unwrap();
    saver.migrate().await.unwrap();
    saver.migrate().await.unwrap(); // idempotent
    run_conformance(&saver).await.unwrap();
    drop(saver);
    let _ = std::fs::remove_file(&path);

    let mem = SqliteSaver::connect("sqlite::memory:").await.unwrap();
    mem.migrate().await.unwrap();
    run_conformance(&mem).await.unwrap();
}

/// Runs when `FLOW_AGENT_TEST_POSTGRES_URL` is set (CI starts a Postgres service).
#[cfg(feature = "postgres")]
#[tokio::test]
async fn postgres_saver() {
    use flow_agent_checkpoint::PostgresSaver;

    let Ok(url) = std::env::var("FLOW_AGENT_TEST_POSTGRES_URL") else {
        eprintln!("skipping: FLOW_AGENT_TEST_POSTGRES_URL not set");
        return;
    };
    let saver = PostgresSaver::connect(&url).await.unwrap();
    saver.migrate().await.unwrap();
    saver.migrate().await.unwrap();
    run_conformance(&saver).await.unwrap();
}
