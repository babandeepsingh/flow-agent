//! An HTTP API around a flow-agent ReAct agent.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p axum-api   # or OPENAI_API_KEY, or nothing for a fake model
//!
//! curl -s localhost:3000/chat -H 'content-type: application/json' \
//!   -d '{"thread_id": "u1", "message": "What time is it in Tokyo?"}'
//! curl -N localhost:3000/chat/stream -H 'content-type: application/json' \
//!   -d '{"thread_id": "u1", "message": "And in Paris?"}'
//! curl -s localhost:3000/threads/u1
//! ```
//!
//! Conversations are stored in `agent.db` (SQLite), so threads survive restarts.

use std::convert::Infallible;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use flow_agent::prelude::*;
use futures::{Stream, StreamExt};
use serde::Deserialize;

#[derive(Clone)]
struct AppState {
    agent: CompiledGraph,
}

#[derive(Deserialize)]
struct ChatRequest {
    thread_id: String,
    message: String,
}

#[derive(Deserialize)]
struct ResumeRequest {
    #[serde(default)]
    value: Value,
}

/// Maps flow-agent errors to JSON HTTP errors.
struct ApiError(Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0.root() {
            Error::InvalidInput(_) | Error::Config(_) => StatusCode::BAD_REQUEST,
            Error::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({ "error": self.0.to_value() }))).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        ApiError(e)
    }
}

fn config(thread_id: &str) -> RunConfig {
    RunConfig::new()
        .thread_id(thread_id)
        .timeout(std::time::Duration::from_secs(120))
}

async fn chat(
    State(app): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<Value>, ApiError> {
    let out = app
        .agent
        .run(json!(req.message), &config(&req.thread_id))
        .await?;
    Ok(Json(json!({
        "reply": last_text(&out.state),
        "interrupts": out.interrupts,
        "checkpoint_id": out.checkpoint_id,
    })))
}

async fn chat_stream(
    State(app): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let events = app
        .agent
        .stream_owned(json!(req.message), &config(&req.thread_id))
        .map(|item| {
            let data = match item {
                Ok(event) => event.to_value(),
                Err(e) => json!({ "type": "error", "error": e.to_value() }),
            };
            Ok(SseEvent::default().data(data.to_string()))
        });
    Sse::new(events).keep_alive(KeepAlive::default())
}

async fn resume(
    State(app): State<AppState>,
    Path(thread_id): Path<String>,
    Json(req): Json<ResumeRequest>,
) -> Result<Json<Value>, ApiError> {
    let out = app.agent.resume(req.value, &config(&thread_id)).await?;
    Ok(Json(json!({ "reply": last_text(&out), "state": out })))
}

async fn get_thread(
    State(app): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Response, ApiError> {
    Ok(match app.agent.get_state(&config(&thread_id)).await? {
        Some(snapshot) => Json(snapshot).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    })
}

/// A real model when an API key is set, otherwise a scripted fake.
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
    tracing::warn!("no API key set; using a fake model");
    Ok(FakeChatModel::new([
        Message::assistant("").with_tool_call("current_time", json!({ "city": "Tokyo" })),
        Message::assistant("It's 09:00 in Tokyo (demo answer from the fake model)."),
    ])
    .cycle()
    .into())
}

fn time_tool() -> Tool {
    Tool::new("current_time", "Current local time in a city")
        .param("city", "string", "City name")
        .handler(|args| async move {
            // A real implementation would look up the city's time zone.
            Ok(json!({ "city": args["city"], "time": "09:00" }))
        })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,flow_agent=debug".into()),
        )
        .init();

    let saver = SqliteSaver::connect("sqlite://agent.db").await?;
    saver.migrate().await?;

    let agent = ReactAgent::builder(model()?)
        .tool(time_tool())
        .system_prompt("You are a helpful assistant. Use tools when useful.")
        .checkpointer(Arc::new(saver))
        .build()?;

    let app = Router::new()
        .route("/chat", post(chat))
        .route("/chat/stream", post(chat_stream))
        .route("/threads/{id}", get(get_thread))
        .route("/threads/{id}/resume", post(resume))
        .with_state(AppState { agent });

    let addr = "127.0.0.1:3000";
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Error::Config(format!("cannot bind {addr}: {e}")))?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| Error::Config(format!("server error: {e}")))
}
