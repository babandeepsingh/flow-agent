//! Turning an `invoke` call into a stream of events.

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedReceiver};

use super::{BoxRunnable, Runnable};
use crate::config::RunConfig;
use crate::error::Result;
use crate::event::Event;

/// A stream of run events. Successful streams end with [`Event::Output`].
pub type EventStream<'a> = BoxStream<'a, Result<Event>>;

/// Default `stream_with`: invoke while forwarding nested events.
pub(crate) fn stream_invoke<'a, R: Runnable + ?Sized>(
    runnable: &'a R,
    input: Value,
    config: &RunConfig,
) -> EventStream<'a> {
    let (tx, rx) = mpsc::unbounded_channel();
    let config = config.with_sink_target(tx);
    drive(
        Box::pin(async move { runnable.invoke_with(input, &config).await }),
        rx,
    )
}

/// Like [`Runnable::stream_with`] but owns the runnable, so the stream is
/// `'static` and can be returned from an HTTP handler (e.g. as SSE).
pub fn stream_owned(
    runnable: BoxRunnable,
    input: Value,
    config: &RunConfig,
) -> EventStream<'static> {
    let (tx, rx) = mpsc::unbounded_channel();
    let config = config.with_sink_target(tx);
    drive(
        Box::pin(async move { runnable.invoke_with(input, &config).await }),
        rx,
    )
}

enum State<'a> {
    Running(BoxFuture<'a, Result<Value>>, UnboundedReceiver<Event>),
    Draining(UnboundedReceiver<Event>, Result<Value>),
    Done,
}

/// Run `fut` to completion, yielding events from `rx` as they arrive, then
/// any events still buffered, then the final output (or error).
pub(crate) fn drive<'a>(
    fut: BoxFuture<'a, Result<Value>>,
    rx: UnboundedReceiver<Event>,
) -> EventStream<'a> {
    futures::stream::unfold(State::Running(fut, rx), |state| async move {
        let mut state = state;
        loop {
            match state {
                State::Running(mut fut, mut rx) => {
                    tokio::select! {
                        biased;
                        Some(ev) = rx.recv() => return Some((Ok(ev), State::Running(fut, rx))),
                        res = &mut fut => state = State::Draining(rx, res),
                    }
                }
                State::Draining(mut rx, res) => {
                    if let Ok(ev) = rx.try_recv() {
                        return Some((Ok(ev), State::Draining(rx, res)));
                    }
                    let item = res.map(|value| Event::Output { value });
                    return Some((item, State::Done));
                }
                State::Done => return None,
            }
        }
    })
    .boxed()
}
