//! `CompletionHandle` for async Contract methods. Implementations
//! call [`CompletionHandle::complete`]; the handle routes through a
//! [`CompletionSink`] (typically `IngressQueue`), which the engine
//! drains on its next poll to unpark the suspended op.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ids::CommandId;

/// Cross-thread completion delivery. Sinks take borrowed slices and
/// must copy into framework-owned storage before returning (Principle
/// 1a: external byte payloads are ephemeral).
pub trait CompletionSink: Send + Sync {
    /// Deliver a successful completion. Implementation copies bytes.
    fn complete(&self, cmd_id: CommandId, result_bytes: &[u8]);
    /// Deliver a failure with `Display` rendering of the error.
    fn fail(&self, cmd_id: CommandId, detail: &str);
}

/// Async handle the Contract method holds. Carries the [`CommandId`]
/// + shared completion sink.
pub struct CompletionHandle<R, E> {
    cmd_id: CommandId,
    sink: Arc<dyn CompletionSink>,
    _marker: PhantomData<fn() -> (R, E)>,
}

impl<R, E> CompletionHandle<R, E> {
    /// Construct a fresh handle.
    pub fn new(cmd_id: CommandId, sink: Arc<dyn CompletionSink>) -> Self {
        Self {
            cmd_id,
            sink,
            _marker: PhantomData,
        }
    }

    /// The parked op's `CommandId`. Read by the dispatch arm before
    /// returning `DispatchResult::Async(cmd_id)`.
    pub fn cmd_id(&self) -> CommandId {
        self.cmd_id
    }
}

impl<R, E> CompletionHandle<R, E>
where
    R: serde::Serialize,
    E: std::fmt::Display,
{
    /// Complete the parked op. `Ok(value)` serializes via bincode;
    /// `Err(e)` delivers the `Display` rendering. Local buffers drop
    /// at end of call (sink copies).
    pub fn complete(self, result: Result<R, E>) {
        match result {
            Ok(value) => {
                let bytes = bincode::serialize(&value).unwrap_or_default();
                self.sink.complete(self.cmd_id, &bytes);
            }
            Err(e) => {
                let detail = e.to_string();
                self.sink.fail(self.cmd_id, &detail);
            }
        }
    }
}

/// Contract-layer mirror of [`crate::atomic::DispatchResult`].
/// `Now(Ok)` → `Immediate` (boxed straight into the slot table);
/// `Now(Err)` → dispatch error; `Later` → `Async(cmd_id)`.
pub enum ContractResponse<R, E> {
    /// Result ready inline; the `CompletionHandle` is unused.
    Now(Result<R, E>),
    /// Handle retained for off-thread completion.
    Later,
}
