//! `bb::Index` — Contract trait for vector indexes.
//!
//! Each method takes the engine's `&mut RuntimeResourceRef<'_>` ctx
//! plus a [`CompletionHandle`] AND returns [`ContractResponse`]. The
//! impl declares per call:
//!
//! - [`ContractResponse::Now(Ok(value))`] — result is ready inline.
//!   The handle is ignored; the framework returns
//!   `DispatchResult::Immediate(vec![(port, Box::new(value) as Box<dyn SlotValue>)])`
//!   — `value` lands in the slot table as `Box<dyn SlotValue>` with no
//!   serialization at this boundary — and skips the park /
//!   ingress-drain cycle.
//! - [`ContractResponse::Later`] — the impl retained the handle (sent
//!   it to a worker thread, spawned a tokio task, queued a remote
//!   RPC). The framework returns `DispatchResult::Async(cmd_id)` and
//!   parks the dispatched op until the user calls
//!   `handle.complete(result)` from off-thread.
//!
//! `ctx` is the per-dispatch runtime surface: impls reach their
//! declared `#[depends(...)]` siblings through
//! [`RuntimeResourceRef::dependency`] (e.g. an index that delegates
//! distance math to a bound `Backend`).

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;

/// User-facing Contract trait for a vector index.
pub trait Index: Send + Sync {
    /// Vector storage. Library makers pick the tree position by
    /// picking the type: `[f32]` for an f32-native index,
    /// `bb_ir::types::AnyTensor` for a generic algorithm-class index
    /// that outsources distance math to a bound `Backend`, a custom
    /// packed type for specialized dtypes.
    type Vector: ?Sized + bb_ir::types::Storage;

    /// Library-maker-defined error type. The handle serializes the
    /// error's `Display` rendering on completion failure.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Insert a vector. Return `Now(Ok(id))` if the assignment is
    /// inline, or retain `completion` and return `Later` to deliver
    /// the id off-thread. `ctx` exposes the per-dispatch runtime
    /// surface, including the
    /// [`RuntimeResourceRef::dependency`] lookup for declared
    /// `#[depends(...)]` siblings.
    fn add(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        vec: &Self::Vector,
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error>;

    /// Top-`k` nearest-neighbor search. Return inline results via
    /// `Now(Ok(pairs))` or retain `completion` for off-thread delivery.
    /// `ctx` carries the runtime surface so the search can resolve
    /// `#[depends(...)]` siblings (e.g. a bound `Backend` supplying
    /// distance kernels).
    fn search(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        query: &Self::Vector,
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error>;

    /// Remove a vector by id. Return `Now(Ok(()))` if the removal is
    /// synchronous, or retain `completion` and return `Later`.
    fn remove(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// Optional training pass — IVF needs centroid k-means, PQ needs
    /// sub-vector codebook learning, flat / hand-tuned indexes skip
    /// this. Default returns `Now(Ok(()))` so impls that do not train
    /// pay zero cost.
    fn train(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _samples: &[&Self::Vector],
        _completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }
}

