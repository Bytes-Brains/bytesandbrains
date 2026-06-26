//! `bb::PeerSelector` — Contract trait for peer-selection protocols.
//!
//! Every method takes `ctx: &mut RuntimeResourceRef<'_>` as the first
//! parameter after `&mut self` (the uniform shape shared by every
//! Contract trait) plus a [`CompletionHandle`] AND returns
//! [`ContractResponse`]. See [`crate::contracts::index`] for the
//! sync (Now) vs async (Later) semantics.
//!
//! Selector impls read `ctx.peers.addresses` to walk the local
//! `AddressBook`, write through it for membership updates from the
//! `dispatch_atomic` arm (`Announce` / `Forget`), and reach the
//! shared `Scheduler` via `ctx.time` when they need to plan a delayed
//! probe. Declared dependencies are reached via
//! `ctx.dependency::<T>(slot)`.

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;
use bb_ir::ids::PeerId;

/// Parameters describing what peer set to select. Different
/// concrete selectors handle the variants they support and return
/// an error variant for the ones they don't (e.g. `GlobalRegistry`
/// supports `Random` + `All`; `DhtView` supports `NearKey`).
///
/// Open enum - new variants are additive; concrete impls match
/// the ones they handle + return an unsupported-params error for
/// the rest.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SelectParams {
    /// Sample `n` peers uniformly at random from the current view.
    Random {
        /// Number of peers requested.
        n: u32,
    },
    /// Sample up to `n` peers whose identifier is closest to `key`
    /// under whatever metric the selector uses (Kademlia XOR, etc.).
    NearKey {
        /// Routing key the selector matches against.
        key: Vec<u8>,
        /// Maximum peers to return.
        n: u32,
    },
    /// Return every peer in the current view. Useful for tiny
    /// fixed-size deployments + tests.
    All,
}

/// User-facing Contract trait for a peer-selection protocol.
pub trait PeerSelector: Send + Sync {
    /// Library-maker-defined error type.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Generic selection — `params` carries selector-specific
    /// config. Concrete impls handle the variants they support
    /// and fail the unsupported ones via `ContractResponse::Now`
    /// carrying an error. `ctx` exposes `ctx.peers.addresses`
    /// (the framework's `AddressBook`), the engine's per-op
    /// runtime surface, and `ctx.dependency::<T>(slot)` for
    /// reaching any concrete bound via `#[depends(...)]`.
    fn select(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        params: SelectParams,
        completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error>;

    /// Sample `n` peers from the current view. Calls
    /// land on `select(SelectParams::Random { n })` by default.
    /// Concrete impls may override to keep an optimized fast path.
    fn sample(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        n: u32,
        completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error> {
        self.select(ctx, SelectParams::Random { n }, completion)
    }

    /// Snapshot the current view of known peers (owned snapshot —
    /// async serialization needs owned values).
    fn current_view(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error>;
}
