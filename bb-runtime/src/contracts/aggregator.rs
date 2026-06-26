//! `bb::Aggregator` — Contract trait for federated aggregators.
//!
//! Each method takes a [`CompletionHandle`] AND returns
//! [`ContractResponse`]. See [`crate::contracts::index`] for the
//! sync (Now) vs async (Later) semantics.
//!
//! ## Shape
//!
//! Aggregation is a two-op cycle: `contribute(...)` writes one
//! peer's update into an in-progress buffer; `aggregate(...)`
//! reduces the buffer into the current aggregate AND returns it.
//! There is no separate `current_tensor()` op — `aggregate` is the
//! one-stop "compute + emit" call.
//!
//! ## Metadata channel
//!
//! Both `contribute` and `aggregate` carry a **typed** metadata
//! payload alongside the tensor, defined by the impl as the
//! associated type [`Aggregator::Metadata`].
//!
//! The metadata is transported through the slot table as a typed
//! Rust value — the framework's slot-value layer (`bb_ir::slot_value`)
//! holds every value as `Box<dyn SlotValue>` and downcasts to the
//! concrete type via `Any::downcast_ref`. Bincode/serde fires only
//! at the wire boundary (`SlotValue::to_wire_bytes`) and at
//! snapshot time. In-process contribute/aggregate calls see the
//! typed value directly — no serde overhead.
//!
//! This is the channel hierarchical aggregation needs: a child
//! `FedAvg` aggregator's `aggregate(...)` emits
//! `(params, FedAvgMeta { num_samples })`; the parent layer's
//! `contribute(...)` receives that and the `num_samples` weights
//! the child's contribution in the parent reduction. Both halves
//! work with the typed `FedAvgMeta` — only the wire crossing does
//! serde.
//!
//! Impls that have no metadata channel set `type Metadata = ();`.

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;
use bb_ir::ids::PeerId;

/// User-facing Contract trait for a federated/decentralized
/// aggregator. The derive bridges these methods to the engine's
/// [`crate::roles::AggregatorRuntime`] trait.
pub trait Aggregator: Send + Sync {
    /// Storage element type for the tensors this aggregator
    /// operates on. Most f32-native aggregators declare
    /// `type Element = [f32]`.
    ///
    /// The bound `?Sized + bb_ir::types::Storage` allows unsized
    /// slice types like `[f32]` (a `Box<[f32]>` is the owned form
    /// returned from `aggregate`).
    type Element: ?Sized + bb_ir::types::Storage;

    /// Library-maker-defined error type.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Impl-defined metadata that travels alongside the tensor.
    /// Carried as a typed slot value; serde fires only when the
    /// value crosses a wire boundary.
    ///
    /// For FedAvg: `type Metadata = FedAvgMeta { num_samples: u64 };`.
    /// For impls with no metadata channel: `type Metadata = ();`.
    type Metadata: Clone
        + Default
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
        + Send
        + Sync
        + 'static;

    /// Contribute one peer's update to the in-progress aggregation.
    /// `ctx` is the per-dispatch runtime surface; impls reach their
    /// declared `#[depends(...)]` siblings through
    /// [`RuntimeResourceRef::dependency`]. `tensor` is a reference
    /// to the element (e.g. `&[f32]` for `Element = [f32]`).
    /// `metadata` is the typed accompanying data (sample counts for
    /// FedAvg, weights for weighted sum, round ids, …).
    /// Default-constructed `Metadata` is valid for impls that don't
    /// have a real metadata channel.
    fn contribute(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        src: PeerId,
        tensor: &Self::Element,
        metadata: Self::Metadata,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// Reduce the accumulated contributions and return the result.
    /// `ctx` carries the runtime surface so the aggregator's
    /// reduction can resolve `#[depends(...)]` siblings (e.g. the
    /// `Backend` that supplies the composed weighted-sum).
    /// Output is `(params, metadata)`:
    /// - `params`: the aggregated tensor, owned as
    ///   `Box<Self::Element>` (e.g. `Box<[f32]>`). Same allocator
    ///   footprint as a `Vec<f32>` — use `vec.into_boxed_slice()`.
    /// - `metadata`: typed accompanying data describing the
    ///   aggregation (e.g. summed `num_samples` for hierarchical
    ///   FedAvg).
    ///
    /// The output edge fires only when the reduction completes;
    /// downstream consumers wire directly to the `(params,
    /// metadata)` outputs — no separate read op needed.
    #[allow(clippy::type_complexity)]
    fn aggregate(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<(Box<Self::Element>, Self::Metadata), Self::Error>,
    ) -> ContractResponse<(Box<Self::Element>, Self::Metadata), Self::Error>;
}
