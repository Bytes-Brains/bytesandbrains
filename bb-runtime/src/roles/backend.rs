//! `BackendRuntime` - **framework-internal** role trait for backend
//! implementations.
//!
//! Backend's atomic opset IS `ai.onnx v1`, and the role op_types ARE
//! the atomic ops. Every `ai.onnx::*` node goes straight to the
//! atomic dispatch table.
//!
//! **Authoring API is the Contract trait, not this one.** Concrete
//! backends implement [`crate::contracts::Backend`] (the 30 mandatory
//! primitives + `execute(&GraphProto, …)`); the
//! `#[derive(bb::Backend)]` proc-macro generates the matching
//! `impl BackendRuntime` that bridges into the engine's
//! atomic-dispatch table. Library makers don't write
//! `impl BackendRuntime` by hand.
//!
//! Distinct from [`bb_dsl::placeholders::Backend`] - the placeholder
//! unit struct Module authors embed as a generic placeholder slot.

use crate::atomic::{AtomicOpsetDecl, DispatchResult};
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::BackendMaterializeError;
use crate::slot_value::SlotValue;

/// Role trait for backend implementations. Universal contract per
/// `docs/ROLES.md` §2 with no per-role methods (Backend's role
/// opset is `ai.onnx v1` which IS its atomic opset).
///
/// Backends MUST minimally cover `ai.onnx v1` via `atomic_opset`.
/// They MAY declare additional opsets (e.g. `ai.onnx v17` extensions
/// or custom-domain ops like `mybackend.fused.MatMulAdd`) via
/// `extension_opsets`. `Node::ready()` consults both at
/// build time to verify every NodeProto in the loaded graphs has a
/// covering dispatch entry.
pub trait BackendRuntime: Send + Sync {
    /// Backend-impl-specific error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Atomic-op opset this impl owns at minimum - `ai.onnx v1`.
    fn atomic_opset(&self) -> AtomicOpsetDecl;

    /// Additional opsets this backend supports beyond
    /// `atomic_opset`. Default empty - backends that ship pure
    /// `ai.onnx v1` need not override.
    ///
    /// Examples of valid extensions:
    /// - A newer `ai.onnx` version (`(ai.onnx, 17)`) declaring ops
    ///   absent from v1.
    /// - A custom-domain opset (`(mybackend.fused, 1)`) the backend
    ///   recognizes via its `dispatch_atomic` body.
    fn extension_opsets(&self) -> Vec<AtomicOpsetDecl> {
        Vec::new()
    }

    /// Dispatch a single op or `BackendSubgraph` carrier. For
    /// primitive ops (`Add`, `Mul`, …) each arm builds a one-node
    /// `GraphProto` and calls `Backend::execute`. For the
    /// `BackendSubgraph` op_type, the embedded `GraphProto` body
    /// rides on the carrier NodeProto's `"body"` attribute and the
    /// derive arm calls `Backend::dispatch` so user overrides
    /// (caching, async) reach the engine.
    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error>;

    /// Engine-side bridge for `Backend::materialize_from_wire`. The
    /// derive forwards `(type_hash, bytes)` through the user's
    /// Contract method and re-boxes the typed `Self::Tensor` into a
    /// [`BackendTensorCarrier`] wrapped in `Box<dyn SlotValue>` so
    /// the engine can install it in the slot table without knowing
    /// the backend's concrete tensor type. Returns
    /// [`BackendMaterializeError`] on backend error; the engine
    /// surfaces this as
    /// [`crate::bus::WireReceiveErrorKind::BackendMaterializeFailed`].
    ///
    /// Library makers do not implement this method — `#[derive(bb::Backend)]`
    /// emits the bridge.
    fn materialize_from_wire(
        &self,
        type_hash: u64,
        bytes: Vec<u8>,
    ) -> Result<Box<dyn SlotValue>, BackendMaterializeError>;
}
