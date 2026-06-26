//! `ProtocolRuntime` — framework-internal role trait for protocol
//! implementations.
//!
//! Protocols don't share a verb catalog — each one declares its
//! own atomic opset (e.g. `bb-kademlia.Kademlia.atomic v1`) and
//! handles every op type in its `dispatch_atomic`. There are NO
//! fixed role methods.
//!
//! `register_protocol!{}` writes this impl for the user — library
//! authors do not write `ProtocolRuntime` directly.

use crate::atomic::{AtomicOpsetDecl, DispatchResult};
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;

/// Role trait for protocol implementations.
pub trait ProtocolRuntime: Send + Sync {
    /// Protocol-impl-specific error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Atomic-op opset this protocol declares. Both inbound
    /// envelope routing AND user-graph DSL ops register here.
    fn atomic_opset(&self) -> AtomicOpsetDecl;

    /// Single dispatch entry. For inbound envelopes the framework
    /// synthesizes inputs from the wire envelope:
    ///
    /// - `peer_id`:     `Opaque<PeerId>`
    /// - `payload`:     `Opaque<Bytes>` (raw envelope bytes)
    /// - `correlation`: `Opaque<WireCorrelation>`
    ///
    /// For user-graph DSL ops the inputs come from upstream slot
    /// values exactly like any role op.
    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error>;
}
