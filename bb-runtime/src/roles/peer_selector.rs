//! `PeerSelectorRuntime` - role trait for peer-sampling implementations.
//!
//! Per `docs/ROLES.md` §10. The trait carries the universal pair
//! (`atomic_opset` + `dispatch_atomic`); the engine routes through
//! `dispatch_atomic`. Author Contract impls
//! (`crate::contracts::PeerSelector`) define the user-facing surface;
//! `#[derive(bb::PeerSelector)]` emits the bridge into
//! `PeerSelectorRuntime::dispatch_atomic`.

use crate::atomic::{AtomicOpsetDecl, DispatchResult};
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;

/// Role trait for peer-sampling implementations.
pub trait PeerSelectorRuntime: Send + Sync {
    /// Peer-sampling-impl-specific error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Atomic-op opset this impl owns.
    fn atomic_opset(&self) -> AtomicOpsetDecl;

    /// Rust-dispatch entry point for atomic ops.
    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error>;
}
