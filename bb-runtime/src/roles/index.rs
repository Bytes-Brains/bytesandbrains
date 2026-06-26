//! `IndexRuntime` - role trait for vector-index implementations.
//!
//! Per `docs/ROLES.md` §6. The trait carries the universal pair
//! (`atomic_opset` + `dispatch_atomic`); the engine routes through
//! `dispatch_atomic` and never invokes Shape-1 role methods at
//! runtime. Author Contract impls (`crate::contracts::Index`)
//! define the user-facing surface; `#[derive(bb::Index)]` emits the
//! bridge into `IndexRuntime::dispatch_atomic`.
//!
//! The opset declares four ops: `Add`, `Search`, `Remove`, `Train`.
//! `Train` carries the optional calibration pass; impls that skip
//! training keep the default `Contract::train` no-op and the derive
//! routes the op through `dispatch_atomic` like any other arm.

use crate::atomic::{AtomicOpsetDecl, DispatchResult};
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;

/// Role trait for vector-index implementations.
pub trait IndexRuntime: Send + Sync {
    /// Index-impl-specific error type.
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
