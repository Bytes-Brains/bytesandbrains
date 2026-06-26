//! `DataSourceRuntime` - role trait for data-source implementations.
//!
//! Per `docs/ROLES.md` §9. The trait carries the universal pair
//! (`atomic_opset` + `dispatch_atomic`); the engine routes through
//! `dispatch_atomic`. Author Contract impls
//! (`crate::contracts::DataSource`) define the user-facing surface;
//! `#[derive(bb::DataSource)]` emits the bridge into
//! `DataSourceRuntime::dispatch_atomic`.

use crate::atomic::{AtomicOpsetDecl, DispatchResult};
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;

/// Role trait for data-source implementations.
pub trait DataSourceRuntime: Send + Sync {
    /// Data-source-impl-specific error type.
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
