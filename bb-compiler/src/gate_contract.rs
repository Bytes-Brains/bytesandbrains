//! Open `GateContract` inventory consumed by
//! [`crate::validate_runtime_complete`].
//!
//! Each gate-insertion pass declares a `GateContract` impl + emits
//! `inventory::submit! { GateContractRegistration { ... } }`; the
//! validator iterates the inventory and runs every registered
//! contract's `assert_inserted` against the post-insertion graph.
//! Adding a new gate is "ship the inserting pass + register its
//! contract" — no edit to the validator.

use bb_ir::proto::onnx::GraphProto;
use bb_ir::registry::inventory;

use crate::error::CompileError;

/// A single insertion contract a gate-inserting pass owns.
///
/// `assert_inserted` runs on every per-role sub-graph after the
/// gate-insertion phase; it returns `Ok(())` when the gate this
/// contract represents is consistent with the graph the validator
/// observes (either absent because nothing needs it, OR present in
/// the canonical insertion shape).
pub trait GateContract: Send + Sync {
    /// Diagnostic label for the contract (e.g. `"DeadlineCheck"`).
    /// Surfaced in `CompileError::RuntimeIncomplete` so the host
    /// sees which contract failed.
    fn name(&self) -> &'static str;

    /// Assert the contract holds on `sub_graph`. Returns
    /// `Err(CompileError::RuntimeIncomplete { ... })` with a
    /// human-readable description when the gate's insertion is
    /// incomplete.
    fn assert_inserted(&self, sub_graph: &GraphProto) -> Result<(), CompileError>;
}

/// Inventory-collected pointer to a `GateContract` implementation
/// shipped by a gate-insertion pass. Library makers introducing a
/// new gate emit one inventory submission alongside their pass.
pub struct GateContractRegistration {
    /// Static pointer to the contract impl.
    pub contract: &'static dyn GateContract,
}

inventory::collect!(GateContractRegistration);

/// Iterate every `GateContract` the binary links in.
pub fn contracts() -> impl Iterator<Item = &'static GateContractRegistration> {
    inventory::iter::<GateContractRegistration>.into_iter()
}
