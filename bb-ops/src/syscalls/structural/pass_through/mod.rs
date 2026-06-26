//! `PassThrough` - the framework's structural identity syscall.
//!
//! syscall component in bb-ops. The (domain, op_type) constants
//! come from `bb_ir::syscall_ids`; the DSL recording helper, the
//! runtime dispatch entry, and the `inventory::submit!`
//! self-registration all live in one file alongside the sibling
//! tests.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::registry::OpRegistration;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

// --- IR identity --------------------------------------------------

pub use bb_ir::syscall_ids::OP_PASS_THROUGH as OP_TYPE;
/// `(domain, op_type)` key. Re-exported from the foundation so
/// every reference cites one declaration.
pub use bb_ir::syscall_ids::SYSCALL_DOMAIN as DOMAIN;

/// Engine dispatch-table marker - its sole purpose is to provide a
/// unique `TypeId` for the syscall registry.
pub struct PassThroughOp;

// --- Runtime dispatch ---------------------------------------------

/// Invoke fn - forwards the input value via polymorphic
/// `SlotValue::clone_boxed`. The concrete type survives the hop;
/// downstream consumers downcast to the type the graph contract
/// guarantees at the consumer site.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let Some((_, input)) = inputs.first() else {
        return Err(OpError {
            detail: "PassThrough requires one input".to_string(),
            ..Default::default()
        });
    };
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        input.clone_boxed(),
    )]))
}

// --- Inventory self-registration ----------------------------------

inventory::submit! {
    OpRegistration {
        domain: DOMAIN,
        op_type: OP_TYPE,
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

