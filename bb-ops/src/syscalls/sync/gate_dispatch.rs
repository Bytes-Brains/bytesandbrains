//! `GateDispatch` syscall op - generic multi-input synchronization
//! barrier. Variadic inputs collapse to a single Trigger output that
//! fires once every input has landed.
//!
//! Authors compose downstream ops that need cross-cutting
//! synchronization on multiple independent producers. The engine's
//! frontier already gates op dispatch on all-inputs-ready; this op
//! gives that semantic a name so a single trigger output can fan
//! into multiple downstream consumers.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct for `register_syscall::<GateDispatchOp>`.
pub struct GateDispatchOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "GateDispatch";

/// Invoke fn - emits a `TriggerValue` on the `out` output. The
/// engine's all-inputs-ready check already guaranteed every input
/// arrived before this op fired, so the work here is purely
/// declarative.
pub fn invoke(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    Ok(DispatchResult::Immediate(vec![(
        "out".to_string(),
        Box::new(TriggerValue),
    )]))
}


use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: OP_TYPE,
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
