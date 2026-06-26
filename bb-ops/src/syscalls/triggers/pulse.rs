//! `Pulse` syscall - one-shot bootstrap trigger.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct.
pub struct PulseOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "Pulse";

/// Invoke fn - emits one Trigger.
pub fn invoke(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    Ok(DispatchResult::Immediate(vec![(
        "trigger".to_string(),
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
