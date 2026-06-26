//! `Clock` syscall - reads the scheduler's current time.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TimestampValue;

/// Marker struct.
pub struct ClockOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "Clock";

/// Invoke fn - emits `TimestampValue(now_ns)`.
pub fn invoke(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let now_ns = ctx.time.scheduler.now_ns();
    Ok(DispatchResult::Immediate(vec![(
        "now".to_string(),
        Box::new(TimestampValue(now_ns)),
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
