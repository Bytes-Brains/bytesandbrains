//! `Sleep` syscall - schedules a `TimerKind::Sleep(cmd_id)` and
//! returns `Async(cmd_id)`. Sub-F.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::framework::TimerKind;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Marker struct.
pub struct SleepOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "Sleep";

/// Invoke fn - schedules a sleep timer for `duration_ns` and
/// returns the matching `CommandId` as async-suspended.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let duration_ns = node
        .attribute
        .iter()
        .find(|a| a.name == "duration_ns")
        .map(|a| a.i as u64)
        .unwrap_or(0);
    let now = ctx.time.scheduler.now_ns();
    // mints a fresh CommandId via the global counter. Stage
    // 8 may route through Node's ExecId allocator if needed.
    let cmd = ctx.allocate_command_id();
    ctx.time
        .scheduler
        .schedule(now.saturating_add(duration_ns), TimerKind::Sleep(cmd));
    Ok(DispatchResult::Async(cmd))
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
