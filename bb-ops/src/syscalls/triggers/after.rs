//! `After` syscall - delays a Trigger by `delay_ns`.
//!
//! Schedules a `TimerKind::After { key }` on `ctx.time.scheduler` and
//! returns `Async(cmd)` so the engine routes the delayed Trigger
//! through `handle_completion` after maturity.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::framework::TimerKind;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Marker struct.
pub struct AfterOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "After";

/// Invoke fn - schedules a delayed Trigger.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let delay_ns = node
        .attribute
        .iter()
        .find(|a| a.name == "delay_ns")
        .map(|a| a.i as u64)
        .unwrap_or(0);
    let now = ctx.time.scheduler.now_ns();
    let cmd = ctx.allocate_command_id();
    ctx.time.scheduler.schedule(
        now.saturating_add(delay_ns),
        TimerKind::After { key: cmd.as_u64() },
    );
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
