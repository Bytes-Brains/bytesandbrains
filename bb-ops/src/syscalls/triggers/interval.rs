//! `Interval` syscall - periodic Trigger source.
//!
//! Reads `period_ns` attribute, schedules a `TimerKind::Interval`
//! on `ctx.time.scheduler`. Returns immediate Trigger output; the
//! engine's scheduler maturity drain re-arms the next firing.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::framework::TimerKind;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TimestampValue;

/// Marker struct.
pub struct IntervalOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "Interval";

/// Invoke fn - schedules the next firing and emits the current
/// trigger.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let period_ns = node
        .attribute
        .iter()
        .find(|a| a.name == "period_ns")
        .map(|a| a.i as u64)
        .unwrap_or(1_000_000_000);
    let now = ctx.time.scheduler.now_ns();
    let op_key = ctx.current.op_ref.as_u64();
    ctx.time.scheduler.schedule(
        now.saturating_add(period_ns),
        TimerKind::Interval {
            period_ns,
            key: op_key,
        },
    );
    // Emit the current timestamp; `tick` is declared as `Timestamp`.
    Ok(DispatchResult::Immediate(vec![(
        "tick".to_string(),
        Box::new(TimestampValue(now)),
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
