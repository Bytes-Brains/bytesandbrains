//! `RngU64` syscall - emits a random u64 from the framework RNG.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::U64Value;

/// Marker struct.
pub struct RngU64Op;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "RngU64";

/// Invoke fn - emits one u64 from `ctx.syscall.rng`.
pub fn invoke(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let value = ctx.syscall.rng.next_u64();
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        Box::new(U64Value(value)),
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
