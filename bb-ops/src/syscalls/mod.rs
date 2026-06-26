//! Framework syscall components - each sub-directory hosts one or
//! more registerable ops. Self-registration is via
//! `inventory::submit!`; `Engine::register_all_framework_syscalls`
//! walks the registry at Node::ensure_ready.

pub mod clock_rng;
pub mod composite;
pub mod coordination;
pub mod gates;
pub mod lifecycle;
pub mod peers;
pub mod structural;
pub mod sync;
pub mod telemetry;
pub mod triggers;

use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::BytesValue;

/// Read the first input as `BytesValue`. Distinguishes "no input
/// present" (returns `Ok(None)` — legitimate Trigger-only firing)
/// from "input present but not `BytesValue`" (returns `Err` with
/// `OpErrorKind::TypeMismatch` — adversarial / mis-wired graph).
///
/// Use for syscalls that accept an optional payload alongside a
/// trigger: `Hold.Stash`, `Serialize.Enqueue`, `Record`, `AppEmit`.
pub(crate) fn first_input_optional_bytes(
    op_name: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<Option<Vec<u8>>, OpError> {
    let Some((slot_name, value)) = inputs.first() else {
        return Ok(None);
    };
    let Some(bytes) = value.as_any().downcast_ref::<BytesValue>() else {
        return Err(OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_bytes",
            detail: format!("{op_name}: input '{slot_name}' is not BytesValue"),
        });
    };
    Ok(Some(bytes.0.clone()))
}
