//! `DedupGateRx` syscall op - gate inserted by the compiler pass
//! `bb-compiler/src/insert_dedup_gate_rx.rs` at the head of the RX
//! gate chain. Consults [`bb_runtime::framework::InboundDedup`] and
//! drops envelopes whose payload hash has been seen recently.
//!
//! The hash is computed from the input value's wire bytes (via
//! `SlotValue::to_wire_bytes`) so that the same payload from the
//! same peer collapses to a single delivery.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Marker struct for `register_syscall::<DedupGateRxOp>`.
pub struct DedupGateRxOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "DedupGateRx";

/// Invoke fn - hashes the input's wire bytes, records the hash in
/// `InboundDedup`, and forwards the input on first-arrival. Drops
/// with a `duplicate` reason on a repeat.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let (_, value) = inputs.first().ok_or_else(|| OpError {
        detail: "DedupGateRx requires one input".into(),
        ..Default::default()
    })?;

    let bytes = value.to_wire_bytes().map_err(|e| OpError {
        kind: bb_runtime::bus::OpErrorKind::ExecutionFailed,
        reason: "wire_encode_failed",
        detail: format!("DedupGateRx: input wire encode failed: {e}"),
    })?;
    let hash = fnv1a_64(&bytes);
    if ctx.net.dedup.record(hash) {
        return Err(OpError {
            detail: "DedupGateRx dropped envelope: reason=duplicate".into(),
            ..Default::default()
        });
    }

    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        value.clone_boxed(),
    )]))
}

/// FNV-1a 64-bit hash. Mirrors the wire-type hash in
/// `bb_ir::wire::compute_wire_hash` so dedup behavior stays
/// predictable across versions.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
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
