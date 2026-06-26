//! `AddressBook::Lookup(peer) → addresses` — custom op registered via
//! `bb::register_op!` that resolves every multiaddr bound to a peer
//! from the engine's [`bb_runtime::framework::AddressBook`].
//!
//! Returns the full ordered slice; downstream Components that need a
//! single address pick one explicitly. Unknown or empty-address-list
//! peers surface as `OpError` so the recording surface notices the
//! missing seed.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{AddressVecValue, PeerIdValue};

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.address_book";
/// Op type name.
pub const OP_TYPE: &str = "Lookup";
/// Output port carrying the resolved `Vec<Address>`.
pub const PORT_ADDRESSES: &str = "addresses";

/// Invoke fn — read every address for `peer` and emit them on the
/// `addresses` output. Missing peer or empty list → `OpError` with
/// `ExecutionFailed`; missing or mistyped `peer` input → `OpError`
/// with the matching `MissingSlot` / `TypeMismatch` kind.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let peer = downcast_peer(inputs)?;
    let addrs = ctx
        .peers
        .addresses
        .lookup(peer)
        .map(|s| s.to_vec())
        .ok_or_else(|| OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "address_book_lookup_miss",
            detail: format!("AddressBook::Lookup: peer {peer} not in address book"),
        })?;
    Ok(DispatchResult::Immediate(vec![(
        PORT_ADDRESSES.to_string(),
        Box::new(AddressVecValue(addrs)) as Box<dyn SlotValue>,
    )]))
}

fn downcast_peer(inputs: &[(&str, &dyn SlotValue)]) -> Result<PeerId, OpError> {
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| *n == "peer")
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_peer",
            detail: "AddressBook::Lookup: required input `peer` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<PeerIdValue>()
        .map(|p| p.0)
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_peer_id",
            detail: "AddressBook::Lookup: input `peer` is not a PeerId".into(),
        })
}


bb_derive::register_op! {
    domain: "ai.bytesandbrains.address_book",
    op_type: "Lookup",
    invoke: invoke,
}
