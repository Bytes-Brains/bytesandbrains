//! `AddressBook::Insert(peer, addr)` - custom op registered via
//! `bb::register_op!` that pushes a multiaddr into the engine's
//! [`bb_runtime::framework::AddressBook`] from the data plane.
//!
//! The op routes through the existing `add_peer` / `register_address`
//! API: a new peer creates an entry with `ref_count = 1`; a known
//! peer dedupe-appends the address without a `ref_count` change.
//! The `(peer, addr)` carriers ride the graph as `TYPE_PEER_ID` and
//! `TYPE_MULTIADDRESS`.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::framework::Address;
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{AddressValue, PeerIdValue};

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.address_book";
/// Op type name.
pub const OP_TYPE: &str = "Insert";

/// Invoke fn - merge the address into the AddressBook.
///
/// New peer → `add_peer` (entry created with `ref_count = 1`).
/// Known peer → `register_address` (idempotent append, no
/// `ref_count` change). Returns `Immediate(vec![])` on success;
/// missing or mistyped inputs surface as `OpError` so the recording
/// surface notices the wiring mistake.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let peer = downcast_peer(inputs)?;
    let addr = downcast_addr(inputs)?;
    let result = if ctx.peers.addresses.lookup(peer).is_some() {
        ctx.peers.addresses.register_address(peer, addr)
    } else {
        ctx.peers.addresses.add_peer(peer, vec![addr])
    };
    match result {
        Ok(()) => Ok(DispatchResult::Immediate(Vec::new())),
        Err(e) => Err(OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "address_book_insert_failed",
            detail: format!("AddressBook::Insert: {e}"),
        }),
    }
}

fn downcast_peer(inputs: &[(&str, &dyn SlotValue)]) -> Result<PeerId, OpError> {
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| *n == "peer")
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_peer",
            detail: "AddressBook::Insert: required input `peer` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<PeerIdValue>()
        .map(|p| p.0)
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_peer_id",
            detail: "AddressBook::Insert: input `peer` is not a PeerId".into(),
        })
}

fn downcast_addr(inputs: &[(&str, &dyn SlotValue)]) -> Result<Address, OpError> {
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| *n == "addr")
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_addr",
            detail: "AddressBook::Insert: required input `addr` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<AddressValue>()
        .map(|a| a.0.clone())
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_multiaddress",
            detail: "AddressBook::Insert: input `addr` is not a Multiaddress".into(),
        })
}


bb_derive::register_op! {
    domain: "ai.bytesandbrains.address_book",
    op_type: "Insert",
    invoke: invoke,
}
