//! `AddressBook::InsertMany(peer, addresses)` — custom op registered
//! via `bb::register_op!` that pushes a full address bag into the
//! engine's [`bb_runtime::framework::AddressBook`] from the data
//! plane.
//!
//! Routes through `AddressBook::add_peer` for a new peer (entry
//! created with `ref_count = 1`) and serially through
//! `register_address` for a known peer (dedupe-append, no
//! `ref_count` change — same semantics as `AddressBook::Insert`, just
//! batched). Carriers are `TYPE_PEER_ID` and `TYPE_ADDRESS_VEC`.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::framework::Address;
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{AddressVecValue, PeerIdValue};

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.address_book";
/// Op type name.
pub const OP_TYPE: &str = "InsertMany";

/// Invoke fn — merge the address bag into the AddressBook.
///
/// New peer → `add_peer` with the full vec (entry created with
/// `ref_count = 1`). Known peer → one `register_address` per address
/// (idempotent append, no `ref_count` change). An empty input vec
/// surfaces as `OpError` — graph-level intent to insert zero
/// addresses is a wiring mistake.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let peer = downcast_peer(inputs)?;
    let addresses = downcast_addresses(inputs)?;
    if addresses.is_empty() {
        return Err(OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "address_book_insert_many_empty",
            detail: "AddressBook::InsertMany: input `addresses` is empty".into(),
        });
    }
    let result = if ctx.peers.addresses.lookup(peer).is_some() {
        let mut last = Ok(());
        for addr in addresses {
            last = ctx.peers.addresses.register_address(peer, addr);
            if last.is_err() {
                break;
            }
        }
        last
    } else {
        ctx.peers.addresses.add_peer(peer, addresses)
    };
    match result {
        Ok(()) => Ok(DispatchResult::Immediate(Vec::new())),
        Err(e) => Err(OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "address_book_insert_many_failed",
            detail: format!("AddressBook::InsertMany: {e}"),
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
            detail: "AddressBook::InsertMany: required input `peer` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<PeerIdValue>()
        .map(|p| p.0)
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_peer_id",
            detail: "AddressBook::InsertMany: input `peer` is not a PeerId".into(),
        })
}

fn downcast_addresses(inputs: &[(&str, &dyn SlotValue)]) -> Result<Vec<Address>, OpError> {
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| *n == "addresses")
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_addresses",
            detail: "AddressBook::InsertMany: required input `addresses` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<AddressVecValue>()
        .map(|a| a.0.clone())
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_address_vec",
            detail: "AddressBook::InsertMany: input `addresses` is not an AddressVec".into(),
        })
}


bb_derive::register_op! {
    domain: "ai.bytesandbrains.address_book",
    op_type: "InsertMany",
    invoke: invoke,
}
