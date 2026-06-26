//! DSL-side syscall helpers — record canonical `NodeProto`s into a
//! `Graph`. The runtime-side dispatch impls live in `bb-ops`; the
//! two sides agree on stable `(domain, op_type)` string constants
//! re-exported from `bb_ir::syscall_ids`.

use crate::graph::Graph;
use crate::output::Output;
use bb_ir::proto::onnx::NodeProto;

/// Canonical `(domain, op_type)` string constants the DSL helpers
/// stamp onto recorded `NodeProto`s. The strings live in
/// `bb_ir::syscall_ids` so the DSL + compiler + runtime cite one
/// declaration; this module re-exports them under shorter local
/// aliases for the helper bodies below.
pub mod ids {
    pub use bb_ir::syscall_ids::{
        OP_GATE_DISPATCH as GATE_DISPATCH_OP, OP_PASS_THROUGH as PASS_THROUGH_OP, SYSCALL_DOMAIN,
    };
}

/// `(domain, op_type)` registration key for the `ai.bytesandbrains.address_book`
/// custom-op family. Matches the strings registered in
/// `bb-ops/src/syscalls/peers/{insert,insert_many,lookup}.rs` via
/// `register_op!`.
const ADDRESS_BOOK_DOMAIN: &str = "ai.bytesandbrains.address_book";
const INSERT_MANY_OP: &str = "InsertMany";
const LOOKUP_OP: &str = "Lookup";

/// Record a `PassThrough` syscall NodeProto into a `Graph`.
///
/// The framework's structural identity op - threads a value through
/// a partition without doing any compute. Authors reach for it when
/// a partition needs a non-wire node (e.g. a receiver class that
/// only forwards values it receives over the wire). The recorded
/// NodeProto's home class is inferred by the compiler from the
/// input's home.
pub fn pass_through(g: &mut Graph, input: Output) -> Output {
    let out_name = g.next_site_name();
    g.push_node(NodeProto {
        op_type: ids::PASS_THROUGH_OP.into(),
        domain: ids::SYSCALL_DOMAIN.into(),
        input: vec![input.name],
        output: vec![out_name.clone()],
        ..Default::default()
    });
    g.declare_value_info(&out_name, input.type_node);
    Output::new(out_name, input.type_node)
}

/// Record an `AddressBook::InsertMany(peer, addresses)` custom-op
/// NodeProto into a `Graph`. New peer creates an entry with
/// `ref_count = 1`; known peer dedupe-appends every address without
/// touching `ref_count`. Empty `addresses` vec surfaces as a
/// dispatch-time `OpError`.
pub fn address_book_insert_many(g: &mut Graph, peer: Output, addresses: Output) -> Output {
    let out_name = g.next_site_name();
    g.push_node(NodeProto {
        op_type: INSERT_MANY_OP.into(),
        domain: ADDRESS_BOOK_DOMAIN.into(),
        input: vec![peer.name, addresses.name],
        output: vec![out_name.clone()],
        ..Default::default()
    });
    g.declare_value_info(&out_name, &bb_ir::types::TYPE_TRIGGER);
    Output::new(out_name, &bb_ir::types::TYPE_TRIGGER)
}

/// Record an `AddressBook::Lookup(peer)` custom-op NodeProto into a
/// `Graph`. Output carries the full ordered `TYPE_ADDRESS_VEC`;
/// callers that need a single address pick one downstream. Unknown
/// or empty-address peer surfaces as a dispatch-time `OpError`.
pub fn address_book_lookup(g: &mut Graph, peer: Output) -> Output {
    let out_name = g.next_site_name();
    g.push_node(NodeProto {
        op_type: LOOKUP_OP.into(),
        domain: ADDRESS_BOOK_DOMAIN.into(),
        input: vec![peer.name],
        output: vec![out_name.clone()],
        ..Default::default()
    });
    g.declare_value_info(&out_name, &bb_ir::types::TYPE_ADDRESS_VEC);
    Output::new(out_name, &bb_ir::types::TYPE_ADDRESS_VEC)
}

/// Record a `GateDispatch` syscall NodeProto into a `Graph` - a
/// multi-edge synchronization barrier.
pub fn gate_dispatch(g: &mut Graph, inputs: &[Output]) -> Output {
    let out_name = g.next_site_name();
    g.push_node(NodeProto {
        op_type: ids::GATE_DISPATCH_OP.into(),
        domain: ids::SYSCALL_DOMAIN.into(),
        input: inputs.iter().map(|o| o.name.clone()).collect(),
        output: vec![out_name.clone()],
        ..Default::default()
    });
    g.declare_value_info(&out_name, &bb_ir::types::TYPE_BYTES);
    Output::new(out_name, &bb_ir::types::TYPE_BYTES)
}
