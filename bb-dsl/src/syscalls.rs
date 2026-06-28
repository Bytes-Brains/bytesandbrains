//! DSL-side syscall helpers — record canonical `NodeProto`s into a
//! `Graph`. The runtime-side dispatch impls live in `bb-ops`; the
//! two sides agree on stable `(domain, op_type)` string constants
//! re-exported from `bb_ir::syscall_ids`.

use crate::graph::{attr_tensor, kv, Graph};
use crate::output::Output;
use bb_ir::proto::onnx::{tensor_proto, NodeProto, TensorProto};

/// Canonical `(domain, op_type)` string constants the DSL helpers
/// stamp onto recorded `NodeProto`s. The strings live in
/// `bb_ir::syscall_ids` so the DSL + compiler + runtime cite one
/// declaration; this module re-exports them under shorter local
/// aliases for the helper bodies below.
pub mod ids {
    pub use bb_ir::syscall_ids::{
        OP_CONSTANT as CONSTANT_OP, OP_GATE_DISPATCH as GATE_DISPATCH_OP,
        OP_PASS_THROUGH as PASS_THROUGH_OP, SYSCALL_DOMAIN,
    };
}

/// `(domain, op_type)` registration key for the `ai.bytesandbrains.address_book`
/// custom-op family. Matches the strings registered in
/// `bb-ops/src/syscalls/peers/{insert,insert_many,lookup}.rs` via
/// `register_op!`.
const ADDRESS_BOOK_DOMAIN: &str = "ai.bytesandbrains.address_book";
const INSERT_MANY_OP: &str = "InsertMany";
const LOOKUP_OP: &str = "Lookup";

/// `(domain, op_type)` registration key for the `GlobalRegistryClient`
/// `Announce` op. Mirrors the literal string registered in
/// `bb-ops/src/protocols/global_registry/mod.rs` (the constant
/// `GLOBAL_REGISTRY_DOMAIN`); duplicated here as a string literal
/// because `bb-dsl` cannot depend on `bb-ops` (`bb-ops` re-exports
/// `bb-dsl`'s authoring surface).
const GLOBAL_REGISTRY_DOMAIN: &str = "ai.bytesandbrains.protocol.global_registry";
const ANNOUNCE_OP: &str = "Announce";

/// `metadata_props` key for the bootstrap-seed label stamped on
/// `constant` NodeProtos for diagnostics.
const BOOTSTRAP_SEED_KEY: &str = "ai.bytesandbrains.bootstrap.seed";

/// `metadata_props` key prefix for input-name remapping on the
/// `Announce` NodeProto. The runtime's `ProtocolRuntime` dispatch
/// reads these to recover the logical input role (here:
/// `server_peer`).
const ANNOUNCE_SERVER_PEER_KEY: &str = "ai.bytesandbrains.input.server_peer";

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

/// Record a typed `Constant` syscall NodeProto. Bootstrap-stage
/// constants seed the AddressBook + GlobalRegistry ops with the
/// server's PeerId and dial bag at install time. The compiler's
/// `expand_constant` pass requires `value: TensorProto`; this
/// helper satisfies that contract with an empty tensor sized for
/// the declared scalar kind.
///
/// `label` rides on `metadata_props` under
/// `ai.bytesandbrains.bootstrap.seed` for diagnostics. `output_type`
/// is the recorded `&'static TypeNode` consumers downcast on
/// (`TYPE_PEER_ID`, `TYPE_ADDRESS_VEC`, …).
pub fn constant(
    g: &mut Graph,
    label: &'static str,
    output_type: &'static bb_ir::types::TypeNode,
    data_type: tensor_proto::DataType,
) -> Output {
    let out_name = g.next_site_name();
    let tensor = TensorProto {
        data_type: data_type as i32,
        dims: vec![1],
        ..Default::default()
    };
    g.push_node(NodeProto {
        op_type: ids::CONSTANT_OP.into(),
        domain: ids::SYSCALL_DOMAIN.into(),
        input: vec![],
        output: vec![out_name.clone()],
        attribute: vec![attr_tensor("value", tensor)],
        metadata_props: vec![kv(BOOTSTRAP_SEED_KEY, label)],
        ..Default::default()
    });
    g.declare_value_info(&out_name, output_type);
    Output::new(out_name, output_type)
}

/// Record a `GlobalRegistryClient::Announce` NodeProto. The client
/// reads `ctx.local_addresses()` automatically, throttles
/// sub-interval calls to the server's last advertised heartbeat
/// interval, and merges the server's address bag from the Handshake
/// reply.
///
/// `server_peer` is the `PeerId` `Output` the announcing client
/// ships its envelope toward (typically a `Constant` of the server's
/// stable id, recorded with `constant`). The output is typed as
/// `TYPE_TRIGGER` — downstream nodes can chain on the wakeup.
pub fn announce(g: &mut Graph, server_peer: Output) -> Output {
    let out_name = g.next_site_name();
    g.push_node(NodeProto {
        op_type: ANNOUNCE_OP.into(),
        domain: GLOBAL_REGISTRY_DOMAIN.into(),
        input: vec![server_peer.name.clone()],
        output: vec![out_name.clone()],
        metadata_props: vec![kv(ANNOUNCE_SERVER_PEER_KEY, &server_peer.name)],
        ..Default::default()
    });
    g.declare_value_info(&out_name, &bb_ir::types::TYPE_TRIGGER);
    Output::new(out_name, &bb_ir::types::TYPE_TRIGGER)
}
