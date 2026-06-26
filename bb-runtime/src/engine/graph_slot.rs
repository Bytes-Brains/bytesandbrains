//! `GraphSlot` - per-FunctionProto compiled data per
//! `docs/ENGINE.md` §3.
//!
//! Under the runtime-linker model each entry in
//! `Node.model.functions[]` becomes one `GraphSlot` in
//! `Engine.graphs`, keyed by the function's
//! `(domain, name, overload)`. The map IS the symbol table - a call
//! NodeProto whose `(domain, op_type, overload)` matches a key
//! resolves to the target installed graph.
//!
//! Each NodeProto's dispatch kind is pre-stamped at install time
//! into `op_dispatch[i]` (parallel to `function.node[i]`). Runtime
//! invoke is one indexed probe; no HashMap lookups on hot path.

use std::collections::HashMap;

use crate::engine::dispatch_entry::OpDispatch;
use crate::ids::{NodeSiteId, OpRef};
use bb_ir::proto::onnx::FunctionProto;

/// Per-FunctionProto compiled data installed on the `Engine`. Keyed
/// in `Engine.graphs` by `(domain, name, overload)` - the canonical
/// symbol-table key the linker dedupes on. The map key IS the name,
/// so `GraphSlot` carries no separate `name` field.
pub struct GraphSlot {
    /// The post-analysis FunctionProto body. Source of truth for
    /// op_type / domain / input + output names. `function.node[i]`
    /// is the NodeProto for OpRef `pack(graph_idx, i)`.
    pub function: FunctionProto,

    /// Per-NodeProto dispatch decision, indexed parallel to
    /// `function.node[]`. Populated by `Engine::resolve_dispatch`
    /// after install; runtime invoke is `op_dispatch[node_idx]`
    /// where `node_idx = op_ref.split().1`.
    pub op_dispatch: Vec<OpDispatch>,

    /// Producer site → downstream consumer ops. Used by
    /// `write_outputs` to push ready consumers onto the frontier.
    pub consumers: HashMap<NodeSiteId, Vec<OpRef>>,

    /// Site name → `NodeSiteId` allocation.
    pub site_names: HashMap<String, NodeSiteId>,

    /// Top-level `function.output` sites, mapped from `NodeSiteId` to
    /// the declared output name. When a value lands at one of these
    /// sites and `consumers[site]` is empty, the engine surfaces it
    /// as `EngineStep::AppEvent { module_name, topic: <output name> }`
    /// — the "function signature is the engine I/O contract" path.
    /// Populated only for entry-point GraphSlots (`is_entry_point`).
    pub top_level_outputs: HashMap<NodeSiteId, String>,

    /// `wire.Recv` payload site → sender site pairing. Recv NodeProtos
    /// emit two outputs — `(payload, sender)`. The inbound envelope
    /// delivers its byte payload to the payload site; the engine
    /// also writes `PeerIdValue(src_peer)` to the sender site on the
    /// same execution so downstream user ops can read the
    /// provenance of the received message and reply by sending back
    /// to the same peer. Populated at install time by scanning
    /// every `ai.bytesandbrains.wire` Recv NodeProto with two
    /// outputs.
    pub recv_sender_sites: HashMap<NodeSiteId, NodeSiteId>,

    /// `wire.Recv` payload site → expected wire-type hash. When the
    /// compiler stamped `ValueInfoProto.type_node` on a Recv's
    /// payload output, the runtime carries the producer-side
    /// `type_hash` here so the typed-receive path can fire a
    /// `WireReceiveError { kind: TypeMismatch }` when an inbound
    /// fill's `type_hash` does not match the slot contract.
    /// Slots without an entry are treated as dynamic / Any: the
    /// decoder-registry lookup proceeds without the mismatch
    /// check. Populated at install time alongside
    /// [`Self::recv_sender_sites`].
    pub recv_wire_type_hash: HashMap<NodeSiteId, u64>,

    /// `wire.Recv` payload site → bound role slot id. Built at
    /// install time by reading the compiler-stamped
    /// `RECV_SLOT_ID_KEY` off each `wire.Recv` NodeProto and pairing
    /// it with the Recv's allocated `NodeSiteId`. Consumed by
    /// `decode_typed_fill` to cross from data-plane identity
    /// (`NodeSiteId`) to binding identity (`slot_id`) before
    /// dispatching the backend-mediated tensor path. Recv sites whose
    /// payload does not flow into a role-bound slot are absent.
    pub recv_site_to_slot_id: HashMap<NodeSiteId, u32>,

    /// `true` if this function is an entry point (a registered
    /// Module's main partition function). Entry-point GraphSlots
    /// have `top_level_outputs` populated; sub-function bodies do not
    /// (their outputs flow through `CallContext.output_forwarding` at
    /// call time).
    pub is_entry_point: bool,
}

impl GraphSlot {
    /// Test-only constructor. Builds a `GraphSlot` from the
    /// supplied FunctionProto with empty op_index / consumers /
    /// site_names tables. Production code uses
    /// [`Self::from_function`].
    #[cfg(any(test, feature = "test-components"))]
    pub fn new_for_test(_name: String, function: FunctionProto) -> Self {
        Self {
            function,
            op_dispatch: Vec::new(),
            consumers: HashMap::new(),
            site_names: HashMap::new(),
            top_level_outputs: HashMap::new(),
            recv_sender_sites: HashMap::new(),
            recv_wire_type_hash: HashMap::new(),
            recv_site_to_slot_id: HashMap::new(),
            is_entry_point: false,
        }
    }

    /// Canonical install path: walks the FunctionProto's
    /// nodes assigning positional `OpRef`s + fresh `NodeSiteId`s,
    /// populates `op_index` + `site_names` + `consumers` per
    /// `docs/ENGINE.md` §3.
    ///
    /// `OpRef`s pack as `(graph_idx << 32) | node_idx` so the engine
    /// hot path resolves `OpRef → NodeProto` via two array accesses
    /// (the `op_index` HashMap is retained only as a     /// migration aide; C8 deletes it). `graph_idx` is the engine's
    /// number of already-installed graphs at the moment of install,
    /// passed in by the caller (typically `Engine::install_graph` /
    /// `install_function_library`). `NodeSiteId`s remain globally
    /// monotonic via `next_node_site_id` since they cross graphs.
    pub fn from_function(
        _name: String,
        function: FunctionProto,
        graph_idx: u32,
        next_node_site_id: &mut u64,
    ) -> Self {
        let mut site_names: HashMap<String, NodeSiteId> = HashMap::new();
        let mut consumers: HashMap<NodeSiteId, Vec<OpRef>> = HashMap::new();

        // First pass: mint NodeSiteIds for every produced value name +
        // pre-fill op_dispatch with Unresolved sentinels (one slot per
        // NodeProto in install order). OpRefs are positional:
        // `OpRef::pack(graph_idx, node_idx)` directly indexes
        // `function.node[]`.
        let mut op_refs: Vec<OpRef> = Vec::with_capacity(function.node.len());
        let mut op_dispatch: Vec<OpDispatch> = Vec::with_capacity(function.node.len());
        for (idx, node) in function.node.iter().enumerate() {
            let op_ref = OpRef::pack(graph_idx, idx as u32);
            op_refs.push(op_ref);
            op_dispatch.push(OpDispatch::Unresolved);
            for out in &node.output {
                if out.is_empty() {
                    continue;
                }
                site_names.entry(out.clone()).or_insert_with(|| {
                    let r = NodeSiteId::from(*next_node_site_id);
                    *next_node_site_id = next_node_site_id.saturating_add(1);
                    r
                });
            }
        }

        // Second pass: populate `consumers` - for each non-empty
        // input on each node, record the consuming op_ref under the
        // producer's NodeSiteId.
        for (idx, node) in function.node.iter().enumerate() {
            let consumer = op_refs[idx];
            for input in &node.input {
                if input.is_empty() {
                    continue;
                }
                let Some(&site) = site_names.get(input) else {
                    continue;
                };
                consumers.entry(site).or_default().push(consumer);
            }
        }

        // Resolve top-level output sites for the AppEvent surfacing
        // path. Each entry in `function.output` is a declared output
        // port; if it maps to a registered NodeSiteId, that site is
        // a candidate for `EngineStep::AppEvent` when no downstream
        // consumer reads it.
        let mut top_level_outputs: HashMap<NodeSiteId, String> = HashMap::new();
        for name in &function.output {
            if let Some(&site) = site_names.get(name) {
                top_level_outputs.insert(site, name.clone());
            }
        }

        // Pair each wire.Recv's payload site with its sender site so
        // inbound envelope delivery can populate both at the same
        // ExecId. Recv NodeProtos are emitted by the DSL
        // `Graph::wire` with `output: [payload, sender]`.
        //
        // Also read the compiler-stamped `RECV_SLOT_ID_KEY` off each
        // Recv node and pair the payload site's `NodeSiteId` with
        // the downstream role's `slot_id`. The map is consumed by
        // `decode_typed_fill` to route backend-mediated tensor fills
        // through the bound backend instance.
        let mut recv_sender_sites: HashMap<NodeSiteId, NodeSiteId> = HashMap::new();
        let mut recv_site_to_slot_id: HashMap<NodeSiteId, u32> = HashMap::new();
        for node in &function.node {
            if node.domain != "ai.bytesandbrains.wire" || node.op_type != "Recv" {
                continue;
            }
            let payload = node.output.first().and_then(|n| site_names.get(n));
            let sender = node.output.get(1).and_then(|n| site_names.get(n));
            if let (Some(&p), Some(&s)) = (payload, sender) {
                recv_sender_sites.insert(p, s);
            }
            if let Some(&payload_site) = payload {
                let slot_id = node
                    .metadata_props
                    .iter()
                    .find(|kv| kv.key == bb_ir::keys::RECV_SLOT_ID_KEY)
                    .and_then(|kv| kv.value.parse::<u32>().ok());
                if let Some(slot_id) = slot_id {
                    recv_site_to_slot_id.insert(payload_site, slot_id);
                }
            }
        }

        // `recv_wire_type_hash` stays empty at install time. The
        // compiler does not yet stamp `ValueInfoProto.type_node` on
        // Recv payload outputs with a stable hash that matches the
        // producer-side `SlotValue::type_hash()` derivation (FNV-1a
        // over the concrete type name); the `TypeNode.wire_hash`
        // field is a parallel handle-assigned identifier that does
        // not align with the runtime `type_hash`. Until the
        // compiler-stamp follow-up lands (per
        // `docs/internal/superpowers/specs/2026-06-24-wire-recv-typed-receive-and-bundle-bench.md`
        // §7), the TypeMismatch check is dormant in production:
        // entries can be inserted by tests + future install passes
        // without changing the typed-receive happy path.
        let recv_wire_type_hash: HashMap<NodeSiteId, u64> = HashMap::new();

        let _ = op_refs;
        Self {
            function,
            op_dispatch,
            consumers,
            site_names,
            top_level_outputs,
            recv_sender_sites,
            recv_wire_type_hash,
            recv_site_to_slot_id,
            is_entry_point: true,
        }
    }
}

