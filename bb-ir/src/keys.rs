//! Single source of truth for metadata-key string constants shared
//! across DSL → Compiler → Runtime. Reference these constants rather
//! than re-typing `"ai.bytesandbrains.*"` literals.

// ── Wire transport classification (compiler → runtime) ─────────────

/// Per-edge classification stamped by `analyze_wire_edges`. Value
/// is [`WIRE_TRANSPORT_DATA`] or [`WIRE_TRANSPORT_TRIGGER_ONLY`].
pub const WIRE_TRANSPORT_KEY: &str = "ai.bytesandbrains.wire_transport";

/// Value of [`WIRE_TRANSPORT_KEY`] for full-payload edges.
pub const WIRE_TRANSPORT_DATA: &str = "data";

/// Value of [`WIRE_TRANSPORT_KEY`] for trigger-only edges.
pub const WIRE_TRANSPORT_TRIGGER_ONLY: &str = "trigger_only";

// ── Destination routing (compiler → runtime envelope) ──────────────

/// Prefix for per-fill multiaddr suffixes stamped on `Send` nodes.
/// Suffixes are keyed `dest_suffix.<i>` for fill index `i`.
pub const DEST_SUFFIX_ATTR_PREFIX: &str = "ai.bytesandbrains.dest_suffix.";

/// Prefix for per-fill destination site names stamped on `Send`
/// nodes. Used by the lower-network-io pass to derive the partition
/// name on the receive side.
pub const DEST_SITE_NAME_PREFIX: &str = "ai.bytesandbrains.dest_site_name.";

// ── DSL→Compiler binding identity ──────────────────────────────────

/// Concrete-component type. Paired with [`INSTANCE_KEY`].
pub const CONCRETE_TYPE_KEY: &str = "ai.bytesandbrains.concrete_type";

/// Per-instance disambiguator paired with [`CONCRETE_TYPE_KEY`].
pub const INSTANCE_KEY: &str = "ai.bytesandbrains.instance";

/// Required-trait identifier for a generic slot. Paired with
/// [`SLOT_ID_KEY`].
pub const REQUIRED_TRAIT_KEY: &str = "ai.bytesandbrains.required_trait";

/// Per-slot disambiguator paired with [`REQUIRED_TRAIT_KEY`].
pub const SLOT_ID_KEY: &str = "ai.bytesandbrains.slot_id";

/// Stamped on `wire.Recv` whose payload feeds a role NodeProto's
/// `slot_id` input. Drives `decode_typed_fill`'s backend-mediated
/// branch. Absent on framework-carrier Recv nodes.
pub const RECV_SLOT_ID_KEY: &str = "ai.bytesandbrains.recv_slot_id";

// ── Class tagging ──────────────────────────────────────────────────

/// Default-class tag used by `infer_peer_classes` when a peer
/// expression has no explicit class denotation.
pub const DEFAULT_CLASS: &str = "ai.bytesandbrains.default_class";

// ── Module phase (DSL → install → engine) ──────────────────────────

/// Distinguishes a Module's body recording from its bootstrap.
/// Drives `Engine::bootstrap_function_key` seeding on first poll.
pub const MODULE_PHASE_KEY: &str = "ai.bytesandbrains.module_phase";

/// Body-phase value of [`MODULE_PHASE_KEY`].
pub const MODULE_PHASE_BODY: &str = "body";

/// Bootstrap-phase value of [`MODULE_PHASE_KEY`]. Fired once on
/// first poll; body parks until descendants drain.
pub const MODULE_PHASE_BOOTSTRAP: &str = "bootstrap";

// ── Backend subgraph ───────────────────────────────────────────────

/// `op_type` for a carrier NodeProto wrapping a `GraphProto` body.
/// Engine forwards these to `Backend::dispatch` for native-graph
/// specialization.
pub const BACKEND_SUBGRAPH_OP: &str = "BackendSubgraph";

/// Attribute key for the embedded `GraphProto` body inside a
/// `BackendSubgraph` carrier. Stamped by `collapse_backend_subgraphs`.
pub const BACKEND_SUBGRAPH_BODY_ATTR: &str = "body";

// ── Dedup ──────────────────────────────────────────────────────────

/// Attribute key carrying the dedup slot identifier on a
/// `DedupGateRx` op. Together with the inbound peer this seeds the
/// `(peer, slot)` dedup key.
pub const DEDUP_SLOT: &str = "ai.bytesandbrains.dedup_slot";

// ── Wire pairing (DSL Graph::wire → compiler/runtime) ──────────────

/// Per-edge wire-pairing token minted by `Graph::wire` and read by
/// the compiler's `discover_wire_edges` pass.
pub const WIRE_ID_KEY: &str = "ai.bytesandbrains.wire.wire_id";

// ── Wire batching (compiler → runtime) ─────────────────────────────

/// Per-edge batch grouping id stamped by `analyze_wire_edges`.
pub const BATCH_GROUP_KEY: &str = "ai.bytesandbrains.batch_group_id";

/// Trigger type denotation; matches `<Trigger as WireType>::DENOTATION`.
pub const TRIGGER_DENOTATION: &str = "bb.trigger";

// ── Wire chain (compiler → runtime deadline derivation) ────────────

/// Static chain depth from this Send. Runtime multiplies by
/// `per_hop_budget_ns` to size deadlines against the full round-trip.
pub const CHAIN_DEPTH_KEY: &str = "ai.bytesandbrains.wire.chain_depth";

/// Comma-separated targets participating in the chain (originating
/// target excluded). Used for per-hop budget renegotiation.
pub const CHAIN_TARGETS_KEY: &str = "ai.bytesandbrains.wire.chain_targets";

// ── ATTR_PEER (re-export for ergonomic single-import) ──────────────

/// Re-export of [`crate::syscall_ids::ATTR_PEER`] so a single
/// `bb_ir::keys::*` import covers every wire-node key.
pub use crate::syscall_ids::ATTR_PEER;

/// Re-export of [`crate::version::FRAMEWORK_IR_VERSION_KEY`] for
/// the same reason.
pub use crate::version::FRAMEWORK_IR_VERSION_KEY;

// ── Generic-slot dependency metadata (compiler → runtime) ──────────

/// Prefix for per-dependency entries: `dep.<role> = "<slot>"`.
/// Stamped by `resolve_component_dependencies`.
pub const DEP_SLOT_KEY_PREFIX: &str = "ai.bytesandbrains.dep.";

/// Build the metadata key for a dependency on the canonical role
/// string (PascalCase - e.g. `"Backend"`, `"Index"`).
pub fn dep_slot_key(role: &str) -> String {
    format!("{DEP_SLOT_KEY_PREFIX}{role}")
}

/// If `key` is a `DEP_SLOT_KEY_PREFIX`-namespaced dependency entry,
/// return the bare role string. Otherwise `None`.
pub fn role_from_dep_slot_key(key: &str) -> Option<&str> {
    key.strip_prefix(DEP_SLOT_KEY_PREFIX)
}

// ── Constructor helpers ────────────────────────────────────────────

use crate::proto::onnx::{attribute_proto, AttributeProto};

/// Per-input `dest_suffix.<name>` AttributeProto carrying the
/// resolved destination multiaddr suffix.
pub fn dest_suffix_attribute(input_name: &str, address_bytes: Vec<u8>) -> AttributeProto {
    AttributeProto {
        name: format!("{DEST_SUFFIX_ATTR_PREFIX}{input_name}"),
        r#type: attribute_proto::AttributeType::String as i32,
        s: address_bytes,
        ..Default::default()
    }
}

// ── Dependency-metadata stamping + reading helpers ─────────────────

use crate::component::DependencyDecl;
use crate::proto::onnx::{NodeProto, StringStringEntryProto};

/// Stamp `deps` onto `node.metadata_props` as
/// `dep.<role> = "<slot>"`. Idempotent on duplicates.
pub fn stamp_dependency_metadata(node: &mut NodeProto, deps: &[DependencyDecl]) {
    for dep in deps {
        let key = dep_slot_key(dep.role);
        let already = node
            .metadata_props
            .iter()
            .any(|e| e.key == key && e.value == dep.slot);
        if already {
            continue;
        }
        node.metadata_props.push(StringStringEntryProto {
            key,
            value: dep.slot.to_string(),
        });
    }
}

/// Read `dep.<role> = "<slot>"` entries as borrowed `(role, slot)`.
pub fn read_dependency_metadata(node: &NodeProto) -> impl Iterator<Item = (&str, &str)> + '_ {
    node.metadata_props.iter().filter_map(|entry| {
        let role = role_from_dep_slot_key(&entry.key)?;
        Some((role, entry.value.as_str()))
    })
}

// ── Compilation passport + slot binding (compiler → install) ───────

use crate::proto::onnx::ModelProto;

/// Compilation passport key. Missing → `InstallError::NotCompiled`;
/// mismatched value → `InstallError::IncompatibleCompiledVersion`.
pub const COMPILED_KEY: &str = "ai.bytesandbrains.compiled";

/// Current compilation passport value. Bumps on IR-breaking changes.
pub const COMPILED_CURRENT_VERSION: &str = "v1";

/// Prefix for per-target per-slot binding entries:
/// `binding.<target>.<slot> = "<role>|<TYPE_NAME>|<slot_id|-1>"`.
pub const BINDING_KEY_PREFIX: &str = "ai.bytesandbrains.binding.";

/// Build the binding key for `(target, slot)`. Splits at the FIRST
/// dot after the prefix when parsing — slot names may contain dots.
pub fn binding_key(target: &str, slot: &str) -> String {
    format!("{BINDING_KEY_PREFIX}{target}.{slot}")
}

/// Parse a binding key into `(target, slot)`. `None` for non-binding.
pub fn parse_binding_key(key: &str) -> Option<(&str, &str)> {
    let rest = key.strip_prefix(BINDING_KEY_PREFIX)?;
    let (target, slot) = rest.split_once('.')?;
    Some((target, slot))
}

/// Encode a binding's `(role, TYPE_NAME, slot_id_or_neg1)` triple
/// into the pipe-delimited value `"<role>|<TYPE_NAME>|<slot_id_or_-1>"`.
/// Reads inversely via [`parse_binding_value`].
pub fn encode_binding_value(role: &str, type_name: &str, slot_id_or_neg1: i64) -> String {
    format!("{role}|{type_name}|{slot_id_or_neg1}")
}

/// Decompose a binding value into `(role, TYPE_NAME, slot_id_or_-1)`.
/// `None` when the format doesn't match (the caller surfaces this as
/// `InstallError::InvalidBindingTable`).
pub fn parse_binding_value(value: &str) -> Option<(&str, &str, i64)> {
    let mut parts = value.splitn(3, '|');
    let role = parts.next()?;
    let type_name = parts.next()?;
    let slot_id: i64 = parts.next()?.parse().ok()?;
    Some((role, type_name, slot_id))
}

/// Stamp `(key, value)` onto `model.metadata_props`, replacing any
/// existing entry with the same key. Idempotent on re-runs.
pub fn stamp_model_metadata(model: &mut ModelProto, key: &str, value: &str) {
    if let Some(existing) = model.metadata_props.iter_mut().find(|e| e.key == key) {
        existing.value = value.to_string();
        return;
    }
    model.metadata_props.push(StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    });
}

/// Read a `ModelProto.metadata_props` entry by key. Returns the
/// borrowed value string; `None` when the key isn't present.
pub fn read_model_metadata<'a>(model: &'a ModelProto, key: &str) -> Option<&'a str> {
    model
        .metadata_props
        .iter()
        .find(|e| e.key == key)
        .map(|e| e.value.as_str())
}

// ── FunctionProto metadata helpers ─────────────────────────────────

use crate::proto::onnx::FunctionProto;

/// Read `MODULE_PHASE_KEY` off a FunctionProto. `None` when the
/// key is absent.
pub fn read_function_module_phase(function: &FunctionProto) -> Option<&str> {
    function
        .metadata_props
        .iter()
        .find(|e| e.key == MODULE_PHASE_KEY)
        .map(|e| e.value.as_str())
}

/// Stamp `MODULE_PHASE_KEY` onto a FunctionProto. Replaces existing.
pub fn stamp_function_module_phase(function: &mut FunctionProto, phase: &str) {
    if let Some(existing) = function
        .metadata_props
        .iter_mut()
        .find(|e| e.key == MODULE_PHASE_KEY)
    {
        existing.value = phase.to_string();
        return;
    }
    function.metadata_props.push(StringStringEntryProto {
        key: MODULE_PHASE_KEY.to_string(),
        value: phase.to_string(),
    });
}

