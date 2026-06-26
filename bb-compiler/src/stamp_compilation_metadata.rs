//! `stamp_compilation_metadata` — the final pass that turns each
//! per-partition `ModelProto` into a complete artifact by writing
//! the compilation passport + slot binding table into
//! `metadata_props`. Install reads these stamps; the typed
//! `BindingSpec` Rust struct never crosses the compile boundary
//! after this pass runs.
//!
//! Per-partition: each partition's root function name is the
//! target identifier. Bindings are stamped against that target.
//! A future multi-partition merge step concatenates the partitions
//! into one ModelProto with each target as a distinct entry in
//! `functions[]`; this pass is partition-local so the merge can
//! happen at any granularity.
//!
//! Slot-id resolution: walk the partition's NodeProtos for the
//! `(REQUIRED_TRAIT_KEY, SLOT_ID_KEY)` pair stamped by the
//! recorder's placeholder DSL methods. For each `BindingSlot`,
//! find the slot_id whose role matches; dep-only slots (no
//! NodeProto references) encode as `-1`.

use bb_ir::keys::{
    binding_key, encode_binding_value, stamp_model_metadata, COMPILED_CURRENT_VERSION,
    COMPILED_KEY, RECV_SLOT_ID_KEY, REQUIRED_TRAIT_KEY, SLOT_ID_KEY,
};
use bb_ir::proto::onnx::{ModelProto, NodeProto, StringStringEntryProto};

use crate::artifact::BindingSpec;

/// Stamp `bb.compiled = "v1"` on the model + one binding entry per
/// slot under the per-target prefix. Each partition's root function
/// is its target; pass each partition's `target_name` alongside its
/// model. The pass mutates `model.metadata_props` in place and is
/// idempotent (re-stamping with the same values is a no-op).
///
/// Also walks every function for `wire.Recv` NodeProtos whose
/// payload output feeds a role NodeProto (carrying [`SLOT_ID_KEY`])
/// and stamps [`RECV_SLOT_ID_KEY`] on the Recv node's
/// `metadata_props`. The install path reads this to populate
/// `GraphSlot::recv_site_to_slot_id` so `decode_typed_fill` can route
/// backend-bound tensor fills through the bound backend's
/// `materialize_from_wire`. Recv nodes whose payload does not flow
/// into a role NodeProto are left unstamped (framework-carrier path).
pub(crate) fn stamp_compilation_metadata(
    model: &mut ModelProto,
    bindings: &BindingSpec,
    target_name: &str,
) {
    // 1) Compilation passport — present on every compiled model.
    stamp_model_metadata(model, COMPILED_KEY, COMPILED_CURRENT_VERSION);

    // 2) Build a (role → slot_id) map by walking the model's
    //    NodeProtos for the recorder's (required_trait, slot_id)
    //    placeholder stamps. The compiler-internal BindingSpec lists
    //    every slot the install path must fill; bindings whose role
    //    doesn't appear in any NodeProto are dep-only and encode -1.
    let role_to_slot_id = collect_role_slot_ids(model);

    // 3) Stamp one binding entry per BindingSlot.
    for slot in &bindings.slots {
        let role_canon = canonical_role(&slot.role);
        let slot_id_or_neg1 = role_to_slot_id
            .iter()
            .find(|(role, _)| canonical_role(role) == role_canon)
            .map(|(_, id)| *id as i64)
            .unwrap_or(-1);
        let key = binding_key(target_name, &slot.slot);
        let value = encode_binding_value(&role_canon, &slot.concrete_type_name, slot_id_or_neg1);
        stamp_model_metadata(model, &key, &value);
    }

    // 4) Stamp `RECV_SLOT_ID_KEY` on each `wire.Recv` whose payload
    //    output flows into a role NodeProto. The install pass reads
    //    this to map the Recv's allocated `NodeSiteId` onto the
    //    role's `slot_id`.
    stamp_recv_slot_ids(model);
}

/// Walk every function in `model.functions`, find `wire.Recv` nodes,
/// and stamp `RECV_SLOT_ID_KEY` on each whose first output flows
/// into a role NodeProto's input. The Recv's first output is its
/// payload site; we look up consumers by name match.
fn stamp_recv_slot_ids(model: &mut ModelProto) {
    for function in &mut model.functions {
        // Build a (payload_name → recv_index) map for every Recv,
        // and a (consumer_input_name → slot_id) map by scanning role
        // NodeProtos.
        let mut recv_indices: Vec<(usize, String)> = Vec::new();
        let mut consumer_slot_ids: Vec<(String, u32)> = Vec::new();
        for (idx, node) in function.node.iter().enumerate() {
            if is_wire_recv(node) {
                if let Some(payload) = node.output.first() {
                    if !payload.is_empty() {
                        recv_indices.push((idx, payload.clone()));
                    }
                }
                continue;
            }
            let Some(slot_id) =
                metadata_value(node, SLOT_ID_KEY).and_then(|v| v.parse::<u32>().ok())
            else {
                continue;
            };
            for input in &node.input {
                if !input.is_empty() {
                    consumer_slot_ids.push((input.clone(), slot_id));
                }
            }
        }
        for (recv_idx, payload_name) in recv_indices {
            // Pick the first downstream consumer's slot_id. If the
            // same payload feeds multiple role NodeProtos with
            // distinct slot_ids, the compiler would have rejected the
            // graph upstream (one Recv site == one destination
            // binding); the first hit is therefore the only hit on
            // any valid input.
            let Some(slot_id) = consumer_slot_ids
                .iter()
                .find(|(name, _)| name == &payload_name)
                .map(|(_, id)| *id)
            else {
                continue;
            };
            stamp_node_metadata(
                &mut function.node[recv_idx],
                RECV_SLOT_ID_KEY,
                &slot_id.to_string(),
            );
        }
    }
}

fn is_wire_recv(node: &NodeProto) -> bool {
    node.domain == "ai.bytesandbrains.wire" && node.op_type == "Recv"
}

fn stamp_node_metadata(node: &mut NodeProto, key: &str, value: &str) {
    if let Some(existing) = node.metadata_props.iter_mut().find(|p| p.key == key) {
        existing.value = value.to_string();
        return;
    }
    node.metadata_props.push(StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    });
}

/// Walk every NodeProto in every function for the recorder's
/// `(REQUIRED_TRAIT_KEY, SLOT_ID_KEY)` pair; return the distinct
/// `(required_trait, slot_id)` pairs seen. Order is deterministic
/// (insertion order over the walk).
fn collect_role_slot_ids(model: &ModelProto) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    for function in &model.functions {
        for node in &function.node {
            let Some(role) = metadata_value(node, REQUIRED_TRAIT_KEY) else {
                continue;
            };
            let Some(slot_id) =
                metadata_value(node, SLOT_ID_KEY).and_then(|v| v.parse::<u32>().ok())
            else {
                continue;
            };
            if !out.iter().any(|(r, id)| r == role && *id == slot_id) {
                out.push((role.to_string(), slot_id));
            }
        }
    }
    out
}

fn metadata_value<'a>(node: &'a NodeProto, key: &str) -> Option<&'a str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

/// `BindingSlot.role` carries the engine-side trait name
/// (`"BackendRuntime"`, `"IndexRuntime"`, …). Both the install path
/// and the runtime `RuntimeResourceRef::dependency` lookups use the
/// canonical Contract role identifier (PascalCase, no `Runtime`
/// suffix). The stamped binding values use the canonical form so
/// install doesn't have to re-normalize per slot.
fn canonical_role(role: &str) -> String {
    role.strip_suffix("Runtime").unwrap_or(role).to_string()
}

/// Test-only helper: stamp a model with a synthetic
/// `BindingSpec` so test fixtures can drive `install()` without
/// running the full compile pipeline. The first FunctionProto's
/// name is used as the `target`. Bindings are `(slot, role,
/// TYPE_NAME)` triples; the role string is the canonical
/// PascalCase identifier (`"Backend"`, `"Index"`, etc.) — the
/// stamp pass canonicalizes either form.
///
/// Drives through the same stamp path the compiler uses, so tests
/// exercise the real encoding.
pub fn stamp_for_test(model: &mut ModelProto, bindings: &[(&str, &str, &str)]) {
    let target = model
        .functions
        .first()
        .map(|f| f.name.clone())
        .unwrap_or_default();
    let mut spec = BindingSpec::new();
    for (slot, role, type_name) in bindings {
        spec.push(slot.to_string(), role.to_string(), type_name.to_string());
    }
    stamp_compilation_metadata(model, &spec, &target);
}

