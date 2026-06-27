//! `refine_polymorphic_value_info` — narrows the placeholder
//! `TYPE_TENSOR` denotation stamped by the DSL recorder on every
//! Contract-method NodeProto to each bound concrete's actual
//! `Storage::TYPE`.
//!
//! Runs BEFORE `run_pipeline` (and therefore BEFORE `type_solver`)
//! so the solver walks the narrowed denotations, not the placeholder
//! ones. Because this pass needs access to `BindingSpec` it lives in
//! `Compiler::compile()` alongside `validate_all_slots_bound`, not
//! inside the canonical runner pipeline (which has no binding
//! context). `validate_all_slots_bound` runs AFTER the pipeline to
//! confirm every bound slot was used.
//!
//! Pass order: `refine_polymorphic_value_info` → `run_pipeline`
//! (containing `type_solver`) → `validate_all_slots_bound`.
//!

use std::collections::HashMap;

use crate::artifact::BindingSpec;
use crate::error::CompileError;
use bb_ir::proto::onnx::{ModelProto, NodeProto};
use bb_ir::syscall_ids::{OP_PASS_THROUGH, OP_WIRE_SEND, SYSCALL_DOMAIN, WIRE_DOMAIN};
use bb_ir::types::TypeNode;

/// Walk every Contract-method NodeProto in every `FunctionProto` and
/// refine its `value_info` denotation from the polymorphic
/// `TYPE_TENSOR` placeholder to the bound concrete's actual
/// `Storage::TYPE`.
///
/// Only nodes carrying both `ai.bytesandbrains.required_trait` **and**
/// `ai.bytesandbrains.slot_id` metadata are considered Contract-method
/// nodes. Nodes without these metadata entries are silently skipped.
///
/// If the resolved `BindingSlot` has empty `storage_types` (e.g. a
/// hand-implemented concrete that didn't use `#[derive(bb::<Role>)]`),
/// the denotation is left unchanged — no error is returned. This is the
/// documented graceful-degradation path.
pub(crate) fn refine_polymorphic_value_info(
    model: &mut ModelProto,
    spec: &BindingSpec,
) -> Result<(), CompileError> {
    // Collect (output_value_name → narrowed TypeNode) pairs in a
    // first pass over functions, then apply them in a second pass
    // to avoid borrowing `model` mutably while reading it.
    let mut refinements: Vec<(String, &'static TypeNode)> = Vec::new();

    for function in model.functions.iter() {
        for node in function.node.iter() {
            // Nodes without required_trait and without slot_id are
            // not Contract-method nodes — silently skip them.
            let has_required_trait =
                metadata_value(node, "ai.bytesandbrains.required_trait").is_some();
            let has_slot_id = metadata_value(node, "ai.bytesandbrains.slot_id").is_some();

            if !has_required_trait && !has_slot_id {
                continue;
            }

            // A node with slot_id but no required_trait is malformed IR —
            // the DSL recorder always stamps both together.
            if has_slot_id && !has_required_trait {
                return Err(CompileError::MissingRequiredTraitMetadata {
                    node: node.name.clone(),
                });
            }

            // At this point has_required_trait is true.
            let required_trait =
                metadata_value(node, "ai.bytesandbrains.required_trait").expect("checked above");

            // Guard: two or more slots sharing a role means
            // lookup_by_role would silently pick the first and apply
            // the wrong concrete's storage type to nodes belonging to
            // the other slot. Surface this as a hard error when the
            // model actually contains a node referencing the
            // ambiguous role. (Models with no Contract nodes for a
            // given role are unaffected even if the spec has
            // duplicate role bindings.)
            {
                let matching: Vec<&str> = spec
                    .slots
                    .iter()
                    .filter(|s| s.role == required_trait)
                    .map(|s| s.slot.as_str())
                    .collect();
                if matching.len() > 1 {
                    return Err(CompileError::AmbiguousRoleBinding {
                        role: required_trait.to_string(),
                        slot_names: matching.iter().map(|s| s.to_string()).collect(),
                    });
                }
            }

            // slot_id must be present and parseable on a Contract-method
            // node; if it isn't, surface an error — the DSL recorder
            // always stamps it.
            let slot_id_str =
                metadata_value(node, "ai.bytesandbrains.slot_id").ok_or_else(|| {
                    CompileError::InvalidSlotId {
                        node: node.name.clone(),
                        value: String::new(),
                    }
                })?;
            // `_slot_id` is validated for malformed-IR detection;
            // lookup uses `lookup_by_role` until slot_id-keyed
            // BindingSpec lookup is added.
            let _slot_id: u32 = slot_id_str
                .parse()
                .map_err(|_| CompileError::InvalidSlotId {
                    node: node.name.clone(),
                    value: slot_id_str.to_string(),
                })?;

            // Look up the binding slot by its role (the `required_trait`
            // string is the role-runtime identifier).
            let slot =
                spec.lookup_by_role(required_trait)
                    .ok_or_else(|| CompileError::UnknownSlotId {
                        node: node.name.clone(),
                        slot_id: _slot_id,
                    })?;

            // Map required_trait → port label; CodecRuntime also needs
            // the per-node `codec.port` metadata. `None` means this
            // role carries no storage-typed port (e.g. PeerSelector).
            let Some(port) = port_name_for_trait(required_trait, node)? else {
                continue;
            };

            // Resolve the storage TypeNode. An empty `storage_types`
            // (graceful degradation) returns `None`; we skip quietly.
            let Some(narrowed) = slot.storage_type_opt(port) else {
                continue;
            };

            for output in &node.output {
                refinements.push((output.clone(), narrowed));
            }
        }
    }

    propagate_through_value_preserving_ops(model, &mut refinements);

    apply_refinements(model, &refinements);
    Ok(())
}

/// Forward closure walk: `PassThrough` (syscall) and `wire.Send`
/// preserve their input value's type on their first output (the
/// renamed port / re-published value). Once the Contract-method
/// upstream of one of these ops has been refined, the refined type
/// must travel forward so the port name's `value_info` stops
/// resolving as abstract `TYPE_TENSOR`.
///
/// Without this propagation the type solver's strict mode rejects
/// `g.net_out(name, peers, role_method_output)` because the port
/// name's `value_info` keeps the recorder's placeholder denotation.
fn propagate_through_value_preserving_ops(
    model: &ModelProto,
    refinements: &mut Vec<(String, &'static TypeNode)>,
) {
    if refinements.is_empty() {
        return;
    }
    let mut by_name: HashMap<String, &'static TypeNode> =
        refinements.iter().map(|(n, t)| (n.clone(), *t)).collect();

    loop {
        let mut added = false;
        for function in model.functions.iter() {
            for node in function.node.iter() {
                if !is_value_preserving(node) {
                    continue;
                }
                let Some(input_name) = node.input.first() else {
                    continue;
                };
                let Some(&narrowed) = by_name.get(input_name) else {
                    continue;
                };
                let Some(output_name) = node.output.first() else {
                    continue;
                };
                if by_name.contains_key(output_name) {
                    continue;
                }
                by_name.insert(output_name.clone(), narrowed);
                refinements.push((output_name.clone(), narrowed));
                added = true;
            }
        }
        if !added {
            break;
        }
    }
}

/// Whether a NodeProto carries its first input's type to its first
/// output unchanged. Covers the recorder-emitted PassThrough (the
/// idempotent `g.output` re-name) and `wire.Send` (which republishes
/// the value under the port name on the sender partition).
fn is_value_preserving(node: &NodeProto) -> bool {
    matches!(
        (node.domain.as_str(), node.op_type.as_str()),
        (SYSCALL_DOMAIN, OP_PASS_THROUGH) | (WIRE_DOMAIN, OP_WIRE_SEND)
    )
}

/// Map a `required_trait` string to the port label used in
/// `BindingSlot.storage_types`. For `CodecRuntime` the port is
/// read from `ai.bytesandbrains.codec.port` metadata on the node.
///
/// Returns `Ok(None)` for roles that carry no storage-typed port
/// (e.g. `PeerSelectorRuntime`). The caller skips refinement for
/// those nodes.
fn port_name_for_trait(
    required_trait: &str,
    node: &NodeProto,
) -> Result<Option<&'static str>, CompileError> {
    match required_trait {
        "IndexRuntime" => Ok(Some("vector")),
        "AggregatorRuntime" => Ok(Some("element")),
        "ModelRuntime" => Ok(Some("tensor")),
        "DataSourceRuntime" => Ok(Some("sample")),
        "BackendRuntime" => Ok(Some("tensor")),
        "PeerSelectorRuntime" => Ok(None), // peer selectors have no storage-typed port
        "CodecRuntime" => {
            let port_meta =
                metadata_value(node, "ai.bytesandbrains.codec.port").ok_or_else(|| {
                    CompileError::MissingCodecPortMetadata {
                        node: node.name.clone(),
                    }
                })?;
            match port_meta {
                "in" => Ok(Some("in")),
                "out" => Ok(Some("out")),
                other => Err(CompileError::InvalidCodecPort {
                    node: node.name.clone(),
                    value: other.to_string(),
                }),
            }
        }
        _ => Err(CompileError::UnknownRoleRuntime {
            node: node.name.clone(),
            role: required_trait.to_string(),
        }),
    }
}

/// Stamp the collected refinements onto the model's
/// `FunctionProto.value_info` entries in place.
fn apply_refinements(model: &mut ModelProto, refinements: &[(String, &'static TypeNode)]) {
    if refinements.is_empty() {
        return;
    }
    for function in model.functions.iter_mut() {
        for vi in function.value_info.iter_mut() {
            if let Some((_, narrowed)) = refinements.iter().find(|(name, _)| *name == vi.name) {
                if let Some(ref mut t) = vi.r#type {
                    t.denotation = narrowed.denotation.to_string();
                }
            }
        }
    }
}

/// Read a metadata value from a `NodeProto.metadata_props` list.
fn metadata_value<'a>(node: &'a NodeProto, key: &str) -> Option<&'a str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

