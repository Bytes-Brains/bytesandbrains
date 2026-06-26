//! Free fns that derive compiler-friendly summaries off a bare
//! `ModelProto`. The chosen-path install constructs concrete
//! instances via the inventory's `construct_fn` at install time;
//! the IR carries no instance state, so the only derivations the
//! compiler + Node need are the generic-placeholder slot specs
//! and the partition name.

use std::collections::BTreeMap;

use bb_ir::proto::onnx::{FunctionProto, ModelProto};

/// Description of a generic placeholder slot the user must bind at
/// Node construction via `with_<role>(impl)`.
///
/// Previously lived in `crate::built_module`; relocated here as
/// part of Concern 1 (ModelProto deletion).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GenericSlotSpec {
    pub(crate) slot_id: u32,
    pub(crate) required_trait: &'static str,
}

impl GenericSlotSpec {
    /// Construct a slot spec.
    pub fn new(slot_id: u32, required_trait: &'static str) -> Self {
        Self {
            slot_id,
            required_trait,
        }
    }

    /// Slot id minted by `Graph::register_generic` at DSL recording.
    pub fn slot_id(&self) -> u32 {
        self.slot_id
    }

    /// Name of the role trait this slot must be bound to.
    pub fn required_trait(&self) -> &'static str {
        self.required_trait
    }
}

/// Walk every NodeProto in every function of the ModelProto;
/// collect distinct `(slot_id, required_trait)` pairs from
/// `metadata_props`.
pub fn derive_generic_slots(model: &ModelProto) -> Vec<GenericSlotSpec> {
    let mut seen: BTreeMap<u32, &'static str> = BTreeMap::new();
    for function in &model.functions {
        derive_generic_slots_in(function, &mut seen);
    }
    seen.into_iter()
        .map(|(slot_id, required_trait)| GenericSlotSpec {
            slot_id,
            required_trait,
        })
        .collect()
}

fn derive_generic_slots_in(function: &FunctionProto, seen: &mut BTreeMap<u32, &'static str>) {
    for node in &function.node {
        let Some(slot_id) =
            metadata_value(node, "ai.bytesandbrains.slot_id").and_then(|v| v.parse::<u32>().ok())
        else {
            continue;
        };
        let Some(rt) = metadata_value(node, "ai.bytesandbrains.required_trait") else {
            continue;
        };
        // Coerce to a known `&'static str` rather than leaking (the
        // role-trait set is closed).
        let static_rt: Option<&'static str> = match rt {
            "BackendRuntime" => Some("BackendRuntime"),
            "ModelRuntime" => Some("ModelRuntime"),
            "IndexRuntime" => Some("IndexRuntime"),
            "AggregatorRuntime" => Some("AggregatorRuntime"),
            "CodecRuntime" => Some("CodecRuntime"),
            "DataSourceRuntime" => Some("DataSourceRuntime"),
            "PeerSelectorRuntime" => Some("PeerSelectorRuntime"),
            "ProtocolRuntime" => Some("ProtocolRuntime"),
            "WireRuntime" => Some("WireRuntime"),
            _ => None,
        };
        if let Some(rt_static) = static_rt {
            seen.entry(slot_id).or_insert(rt_static);
        }
    }
}

/// Concern 1 - the composite partition name lives on
/// `model.functions[0].name`. Convenience accessor.
pub fn derive_partition_name(model: &ModelProto) -> &str {
    model
        .functions
        .first()
        .map(|f| f.name.as_str())
        .unwrap_or("")
}

fn metadata_value<'a>(node: &'a bb_ir::proto::onnx::NodeProto, key: &str) -> Option<&'a str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

