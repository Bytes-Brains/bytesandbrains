//! Helpers that adapt ONNX proto types between equivalent shapes.

use std::collections::HashMap;

use super::onnx::{FunctionProto, GraphProto, ValueInfoProto};

/// Build a `GraphProto` view from a `FunctionProto`'s body - most
/// compiler passes operate on a `GraphProto` per `docs/COMPILER.md`,
/// and the engine's backend-subgraph dispatcher needs the same view
///
///
/// The view resolves `function.input` / `output` (which are
/// `repeated string` at the FunctionProto level) into `ValueInfoProto`s
/// by looking up the matching entry in `function.value_info`. Missing
/// entries surface as `ValueInfoProto { name, ..default() }`; the
/// compiler's `validate` pass then catches them as `MissingTypeInfo`.
pub fn function_to_graph_view(function: &FunctionProto) -> GraphProto {
    let lookup: HashMap<&str, &ValueInfoProto> = function
        .value_info
        .iter()
        .map(|v| (v.name.as_str(), v))
        .collect();
    let resolve = |name: &str| -> ValueInfoProto {
        lookup
            .get(name)
            .map(|v| (*v).clone())
            .unwrap_or(ValueInfoProto {
                name: name.to_string(),
                ..Default::default()
            })
    };
    GraphProto {
        node: function.node.clone(),
        name: function.name.clone(),
        input: function.input.iter().map(|n| resolve(n)).collect(),
        output: function.output.iter().map(|n| resolve(n)).collect(),
        value_info: function.value_info.clone(),
        ..Default::default()
    }
}
