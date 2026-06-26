//! Shared helpers for compiler passes that emit FunctionProtos.
//!
//! [`hash_node_bodies`] produces a stable u64 over a node sequence
//! so identical bodies converge to the same FunctionProto name and
//! dedup at link time.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bb_ir::proto::onnx::NodeProto;

/// Stable short hash of a node sequence. Used to disambiguate two
/// fused regions with the same binding but different bodies, and to
/// converge identical bodies onto the same FunctionProto name (the
/// linker dedupes at register time).
pub(crate) fn hash_node_bodies(nodes: &[NodeProto]) -> u64 {
    let mut h = DefaultHasher::new();
    for node in nodes {
        node.op_type.hash(&mut h);
        node.domain.hash(&mut h);
        node.overload.hash(&mut h);
        node.input.len().hash(&mut h);
        node.output.len().hash(&mut h);
        for inp in &node.input {
            inp.hash(&mut h);
        }
        for out in &node.output {
            out.hash(&mut h);
        }
        for attr in &node.attribute {
            attr.name.hash(&mut h);
        }
    }
    h.finish()
}
