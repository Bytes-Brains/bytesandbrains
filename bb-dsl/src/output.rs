//! Non-generic `Output` handle threaded through DSL method chains.
//!
//! Per `docs/API_DESIGN.md` §4 + `docs/IR_AND_DSL.md` Part 6. The DSL
//! is non-generic across languages - type metadata rides on
//! `&'static TypeNode`, NOT a `PhantomData<T>` tag. Identity is the
//! `name: String` (the ONNX value name in `FunctionProto.input` /
//! `NodeProto.input` / `NodeProto.output`); the wire-level type
//! identity rides on the `TypeNode` reference.

use bb_ir::types::TypeNode;

/// Handle passed between DSL methods. Carries the recorded ONNX
/// value name plus a `&'static` pointer to the canonical
/// [`TypeNode`] of the value's type.
#[derive(Clone, Debug)]
pub struct Output {
    /// ONNX value name. Matches a `FunctionProto.input` entry, a
    /// `NodeProto.output` entry, or a `next_site_name()` mint.
    pub name: String,

    /// Static `TypeNode` reference. Pointer equality is meaningful
    /// - every canonical type lives in a single `static`.
    pub type_node: &'static TypeNode,
}

impl Output {
    /// Construct an `Output` handle from a value name + canonical
    /// type metadata.
    pub fn new(name: String, type_node: &'static TypeNode) -> Self {
        Self { name, type_node }
    }
}

