//! Hierarchical type system resolved at compile time by the
//! compiler's TypeSolver. The tree is open via inventory: backends
//! and DSL authors register new leaves via `inventory::submit!`. The
//! [`Lattice`] is built once at startup; `is_subtype_of` walks the
//! parent chain with caching. The runtime never sees abstract types.

use std::collections::HashMap;
use std::sync::OnceLock;

pub mod builtins;
pub mod common_relations;
pub mod lattice;
pub mod relations;
pub mod storage;

pub use builtins::*;
pub use common_relations::{
    BROADCAST_BINARY, ELEMENTWISE, MATMUL_BINARY, NO_RELATIONS, REDUCE_AXIS, UNARY_SAME_ELEMENT,
};
pub use lattice::Lattice;
pub use relations::{CustomRelationCtx, PortRef, RelationResult, TypeRelation};
pub use storage::{AnyTensor, Dtype, Storage};

/// Concrete dispatchable leaf vs. abstract interior bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TypeKind {
    /// Interior node — bound only. Never a value's runtime type.
    Abstract,
    /// Dispatchable leaf — always a value's runtime type.
    Concrete,
}

/// Static type-identity carrier. Identity is pointer equality;
/// subtype queries route through [`Lattice`].
#[derive(Clone, Copy, Debug)]
pub struct TypeNode {
    /// Dotted-namespace identifier (`"any"`, `"tensor.f32"`).
    /// Must be unique across submissions.
    pub id: &'static str,
    /// Parent's id, or `None` for the root.
    pub parent: Option<&'static str>,
    /// Dispatchable leaf vs. abstract bound.
    pub kind: TypeKind,
    /// C-FFI struct name (cbindgen). Empty for abstract nodes.
    pub ffi_name: &'static str,
    /// Wire-envelope discriminator. Concrete leaves non-zero;
    /// abstract nodes 0.
    pub wire_hash: u64,
    /// ONNX denotation stamped on `ValueInfoProto.type.denotation`.
    /// Empty when no canonical denotation exists.
    pub denotation: &'static str,
}

impl PartialEq for TypeNode {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for TypeNode {}

impl std::hash::Hash for TypeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}

impl TypeNode {
    /// `true` if `self` is `other` or a descendant. Cached on
    /// the global [`Lattice`].
    pub fn is_subtype_of(&'static self, other: &'static TypeNode) -> bool {
        Lattice::get().is_subtype(self, other)
    }

    /// `true` iff this node is a concrete (dispatchable) leaf.
    pub fn is_concrete(&self) -> bool {
        matches!(self.kind, TypeKind::Concrete)
    }

    /// `true` iff this node is an abstract interior bound.
    pub fn is_abstract(&self) -> bool {
        matches!(self.kind, TypeKind::Abstract)
    }
}

/// Inventory submission carrier for `TypeNode`s.
pub struct TypeNodeReg(pub &'static TypeNode);

inventory::collect!(TypeNodeReg);

/// `id → &TypeNode` map used by [`Lattice`] parent-chain resolution.
pub(crate) fn id_to_node_map() -> &'static HashMap<&'static str, &'static TypeNode> {
    static MAP: OnceLock<HashMap<&'static str, &'static TypeNode>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: HashMap<&'static str, &'static TypeNode> = HashMap::new();
        for reg in inventory::iter::<TypeNodeReg> {
            m.insert(reg.0.id, reg.0);
        }
        m
    })
}

/// Resolve a `TypeNode` by its `id` string. `None` if no submission
/// has registered that id at startup.
pub fn lookup_by_id(id: &str) -> Option<&'static TypeNode> {
    id_to_node_map().get(id).copied()
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod storage_tests;

