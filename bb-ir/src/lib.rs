#![warn(missing_docs)]
// peer_class.rs links bb-compiler's `infer_peer_classes`; resolves at
// the facade level.
#![allow(rustdoc::broken_intra_doc_links)]

//! Foundation IR crate. Hosts prost-generated ONNX + `bb.core`
//! bindings and the cross-crate types every other crate depends on:
//!
//! - [`ids`] — `PeerId`, `NodeSiteId`, `OpRef`, `ComponentRef`,
//!   `ExecId`, `CommandId`, `RequestId`.
//! - [`wire`] — `TypeNode` denotations + canonical statics.
//! - [`tensor`] — `Scalar`, `Tensor` traits + `Dense<T>` storage.
//! - [`slot_value`] — universal `SlotValue` trait.
//! - [`atomic`] — `AtomicOpsetDecl`, `AtomicOpDecl`, `AtomicOpKind`.

pub mod atomic;
pub mod component;
pub mod ids;
pub mod keys;
pub mod peer_class;
pub mod proto;
pub mod registry;
pub mod slot_value;
pub mod syscall_ids;
pub mod tensor;
pub mod tensor_primitives;
pub mod types;
pub mod verify;
pub mod version;
pub mod wire;
pub mod wire_shape;

pub use tensor_primitives::{
    opset_covers_primitives, MissingPrimitives, TENSOR_PRIMITIVES_DOMAIN, TENSOR_PRIMITIVES_OPS,
    TENSOR_PRIMITIVES_VERSION,
};

// Re-exported so derive macros resolve `$crate::inventory::submit!`
// at downstream sites without adding a direct dep.
pub use inventory;

// Re-exported so `#[derive(bb::Concrete)]` can emit serialize/
// deserialize paths through the facade.
pub use bincode;
