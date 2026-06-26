//! Prost-generated proto bindings.
//!
//! Re-exports the modules produced by `bb-ir/build.rs` from the
//! canonical proto sources at the workspace root:
//!
//!   - `proto/onnx-ml.proto` - vendored canonical ONNX schema
//!     (Apache-2.0).
//!   - `proto/bb_core.proto` - the framework's core schema covering
//!     the wire envelope, slot-fill batching, peer identity, and
//!     snapshots.

/// Vendored canonical ONNX schema (Apache-2.0). Source of
/// `ModelProto`, `GraphProto`, `NodeProto`, `ValueInfoProto`,
/// `TensorProto`, `AttributeProto`, etc. Matches `package onnx;`.
#[allow(missing_docs, clippy::doc_overindented_list_items)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/// Framework core schema. Covers the wire envelope (`WireEnvelope`,
/// `OpsetId`, `WireCorrelation`, `CorrelationKind`), data-plane
/// batching (`SlotFillBatch`, `SlotFill`), peer identity
/// (`PeerProto`), and snapshots (`NodeSnapshotProto`,
/// `ComponentSnapshotProto`). Matches `package bb.core;`.
#[allow(missing_docs, clippy::doc_overindented_list_items)]
pub mod bb_core {
    include!(concat!(env!("OUT_DIR"), "/bb.core.rs"));
}

mod onnx_view;
pub use onnx_view::function_to_graph_view;
