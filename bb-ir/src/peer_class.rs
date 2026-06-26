//! Peer-class metadata for partitioning by "what kind of Node does
//! this op run on." [`PEER_CLASS_KEY`] tags `Output<PeerId>`
//! producers; [`HOME_CLASS_KEY`] tags each NodeProto with the class
//! of Node it runs on (stamped by `infer_peer_classes`).

use crate::proto::onnx::{NodeProto, ValueInfoProto};

/// Class of peer an `Output<PeerId>` producer yields. Stamped on
/// `ValueInfoProto` (Graph::input) or `NodeProto` (frontend).
pub const PEER_CLASS_KEY: &str = "ai.bytesandbrains.peer_class";

/// Class of Node a `NodeProto` runs on. Stamped by
/// `infer_peer_classes`; partition keys + cross-class detection
/// consume it.
pub const HOME_CLASS_KEY: &str = "ai.bytesandbrains.home_class";

/// Sentinel for ops whose data inputs all originate on the local
/// Node. Function inputs start here; `wire.Send` re-homes downstream.
pub const SELF_CLASS: &str = "@self";

/// Read `peer_class` off a `ValueInfoProto`. `None` if absent.
pub fn peer_class_of_value_info(vi: &ValueInfoProto) -> Option<&str> {
    vi.metadata_props
        .iter()
        .find(|p| p.key == PEER_CLASS_KEY)
        .map(|p| p.value.as_str())
}

/// Read `peer_class` off a `NodeProto`. `None` if absent.
pub fn peer_class_of_node(node: &NodeProto) -> Option<&str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == PEER_CLASS_KEY)
        .map(|p| p.value.as_str())
}

/// Read `home_class` off a `NodeProto`. `None` for unvisited nodes;
/// partition falls back to [`SELF_CLASS`].
pub fn home_class_of_node(node: &NodeProto) -> Option<&str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == HOME_CLASS_KEY)
        .map(|p| p.value.as_str())
}
