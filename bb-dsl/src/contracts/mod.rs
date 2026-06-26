//! User-facing Contract traits — canonical home is
//! `bb-runtime::contracts`. This module re-exports the traits for
//! authoring ergonomics so `bb_dsl::contracts::Index` etc. keep
//! resolving.

pub use bb_runtime::contracts::*;

/// Re-export of the `bb::Aggregator` Contract trait for federated aggregators.
pub mod aggregator {
    pub use bb_runtime::contracts::aggregator::*;
}
/// Re-export of the `bb::Backend` Contract trait for tensor compute backends.
pub mod backend {
    pub use bb_runtime::contracts::backend::*;
}
/// Re-export of the shared default-walker bridges between `Backend`'s per-op methods and whole-graph `execute`.
pub mod backend_default_walk {
    pub use bb_runtime::contracts::backend_default_walk::*;
}
/// Re-export of the `bb::Bootstrap` Contract trait for Component initialization.
pub mod bootstrap {
    pub use bb_runtime::contracts::bootstrap::*;
}
/// Re-export of the `bb::Codec` Contract trait for bidirectional storage-type codecs.
pub mod codec {
    pub use bb_runtime::contracts::codec::*;
}
/// Re-export of the `bb::DataSource` Contract trait for data loaders.
pub mod data_source {
    pub use bb_runtime::contracts::data_source::*;
}
/// Re-export of the `bb::Index` Contract trait for vector indexes.
pub mod index {
    pub use bb_runtime::contracts::index::*;
}
/// Re-export of the `bb::Model` Contract trait for ML models.
pub mod model {
    pub use bb_runtime::contracts::model::*;
}
/// Re-export of the `bb::PeerSelector` Contract trait for peer-selection protocols.
pub mod peer_selector {
    pub use bb_runtime::contracts::peer_selector::*;
}
