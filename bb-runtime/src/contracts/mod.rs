//! User-facing Contract traits — what a library maker implements
//! when shipping a concrete component. The derives in `bb-derive`
//! bridge these into the per-component `dispatch_fn` the engine
//! holds.
//!
//! Canonical home for `bb-runtime`. `bb-dsl::contracts` re-exports
//! from here so the authoring path `bb_dsl::Index` keeps working.

pub mod aggregator;
pub mod backend;
pub mod backend_default_walk;
pub mod bootstrap;
pub mod codec;
pub mod data_source;
pub mod index;
pub mod model;
pub mod peer_selector;

pub use aggregator::Aggregator;
pub use backend::Backend;
pub use bootstrap::{Bootstrap, BootstrapCtx};
pub use codec::Codec;
pub use data_source::DataSource;
pub use index::Index;
pub use model::Model;
pub use peer_selector::{PeerSelector, SelectParams};
