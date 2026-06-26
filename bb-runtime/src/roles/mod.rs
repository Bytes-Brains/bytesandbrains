//! Framework-internal engine-side `<Role>Runtime` traits.
//!
//! Each role trait pairs with a user-facing `bb::<Role>` Contract
//! trait in [`crate::contracts`]; the `#[derive(bb::<Role>)]`
//! macros bridge the user's Contract impl into the engine-side
//! `<Role>Runtime` impl the engine dispatches against. Library
//! authors should not implement these directly — see
//! [`crate::contracts`].
//!
//! Each role trait follows the universal contract:
//!
//! ```ignore
//! pub trait <Role>Runtime: Send + Sync {
//!     type Error: std::error::Error + Send + Sync + 'static;
//!
//!     fn atomic_opset(&self) -> AtomicOpsetDecl;
//!     fn dispatch_atomic(
//!         &mut self,
//!         op_type: &str,
//!         inputs: &[(&str, &dyn SlotValue)],
//!         ctx: &mut RuntimeResourceRef<'_>,
//!     ) -> Result<DispatchResult, Self::Error>;
//! }
//! ```

pub mod aggregator;
pub mod backend;
pub mod codec;
pub mod data_source;
pub mod index;
pub mod model;
pub mod peer_selector;
pub mod protocol;

pub use aggregator::AggregatorRuntime;
pub use backend::BackendRuntime;
pub use codec::CodecRuntime;
pub use data_source::DataSourceRuntime;
pub use index::IndexRuntime;
pub use model::ModelRuntime;
pub use peer_selector::PeerSelectorRuntime;
pub use protocol::ProtocolRuntime;
