//! Public error taxonomies surfaced by the user-facing API.
//!
//! Four error families ship here, each tied to a specific
//! `Node`-lifecycle step:
//!
//! - [`DeliveryError`] — host-facing `Node::deliver_*` /
//!   `Node::invoke` failures.
//! - [`RestoreError`] — `Node::restore` snapshot reconciliation
//!   failures.
//! - [`SnapshotError`] — `Node::snapshot` failures.
//! - [`BootstrapError`] — `Node::run_bootstrap` input staging +
//!   target selection failures per the host-driven bootstrap
//!   redesign.
//!
//! `Module::build` failures live in `bb_dsl::BuildError`; compiler
//! failures live in `bb_compiler::CompileError`; installation
//! failures live in `bytesandbrains::InstallError`. All three
//! surface upstream of the engine and outside `bb-runtime`'s dep
//! graph.

pub mod bootstrap;
pub mod delivery;
pub mod restore;
pub mod snapshot;

pub use bootstrap::BootstrapError;
pub use delivery::DeliveryError;
pub use restore::RestoreError;
pub use snapshot::SnapshotError;
