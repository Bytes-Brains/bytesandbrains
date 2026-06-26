#![warn(missing_docs)]
//! `bb-ops` — every concrete component the framework ships.
//! Each component lives in one file colocating IR identity (re-exports
//! from `bb_ir::syscall_ids`), DSL helper, runtime `invoke`, and
//! `inventory::submit!` self-registration.

#![allow(rustdoc::broken_intra_doc_links)]

// bb-derive emits `::bytesandbrains::*` paths; alias for resolution.
extern crate self as bytesandbrains;

// Mirror the facade's module surface for derive path resolution.
pub use bb_dsl::concrete;
pub use bb_dsl::graph;
pub use bb_dsl::module;
pub use bb_dsl::output;
pub use bb_ir::bincode;
pub use bb_ir::inventory;
pub use bb_ir::keys;
pub use bb_ir::proto;
pub use bb_ir::syscall_ids;
pub use bb_ir::tensor;
pub use bb_ir::types;
pub use bb_ir::wire;
pub use bb_runtime::atomic;
pub use bb_runtime::bus;
pub use bb_runtime::completion;
pub use bb_runtime::component;
pub use bb_runtime::contracts;
pub use bb_runtime::engine;
pub use bb_runtime::ids;
pub use bb_runtime::registry;
pub use bb_runtime::roles;
pub use bb_runtime::runtime;
pub use bb_runtime::slot_value;

pub mod aggregators;
pub mod backends;
pub mod network;
pub mod placeholders;
pub mod protocols;
pub mod syscalls;

/// Anchor every `inventory::submit!{}` block against linker DCE. New
/// components must add a `black_box(...)` line; the
/// `tests/component_authoring.rs` assertion catches omissions.
pub fn link_force() {
    use std::hint::black_box;
    // black_box keeps the fn-pointer expression alive past MIR
    // optimization, preserving the object file's static initializers.
    black_box(syscalls::structural::pass_through::invoke as usize);
    black_box(syscalls::structural::tee::invoke as usize);
    black_box(syscalls::structural::constant::invoke as usize);
    black_box(syscalls::composite::bundle::invoke as usize);
    black_box(syscalls::composite::unbundle::invoke as usize);
    black_box(syscalls::coordination::link_force as usize);
    black_box(syscalls::gates::dedup_rx::invoke as usize);
    black_box(syscalls::gates::backoff_rx::invoke as usize);
    black_box(syscalls::gates::backoff_tx::invoke as usize);
    black_box(syscalls::gates::peer_health_rx::invoke as usize);
    black_box(syscalls::gates::peer_health_tx::invoke as usize);
    black_box(syscalls::sync::gate_dispatch::invoke as usize);
    black_box(syscalls::lifecycle::link_force as usize);
    black_box(syscalls::peers::insert::invoke as usize);
    black_box(syscalls::peers::insert_many::invoke as usize);
    black_box(syscalls::peers::lookup::invoke as usize);
    black_box(syscalls::clock_rng::clock::invoke as usize);
    black_box(syscalls::clock_rng::deadline_match::invoke as usize);
    black_box(syscalls::clock_rng::rng_u64::invoke as usize);
    black_box(syscalls::clock_rng::sleep::invoke as usize);
    black_box(syscalls::telemetry::link_force as usize);
    black_box(syscalls::triggers::event_source::invoke as usize);
    black_box(syscalls::triggers::on_trigger::invoke as usize);
    black_box(syscalls::triggers::interval::invoke as usize);
    black_box(syscalls::triggers::pulse::invoke as usize);
    black_box(syscalls::triggers::after::invoke as usize);
    black_box(network::wire::invoke as usize);
    black_box(network::wire::invoke_recv as usize);
    #[cfg(feature = "cpu-backend")]
    black_box(backends::link_force as usize);
}
