//! `bb::Bootstrap` — optional Component initialization phase.
//!
//! Components override `Bootstrap::bootstrap()` to record one-shot
//! setup logic the framework fires before any body-phase op invokes
//! their Contract methods. Backends use this to wire backend-native
//! tensor pools; indexes use it to mmap their on-disk state; codecs
//! that need a calibration pass use it to drain a sample buffer.
//!
//! The trait default is a no-op so existing Components (Backend,
//! Codec, Index, Aggregator, …) need no change. Authors opt in by
//! implementing the trait alongside their primary Contract — the
//! framework drives Bootstrap ahead of body ops when any of the
//! Component's other Contract methods is reachable from a queued
//! target.
//!
//! Today the trait ships with its types only; F5 wires the Component
//! bootstrap dispatch path (registration + per-poll seeding).
//! Sibling tests assert the default no-op + override observability.

use crate::ids::ComponentRef;

/// Per-dispatch context handed to a Component bootstrap. F5 will
/// extend this with `RuntimeResourceRef`-style accessors so impls
/// can stage outputs, allocate resources, or surface
/// `CompletionHandle`s for async work. Today the struct only
/// carries the dispatching Component's reference so impls have a
/// stable identifier they can log against.
///
/// Held by-mut so the F5 plumbing can mutate per-dispatch staging
/// state without exposing the framework's internal sequencing to
/// the impl.
pub struct BootstrapCtx {
    /// `ComponentRef` of the Component whose bootstrap is firing.
    /// Impls treat this as opaque; debug printers + telemetry taps
    /// surface it so cross-Component traces can correlate the
    /// bootstrap phase with later Contract-method dispatches.
    pub component_ref: ComponentRef,
}

impl BootstrapCtx {
    /// Construct a fresh context for `component_ref`.
    pub fn new(component_ref: ComponentRef) -> Self {
        Self { component_ref }
    }
}

/// User-facing Contract trait for Component bootstrap. Default no-op
/// means existing Components opt in by implementing the trait — the
/// framework treats every Component as implicitly bootstrap-capable.
///
/// Sized to keep the trait usable as a regular bound; the framework
/// invokes the impl through the engine's component table where each
/// entry already carries the concrete type.
pub trait Bootstrap {
    /// Library-maker-defined error type. Must satisfy the standard
    /// engine error bounds so the framework can box it into the
    /// dispatch error channel.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Run one-shot setup. Default no-op so Components without an
    /// initialization phase need no boilerplate. Authors override
    /// to mmap state, allocate backend tensors, prime calibration
    /// buffers, etc.
    fn bootstrap(&mut self, _ctx: &mut BootstrapCtx) -> Result<(), Self::Error> {
        Ok(())
    }
}

