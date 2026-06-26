//! Optional OpenTelemetry layer constructors for the engine's
//! `tracing::` spans.
//!
//! The engine emits `tracing::debug_span!` calls on every poll
//! phase and on internal dispatch hot-paths regardless of feature
//! flags. To export those spans to an OTLP collector, opt in to
//! the `tracing-otel` feature and call `otel::install_otlp_layer`
//! from your host crate's startup.
//!
//! The framework deliberately does NOT auto-install the subscriber
//! - the host owns the global `tracing::dispatcher::set_global_default`
//! handle. Build the OTel layer here, then compose it with whatever
//! other layers (fmt, json, etc.) your host uses.

#[cfg(feature = "tracing-otel")]
pub mod otel;
