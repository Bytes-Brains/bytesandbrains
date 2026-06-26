//! Opt-in OTLP tracing exporter for the engine's spans
//! (, gated by the `tracing-otel`
//! feature).
//!
//! Construct a layer with [`install_otlp_layer`] and compose it into
//! your host's `tracing_subscriber::Registry`. The framework
//! deliberately does NOT install a global subscriber - span
//! emission, runtime ownership, and dispatch-default registration
//! all stay with the host.
//!
//! ## Example
//!
//! ```ignore
//! use tracing_subscriber::prelude::*;
//! let otel = bytesandbrains::telemetry::otel::install_otlp_layer(
//!     "my-service",
//!     "http://localhost:4317",
//! )?;
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer())
//!     .with(otel)
//!     .init();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;

/// Error returned when constructing the OTLP layer fails.
#[derive(Debug)]
pub struct InstallError(pub String);

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OTLP layer install: {}", self.0)
    }
}

impl std::error::Error for InstallError {}

/// Build an OTLP-exporting `tracing` layer that the host can
/// `.with(...)` into a `tracing_subscriber::Registry`. Spans
/// emitted by the engine (`engine.poll`, `engine.phase*`,
/// `engine.invoke_one`, `engine.deliver_fill`, ...) propagate to
/// the configured collector endpoint.
///
/// `service_name` becomes the OTel `service.name` resource
/// attribute. `endpoint` is the OTLP gRPC endpoint (typically
/// `http://localhost:4317`).
pub fn install_otlp_layer(
    service_name: &str,
    endpoint: &str,
) -> Result<
    tracing_opentelemetry::OpenTelemetryLayer<
        tracing_subscriber::registry::Registry,
        opentelemetry_sdk::trace::Tracer,
    >,
    InstallError,
> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.to_string())
        .build()
        .map_err(|e| InstallError(format!("build exporter: {e}")))?;

    let resource = Resource::builder()
        .with_attribute(KeyValue::new("service.name", service_name.to_string()))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();
    let tracer = provider.tracer("bytesandbrains");
    // Hand the provider off as the global so downstream `Tracer`
    // lookups via context-propagation find it.
    let _ = opentelemetry::global::set_tracer_provider(provider);

    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}
