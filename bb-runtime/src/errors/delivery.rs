//! `DeliveryError`
//!
//! Returned by the host-facing `Node::deliver_inbound` /
//! `deliver_event` / `invoke` entry points when delivery cannot be
//! enqueued onto the ingress.

use crate::bus::AllocFailReason;

/// Errors surfaced by host-facing delivery methods on `Node`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeliveryError {
    /// The ingress queue is closed (Node is shutting down).
    IngressClosed,

    /// `deliver_event` / `invoke` referenced an unknown module name.
    UnknownModule(String),

    /// `deliver_event` referenced a module that exists but has no
    /// such input port.
    UnknownInput {
        /// Module name (resolved successfully).
        module: String,
        /// Input port name (not found on the module).
        input: String,
    },

    /// `deliver_inbound` received bytes that failed
    /// `EnvelopeCodec::decode_capped` — malformed prost frame,
    /// schema-version mismatch, or one of the `NodeConfig.envelope_caps`
    /// limits exceeded.
    InvalidEnvelope(String),

    /// `deliver_event` / `invoke` payload exceeded the configured
    /// per-item cap (`NodeConfig::max_app_event_bytes` or
    /// `NodeConfig::max_invoke_bytes`). A matching
    /// `InfraEvent::AppIngressError` lands on the bus alongside this
    /// synchronous return so observers see the per-item rejection.
    OversizePayload {
        /// Bytes the caller attempted to admit.
        byte_count: usize,
        /// Cap value the boundary enforced.
        cap: usize,
    },

    /// `invoke` carried more `(name, bytes)` inputs than the
    /// configured `NodeConfig::max_invoke_inputs` cap allowed.
    /// Matching `InfraEvent::AppIngressError` lands on the bus.
    TooManyInputs {
        /// Inputs the caller attempted to admit.
        count: usize,
        /// Cap value the boundary enforced.
        cap: usize,
    },

    /// `deliver_event` / `invoke` could not allocate the
    /// framework-owned buffer needed to hold the caller's payload,
    /// either because `Vec::try_reserve_exact` returned
    /// `TryReserveError` or because admitting the payload would
    /// exceed `NodeConfig::ingress_byte_budget`. Matching
    /// `InfraEvent::AppIngressError` lands on the bus.
    AllocationFailed {
        /// Bytes the boundary tried to admit.
        byte_count: usize,
        /// Why the reservation failed.
        reason: AllocFailReason,
    },

    /// Admitting this payload would push the engine over
    /// `NodeConfig::ingress_byte_budget`. Matching
    /// `InfraEvent::AppIngressError` lands on the bus.
    BudgetExceeded {
        /// Bytes the boundary tried to admit.
        byte_count: usize,
        /// Bytes still available under the configured budget at the
        /// time of the rejection.
        budget_remaining: usize,
    },
}

impl std::fmt::Display for DeliveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IngressClosed => write!(f, "ingress queue closed"),
            Self::UnknownModule(name) => write!(f, "unknown module: {name}"),
            Self::UnknownInput { module, input } => {
                write!(f, "module {module} has no input port '{input}'")
            }
            Self::InvalidEnvelope(detail) => {
                write!(f, "inbound envelope rejected: {detail}")
            }
            Self::OversizePayload { byte_count, cap } => {
                write!(f, "payload of {byte_count} bytes exceeds cap of {cap}")
            }
            Self::TooManyInputs { count, cap } => {
                write!(f, "{count} inputs exceeds cap of {cap}")
            }
            Self::AllocationFailed { byte_count, reason } => match reason {
                AllocFailReason::HeapExhausted => {
                    write!(f, "heap exhausted reserving {byte_count} bytes")
                }
                AllocFailReason::PerItemCapExceeded { cap } => {
                    write!(
                        f,
                        "per-item cap {cap} rejected payload of {byte_count} bytes"
                    )
                }
            },
            Self::BudgetExceeded {
                byte_count,
                budget_remaining,
            } => write!(
                f,
                "ingress budget exceeded: {byte_count} bytes requested, {budget_remaining} remaining"
            ),
        }
    }
}

impl std::error::Error for DeliveryError {}

