//! `BackoffNoticePayload` - typed wire payload for the backpressure
//! protocol per
//! `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
//! §2.
//!
//! The framework synthesizes one of these whenever
//! `BackpressureTracker::observe_overload` returns
//! [`super::BackpressureDecision::EmitNotice`]. The payload is
//! serialized with `bincode` (matches the universal `SlotValue` wire
//! encoding at `bb-ir/src/slot_value.rs:194-196`), stamped with a
//! stable `type_hash` discriminator, packed into a single
//! [`crate::envelope::SlotFill`] inside a [`crate::envelope::WireEnvelope`]
//! addressed to the originating sender, and pushed onto the
//! receiver's `OutboundQueue`.
//!
//! Receivers detect the notice by matching the per-fill `type_hash`
//! against [`backoff_notice_type_hash`] in their inbound envelope
//! routing - the framework intercepts BackoffNotice envelopes before
//! data-plane / control-plane dispatch so user Components never see
//! them. Sender-side handling updates the sender's
//! [`crate::framework::BackoffTable`] so the existing
//! `BackoffGateTx` consultation gates the next outbound send to
//! that peer.

use serde::{Deserialize, Serialize};

use crate::envelope::{CorrelationKind, SlotFill, WireCorrelation, WireEnvelope};
use crate::framework::address_book::Address;
use crate::framework::BackoffCause;
use crate::ids::PeerId;
use crate::slot_value::type_hash_of;

/// Domain string the framework uses to namespace the backpressure
/// protocol per `bb-runtime/src/bus.rs:155` reserved framework
/// prefix. Surfaced for cross-referencing in docs + tests; the
/// actual routing key is the per-fill `type_hash` from
/// [`backoff_notice_type_hash`].
pub const BACKPRESSURE_DOMAIN: &str = "ai.bytesandbrains.backpressure";

/// Wire-encoded BackoffNotice payload.
///
/// One of these rides as the sole `SlotFill.payload` of a
/// BackoffNotice envelope. Field semantics:
///
/// - `min_backoff_ns` - minimum back-off duration the receiver is
///   asking the sender to observe before its next envelope. The
///   sender's `BackoffTable` translates this into a `next_retry_ns`
///   so the existing `BackoffGateTx` already gates outbound sends.
/// - `cause` - why the receiver is requesting back-off.
/// - `suggested_next_send_ns` - optional wall-clock hint (engine-ns
///   since epoch) the receiver believes it will be ready by. `None`
///   when the receiver has no estimate.
///
/// The payload uses `bincode` serialization (the universal
/// `SlotValue::to_wire_bytes` impl at
/// `bb-ir/src/slot_value.rs:194-196` already routes through bincode),
/// and the discriminator [`backoff_notice_type_hash`] is identical
/// on both sides because it derives from
/// `std::any::type_name::<BackoffNoticePayload>()` via FNV-1a.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackoffNoticePayload {
    /// Minimum back-off duration in nanoseconds.
    pub min_backoff_ns: u64,
    /// Why the receiver is requesting back-off.
    pub cause: BackoffCauseWire,
    /// Optional wall-clock hint (engine-ns since epoch) the receiver
    /// expects to be ready by. `0` encodes `None`.
    pub suggested_next_send_ns: u64,
}

/// Wire-stable encoding of [`BackoffCause`]. Serialized as a u8 so
/// the on-wire representation never bit-shifts when the framework
/// enum evolves. Always derived from the framework enum at the
/// send site + mapped back at the receive site.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum BackoffCauseWire {
    /// `IngressQueue` depth crossed the high-water mark.
    QueueFull = 0,
    /// `PhiAccrualState` marked the sender as `Suspect`.
    PhiAccrual = 1,
    /// A Component returned a typed reject (e.g. role rate-limit).
    ExplicitDrop = 2,
}

impl From<BackoffCause> for BackoffCauseWire {
    fn from(cause: BackoffCause) -> Self {
        match cause {
            BackoffCause::QueueFull => Self::QueueFull,
            BackoffCause::PhiAccrual => Self::PhiAccrual,
            BackoffCause::ExplicitDrop => Self::ExplicitDrop,
        }
    }
}

impl From<BackoffCauseWire> for BackoffCause {
    fn from(cause: BackoffCauseWire) -> Self {
        match cause {
            BackoffCauseWire::QueueFull => Self::QueueFull,
            BackoffCauseWire::PhiAccrual => Self::PhiAccrual,
            BackoffCauseWire::ExplicitDrop => Self::ExplicitDrop,
        }
    }
}

impl BackoffNoticePayload {
    /// Construct a payload, encoding `None` as `0` per the field
    /// docstring.
    pub fn new(
        min_backoff_ns: u64,
        cause: BackoffCause,
        suggested_next_send_ns: Option<u64>,
    ) -> Self {
        Self {
            min_backoff_ns,
            cause: cause.into(),
            suggested_next_send_ns: suggested_next_send_ns.unwrap_or(0),
        }
    }

    /// Recover the optional wall-clock hint.
    pub fn suggested_next_send(&self) -> Option<u64> {
        if self.suggested_next_send_ns == 0 {
            None
        } else {
            Some(self.suggested_next_send_ns)
        }
    }

    /// Recover the framework-side [`BackoffCause`].
    pub fn cause(&self) -> BackoffCause {
        self.cause.into()
    }

    /// Serialize via `bincode`. Matches the universal
    /// `SlotValue::to_wire_bytes` impl at
    /// `bb-ir/src/slot_value.rs:194-196`.
    pub fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("BackoffNoticePayload bincode serialize is infallible")
    }

    /// Deserialize via `bincode`. Returns `None` when the payload
    /// bytes don't round-trip (corrupt envelope / version skew).
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize::<Self>(bytes).ok()
    }
}

/// Stable u64 discriminator for `BackoffNoticePayload`. Matches the
/// canonical `SlotFill.type_hash` the framework stamps at send time
/// and consults at receive time per
/// `bb-ir/src/slot_value.rs:203-210`.
pub fn backoff_notice_type_hash() -> u64 {
    type_hash_of::<BackoffNoticePayload>()
}

/// Build a `BackoffNotice` wire envelope addressed to `sender`.
///
/// The envelope's `dest_peer_addresses` carries one `/p2p/<sender>`
/// entry; its sole `SlotFill` uses a reserved framework dest-suffix
/// `/p2p/<self_peer>` so the receiver's `route_envelope` can
/// recognize a notice by type_hash before any data-plane /
/// control-plane dispatch. The fill's `type_hash` is set to
/// [`backoff_notice_type_hash`] so the receiver matches by
/// discriminator instead of payload-content inspection.
pub fn build_backoff_notice_envelope(
    self_peer: PeerId,
    sender: PeerId,
    payload: BackoffNoticePayload,
) -> WireEnvelope {
    let dest_addr = Address::empty().p2p(sender).to_bytes();
    // The fill's dest_suffix uses a reserved `/p2p/<self_peer>`
    // suffix that intentionally does NOT match the data-plane /
    // control-plane suffix shapes. The receiver intercepts on
    // `type_hash` so the suffix is informational only - it carries
    // the originating self-peer for diagnostics.
    let dest_suffix = Address::empty().p2p(self_peer).to_bytes();
    let bytes = payload.encode();
    WireEnvelope {
        dest_peer_addresses: vec![dest_addr],
        fills: vec![SlotFill {
            dest_suffix,
            payload: bytes,
            trigger_only: false,
            type_hash: backoff_notice_type_hash(),
        }],
        correlation: Some(WireCorrelation {
            kind: CorrelationKind::None as i32,
            wire_req_id: 0,
        }),
        remaining_deadline_ns: 0,
        edge_rtt_reports: Vec::new(),
        ..Default::default()
    }
}

