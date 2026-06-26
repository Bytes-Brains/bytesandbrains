//! `WireEnvelope` codec
//!
//! Encodes/decodes the canonical `WireEnvelope` proto message via
//! prost. The envelope carries a resolved
//! `dest_peer_addresses: repeated bytes` ordered address list plus
//! one or more typed `SlotFill`s. Each fill's `dest_suffix`
//! identifies the local destination (data-plane slot or
//! control-plane component op) via *intra-node* address segments;
//! receivers route by parsing the suffix. The wire syscall resolves
//! `PeerId → Vec<Address>` via the framework's `AddressBook` before
//! shipping; the host transport picks one entry by capability.

use prost::Message;

pub use bb_ir::proto::bb_core::{CorrelationKind, SlotFill, WireCorrelation, WireEnvelope};

/// Current `WireEnvelope` schema version. Bumped when a field's
/// semantics changes in a way old code cannot soundly handle.
pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

/// Supported schema versions this build will accept on
/// `decode_capped`. Mismatch surfaces as
/// `EnvelopeDecodeError::VersionMismatch`.
pub const SUPPORTED_SCHEMA_VERSIONS: &[u32] = &[ENVELOPE_SCHEMA_VERSION];

/// bounded-decode caps applied by
/// [`EnvelopeCodec::decode_capped`] before any prost allocation.
/// Production deployments override via `NodeConfig` ();
/// the defaults match the design's "16 MiB / 256 / 4 MiB / 4 KiB"
/// recommendation in
/// `docs-plan/CORRECTED_ARCHITECTURE.md` §Edge bounds.
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeCaps {
    /// Reject inbound buffers whose length exceeds this.
    pub max_total_bytes: usize,
    /// Reject envelopes whose `fills.len()` exceeds this.
    pub max_slot_fills: usize,
    /// Reject any `SlotFill` whose `payload.len()` exceeds this.
    pub max_per_fill_bytes: usize,
    /// Reject any `SlotFill` whose `dest_suffix.len()` exceeds this.
    pub max_dest_suffix_bytes: usize,
    /// Reject envelopes whose `src_peer_addresses.len()` exceeds
    /// this. Caps sender-driven AddressBook growth at the receiver.
    pub max_src_peer_addresses: usize,
    /// Reject any `src_peer_addresses` entry whose length exceeds
    /// this. Caps per-address allocation to a single multiaddr's
    /// realistic envelope.
    pub max_src_peer_address_bytes: usize,
}

impl Default for EnvelopeCaps {
    fn default() -> Self {
        Self {
            max_total_bytes: 16 * 1024 * 1024,
            max_slot_fills: 256,
            max_per_fill_bytes: 4 * 1024 * 1024,
            max_dest_suffix_bytes: 4 * 1024,
            max_src_peer_addresses: 8,
            max_src_peer_address_bytes: 256,
        }
    }
}

impl EnvelopeCaps {
    /// Tighter preset for edge / embedded deployments.
    pub fn edge() -> Self {
        Self {
            max_total_bytes: 256 * 1024,
            max_slot_fills: 16,
            max_per_fill_bytes: 64 * 1024,
            max_dest_suffix_bytes: 512,
            max_src_peer_addresses: 4,
            max_src_peer_address_bytes: 256,
        }
    }
}

/// Errors `EnvelopeCodec::decode_capped` can surface.
#[derive(Debug)]
pub enum EnvelopeDecodeError {
    /// Prost rejected the wire-format bytes.
    Malformed(prost::DecodeError),
    /// buffer exceeded `EnvelopeCaps.max_total_bytes`.
    OversizeEnvelope {
        /// Cap that was breached.
        cap_bytes: usize,
        /// Observed buffer length.
        got_bytes: usize,
    },
    /// A single `SlotFill` exceeded `max_per_fill_bytes` or
    /// `max_dest_suffix_bytes`.
    OversizeSlotFill {
        /// Which limit was breached.
        which: &'static str,
        /// Cap that was breached.
        cap_bytes: usize,
        /// Observed length.
        got_bytes: usize,
    },
    /// `fills.len()` exceeded `max_slot_fills`.
    TooManySlotFills {
        /// Cap that was breached.
        cap: usize,
        /// Observed count.
        got: usize,
    },
    /// envelope's `schema_version` is not in
    /// `SUPPORTED_SCHEMA_VERSIONS`.
    VersionMismatch {
        /// Version the inbound envelope advertised.
        got: u32,
        /// Versions this build supports.
        supported: &'static [u32],
    },
    /// `src_peer_addresses.len()` exceeded
    /// `max_src_peer_addresses`.
    TooManySrcPeerAddresses {
        /// Cap that was breached.
        cap: usize,
        /// Observed count.
        got: usize,
    },
    /// A single `src_peer_addresses` entry exceeded
    /// `max_src_peer_address_bytes`.
    OversizeSrcPeerAddress {
        /// Cap that was breached.
        cap_bytes: usize,
        /// Observed length.
        got_bytes: usize,
    },
}

impl std::fmt::Display for EnvelopeDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(e) => write!(f, "malformed envelope bytes: {e}"),
            Self::OversizeEnvelope {
                cap_bytes,
                got_bytes,
            } => write!(
                f,
                "envelope buffer too large: cap={cap_bytes} got={got_bytes}",
            ),
            Self::OversizeSlotFill {
                which,
                cap_bytes,
                got_bytes,
            } => write!(
                f,
                "slot fill {which} exceeds cap: cap={cap_bytes} got={got_bytes}",
            ),
            Self::TooManySlotFills { cap, got } => write!(
                f,
                "envelope carries too many slot fills: cap={cap} got={got}",
            ),
            Self::VersionMismatch { got, supported } => write!(
                f,
                "envelope schema_version mismatch: got={got} supported={supported:?}",
            ),
            Self::TooManySrcPeerAddresses { cap, got } => write!(
                f,
                "envelope carries too many src_peer_addresses: cap={cap} got={got}",
            ),
            Self::OversizeSrcPeerAddress {
                cap_bytes,
                got_bytes,
            } => write!(
                f,
                "src_peer_addresses entry too large: cap={cap_bytes} got={got_bytes}",
            ),
        }
    }
}

impl std::error::Error for EnvelopeDecodeError {}

/// Envelope encode + decode helper. Stateless; thin façade over
/// prost's `Message::encode_to_vec` / `Message::decode`.
pub struct EnvelopeCodec;

impl EnvelopeCodec {
    /// Encode `env` to a prost wire-format byte vector.
    ///
    /// Callers must stamp [`ENVELOPE_SCHEMA_VERSION`] on the envelope
    /// before encode; production paths land it on push into
    /// `OutboundQueue::push` so this function avoids cloning the
    /// (potentially large) payload bytes solely to set one u32.
    pub fn encode(env: &WireEnvelope) -> Vec<u8> {
        env.encode_to_vec()
    }

    /// Bounded decode — the only inbound decode entry. Rejects the
    /// buffer at the length / fill-count / schema-version layer
    /// BEFORE prost allocation, so an adversarial sender can't
    /// pre-balloon memory by advertising large lengths in the
    /// protobuf header.
    pub fn decode_capped(
        bytes: &[u8],
        caps: &EnvelopeCaps,
    ) -> Result<WireEnvelope, EnvelopeDecodeError> {
        if bytes.len() > caps.max_total_bytes {
            return Err(EnvelopeDecodeError::OversizeEnvelope {
                cap_bytes: caps.max_total_bytes,
                got_bytes: bytes.len(),
            });
        }
        let env = WireEnvelope::decode(bytes).map_err(EnvelopeDecodeError::Malformed)?;
        if !SUPPORTED_SCHEMA_VERSIONS.contains(&env.schema_version) && env.schema_version != 0 {
            return Err(EnvelopeDecodeError::VersionMismatch {
                got: env.schema_version,
                supported: SUPPORTED_SCHEMA_VERSIONS,
            });
        }
        if env.fills.len() > caps.max_slot_fills {
            return Err(EnvelopeDecodeError::TooManySlotFills {
                cap: caps.max_slot_fills,
                got: env.fills.len(),
            });
        }
        for fill in &env.fills {
            if fill.payload.len() > caps.max_per_fill_bytes {
                return Err(EnvelopeDecodeError::OversizeSlotFill {
                    which: "payload",
                    cap_bytes: caps.max_per_fill_bytes,
                    got_bytes: fill.payload.len(),
                });
            }
            if fill.dest_suffix.len() > caps.max_dest_suffix_bytes {
                return Err(EnvelopeDecodeError::OversizeSlotFill {
                    which: "dest_suffix",
                    cap_bytes: caps.max_dest_suffix_bytes,
                    got_bytes: fill.dest_suffix.len(),
                });
            }
        }
        if env.src_peer_addresses.len() > caps.max_src_peer_addresses {
            return Err(EnvelopeDecodeError::TooManySrcPeerAddresses {
                cap: caps.max_src_peer_addresses,
                got: env.src_peer_addresses.len(),
            });
        }
        for addr in &env.src_peer_addresses {
            if addr.len() > caps.max_src_peer_address_bytes {
                return Err(EnvelopeDecodeError::OversizeSrcPeerAddress {
                    cap_bytes: caps.max_src_peer_address_bytes,
                    got_bytes: addr.len(),
                });
            }
        }
        Ok(env)
    }
}

