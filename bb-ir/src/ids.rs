//! Wire-format and compiler-bound identifier types.
//!
//! IDs tied to the IR or wire envelope live here ([`PeerId`],
//! [`RequestId`], [`OpsetId`], [`ComponentTag`]). Engine-internal
//! dispatch IDs (`NodeSiteId`, `OpRef`, `ExecId`, `CommandId`,
//! `ComponentRef`) live in `bb_runtime::ids`.

use std::fmt;

// --- Macro helpers ----------------------------------------------

macro_rules! u64_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
            serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Construct from an explicit value.
            pub const fn new(inner: u64) -> Self { Self(inner) }

            /// Inner value accessor.
            pub const fn as_u64(self) -> u64 { self.0 }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl From<u64> for $name {
            fn from(inner: u64) -> Self { Self(inner) }
        }
    };
}

// --- Wire-format / compiler-bound IDs ---------------------------

u64_id! {
    /// Per-request correlation token. Issued by `SendReqBatched`
    /// senders; echoed by receivers in `SendResp`.
    RequestId
}

// --- Symbolic IDs -----------------------------------------------

/// Typed bus-subscription tag - a static `&str` label naming the
/// kind of event a component receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentTag(pub &'static str);

impl ComponentTag {
    /// Construct from an explicit static label.
    pub const fn new(tag: &'static str) -> Self {
        Self(tag)
    }

    /// Inner label accessor.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for ComponentTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComponentTag({:?})", self.0)
    }
}

/// Opset identifier. `(domain, version)` pair routing wire-level
/// dispatch. See `docs/IR_AND_DSL.md` §5 for the canonical opset
/// catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpsetId {
    /// Reverse-DNS-ish domain (`ai.onnx`, `bb.wire`, `bb.gossip`,
    /// `user.kademlia`, ...).
    pub domain: &'static str,

    /// Major version. Minor/patch live in the component's own
    /// versioning surface; this field gates wire-level
    /// compatibility.
    pub version: i64,
}

impl OpsetId {
    /// Construct from explicit values.
    pub const fn new(domain: &'static str, version: i64) -> Self {
        Self { domain, version }
    }
}

impl fmt::Display for OpsetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{}", self.domain, self.version)
    }
}

// --- PeerId - libp2p-compatible multihash -----------------------

/// Peer identity. Wraps a fixed-capacity `Multihash<64>` so
/// overlays pick their own digest algorithm. Same wire shape as
/// `libp2p_identity::PeerId`; bytes round-trip without translation.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
pub struct PeerId(multihash::Multihash<64>);

impl PeerId {
    /// Multihash code for `identity` (raw bytes, no hashing).
    pub const IDENTITY: u64 = 0x00;

    /// Multihash code for `sha2-256`.
    pub const SHA2_256: u64 = 0x12;

    /// Construct from a multihash directly.
    pub const fn from_multihash(mh: multihash::Multihash<64>) -> Self {
        Self(mh)
    }

    /// Parse the canonical multihash byte form
    /// (`varint(code) ++ varint(size) ++ digest`). Wire-format
    /// compatible with libp2p.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, multihash::Error> {
        multihash::Multihash::from_bytes(bytes).map(Self)
    }

    /// Canonical multihash byte form, matches libp2p's
    /// `PeerId::to_bytes`.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes()
    }

    /// Multihash algorithm code (e.g. `0x00` identity, `0x12`
    /// sha2-256).
    pub fn code(&self) -> u64 {
        self.0.code()
    }

    /// Raw digest bytes (after the multihash code + size prefix).
    pub fn digest(&self) -> &[u8] {
        self.0.digest()
    }

    /// Borrow the inner multihash.
    pub fn as_multihash(&self) -> &multihash::Multihash<64> {
        &self.0
    }

    /// Recover the inner u64 of a `PeerId::from(u64)`. Returns
    /// `None` for non-identity-coded or non-8-byte digests.
    pub fn as_identity_u64(&self) -> Option<u64> {
        if self.0.code() == Self::IDENTITY {
            let d = self.0.digest();
            if d.len() == 8 {
                return Some(u64::from_be_bytes(d.try_into().expect("checked len == 8")));
            }
        }
        None
    }
}

impl From<u64> for PeerId {
    /// Test convenience: wraps the u64 as an 8-byte identity-coded
    /// multihash. **Not for production identity** — use
    /// `PeerId::from_multihash` / `from_bytes` for keyed identities.
    fn from(value: u64) -> Self {
        let bytes = value.to_be_bytes();
        let mh = multihash::Multihash::<64>::wrap(Self::IDENTITY, &bytes)
            .expect("identity hash of 8 bytes always fits in 64");
        Self(mh)
    }
}

impl fmt::Display for PeerId {
    /// Base58btc-encoded multihash; matches libp2p's `PeerId::to_base58`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&bs58::encode(self.to_bytes()).into_string())
    }
}

