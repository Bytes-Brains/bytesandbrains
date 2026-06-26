//! `AddressBook` - global `PeerId → (Vec<Address>, ref_count)`
//! registry per `ENGINE.md` §10.5 + `docs/ADDRESSING.md`.
//!
//! Real-world peers expose multiple reachable endpoints
//! (`/ip4/.../tcp/...`, `/ip6/.../quic/...`, `/relay/...`); the
//! AddressBook holds the ordered list per peer (insertion order =
//! peer-stated preference) and the host transport adapter picks
//! one based on its networking capabilities.
//!
//! Entries are **reference-counted**: multiple overlay protocols
//! can independently "own" the same peer without duplicating
//! address records. `add_peer` increments the count;
//! `drop_peer` decrements; the entry is removed when the count
//! hits zero. Components that need peer metadata (gossip
//! overlays, peer-sampling views) store their own metadata in
//! their own state and call `lookup` here for addresses -
//! addresses live in exactly one place.
//!
//! The wire syscall (`src/syscall/wire.rs`) consults this on
//! every `wire::Send` to populate
//! `WireEnvelope.dest_peer_addresses` with the resolved list;
//! a lookup miss surfaces as `EngineStep::PeerResolveFailed`.
//!
//! Addresses in bytes-and-brains are **multiaddrs**. The framework
//! only carries the segments it routes on internally - transport
//! segments (Ip4 / Tcp / etc.) and any other host-managed routing
//! data live in the host adapter. The framework's `Protocol` enum
//! is the four BB-internal variants only:
//!
//! - `P2p(PeerId)` - peer identity, consulted by the AddressBook.
//! - `Site(NodeSiteId)` - data-plane slot fill target.
//! - `Component(ComponentRef)` - control-plane component identity.
//! - `Op(String)` - control-plane op name for `dispatch_atomic`.
//!
//! Receivers route directly by parsing the multiaddr suffix - no
//! per-message-type subscription tables, no endpoint id lookups.
//! Hosts that need IP/port/sim-channel identity prepend whatever
//! bytes they like; the framework treats those bytes as opaque.

use std::collections::HashMap;
use std::fmt;

use bb_ir::types::{Storage, TypeNode, TYPE_MULTIADDRESS};

use crate::ids::{ComponentRef, NodeSiteId, PeerId};

/// Public DSL alias - the chosen name for `Address` on the graph
/// surface. The internal type is `framework::Address`; the alias
/// keeps user-facing code (`Output<Multiaddress>`, `Multiaddress`
/// constants) reading naturally without introducing a second type.
pub type Multiaddress = Address;

/// One typed protocol segment in an [`Address`] multiaddr. Each
/// variant maps to a stable BB-specific multiaddr code in the
/// 0xE1-0xEF range (`P2p` reuses the libp2p code 0x55 since it
/// carries the BB `PeerId` identity that crosses transport
/// adapters). The binary encoding is `code (u8) || payload` where
/// the payload's length is fixed for every variant except `Op`
/// (which is length-prefixed).
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Protocol {
    /// Peer identity (BB `PeerId`).
    P2p(PeerId),
    /// Data-plane slot fill target - the receiver writes the payload
    /// to the slot at this `NodeSiteId` and pushes downstream
    /// consumers.
    Site(NodeSiteId),
    /// Control-plane component identity. Combined with an `Op`
    /// segment, identifies the receiver's `dispatch_atomic` target.
    Component(ComponentRef),
    /// Op name for control-plane dispatch (e.g. `"FindNode"`). The
    /// receiver calls `component[cref].dispatch_atomic(op_name,
    /// payload, ctx)`.
    Op(String),
}

/// Standard multiaddr protocol code for `/p2p/<peerid>`. Encoded
/// as an unsigned LEB128 varint; the value is a length-prefixed
/// multihash. Matches libp2p's wire format byte-for-byte.
const CODE_P2P: u64 = 421;

/// Framework-internal slot-routing segment. Value is a fixed
/// 8-byte big-endian `NodeSiteId`. Code lives outside the libp2p
/// standard range; values 0xE0-0xEF are unassigned upstream.
const CODE_SITE: u64 = 0xE2;

/// Framework-internal component-routing segment. Value is a
/// fixed 4-byte big-endian `ComponentRef`.
const CODE_COMPONENT: u64 = 0xE3;

/// Framework-internal control-plane op name. Value is a
/// length-prefixed UTF-8 string (varint-prefix matches `/p2p/`).
const CODE_OP: u64 = 0xE4;

impl Protocol {
    /// Multiaddr protocol code. Encoded as an unsigned LEB128
    /// varint on the wire. `P2p` uses the standard libp2p code
    /// 421; framework-internal codes (Site / Component / Op) live
    /// in the 0xE0-range.
    pub const fn code(&self) -> u64 {
        match self {
            Protocol::P2p(_) => CODE_P2P,
            Protocol::Site(_) => CODE_SITE,
            Protocol::Component(_) => CODE_COMPONENT,
            Protocol::Op(_) => CODE_OP,
        }
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        let mut code_buf = unsigned_varint::encode::u64_buffer();
        out.extend_from_slice(unsigned_varint::encode::u64(self.code(), &mut code_buf));
        match self {
            Protocol::P2p(p) => {
                // `/p2p/<peerid>`: value is length-prefixed
                // multihash bytes - matches libp2p exactly.
                let mh_bytes = p.to_bytes();
                let mut len_buf = unsigned_varint::encode::usize_buffer();
                out.extend_from_slice(unsigned_varint::encode::usize(mh_bytes.len(), &mut len_buf));
                out.extend_from_slice(&mh_bytes);
            }
            Protocol::Site(s) => out.extend_from_slice(&s.as_u64().to_be_bytes()),
            Protocol::Component(c) => out.extend_from_slice(&c.as_u32().to_be_bytes()),
            Protocol::Op(name) => {
                let bytes = name.as_bytes();
                let mut len_buf = unsigned_varint::encode::usize_buffer();
                out.extend_from_slice(unsigned_varint::encode::usize(bytes.len(), &mut len_buf));
                out.extend_from_slice(bytes);
            }
        }
    }

    /// Parse one segment from the head of `buf`. On success returns
    /// `(protocol, bytes_consumed)`. The caller advances by
    /// `bytes_consumed` for the next segment.
    fn read_from(buf: &[u8]) -> Result<(Self, usize), AddressError> {
        let (code, rest) =
            unsigned_varint::decode::u64(buf).map_err(|_| AddressError::Truncated)?;
        let code_len = buf.len() - rest.len();
        let (prot, payload_len) = match code {
            CODE_P2P => {
                let (mh_len, after_len) =
                    unsigned_varint::decode::usize(rest).map_err(|_| AddressError::Truncated)?;
                let mh_len_bytes = rest.len() - after_len.len();
                let mh_bytes = after_len.get(0..mh_len).ok_or(AddressError::Truncated)?;
                let peer =
                    PeerId::from_bytes(mh_bytes).map_err(|_| AddressError::InvalidValue {
                        protocol: "p2p".into(),
                        value: format!("malformed multihash ({mh_len} bytes)"),
                    })?;
                (Protocol::P2p(peer), mh_len_bytes + mh_len)
            }
            CODE_SITE => {
                let bytes: [u8; 8] = rest
                    .get(0..8)
                    .ok_or(AddressError::Truncated)?
                    .try_into()
                    .expect("8-byte slice");
                (
                    Protocol::Site(NodeSiteId::from(u64::from_be_bytes(bytes))),
                    8,
                )
            }
            CODE_COMPONENT => {
                let bytes: [u8; 4] = rest
                    .get(0..4)
                    .ok_or(AddressError::Truncated)?
                    .try_into()
                    .expect("4-byte slice");
                (
                    Protocol::Component(ComponentRef::from(u32::from_be_bytes(bytes))),
                    4,
                )
            }
            CODE_OP => {
                let (str_len, after_len) =
                    unsigned_varint::decode::usize(rest).map_err(|_| AddressError::Truncated)?;
                let len_bytes = rest.len() - after_len.len();
                let str_bytes = after_len.get(0..str_len).ok_or(AddressError::Truncated)?;
                let name = std::str::from_utf8(str_bytes)
                    .map_err(|_| AddressError::InvalidValue {
                        protocol: "op".into(),
                        value: format!("non-utf8 {str_len} bytes"),
                    })?
                    .to_string();
                (Protocol::Op(name), len_bytes + str_len)
            }
            other => {
                // `UnknownCode` only carries a u8 today - narrow
                // unrecognized varints into that range for the
                // existing error variant.
                return Err(AddressError::UnknownCode((other & 0xFF) as u8));
            }
        };
        Ok((prot, code_len + payload_len))
    }
}

/// Multiaddr - a sequence of typed [`Protocol`] segments describing
/// a delivery path. Per `docs/ADDRESSING.md`, this is BB's canonical
/// address type: the suffix segments tell the receiver where to
/// route inside its own node, so no per-message-type or
/// per-endpoint-id lookup tables are needed.
///
/// Construct via the builder chain
/// (`Address::empty().p2p(...).site(...).component(...).op(...)`),
/// [`Address::from_bytes`], or [`Address::parse_str`].
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Address {
    segments: Vec<Protocol>,
}

/// Decode errors surfaced by [`Address::from_bytes`] and
/// [`Address::parse_str`].
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum AddressError {
    /// Buffer ended before a segment finished decoding.
    Truncated,
    /// First byte of a segment didn't match any known protocol code.
    UnknownCode(u8),
    /// String form didn't match `/protocol/value/...`.
    MalformedString(String),
    /// String value couldn't be parsed as the named protocol's type.
    InvalidValue {
        /// Protocol whose value failed to parse.
        protocol: String,
        /// Original raw value.
        value: String,
    },
}

impl fmt::Display for AddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressError::Truncated => write!(f, "Address: buffer truncated"),
            AddressError::UnknownCode(c) => write!(f, "Address: unknown protocol code 0x{c:x}"),
            AddressError::MalformedString(s) => write!(f, "Address: malformed string `{s}`"),
            AddressError::InvalidValue { protocol, value } => {
                write!(
                    f,
                    "Address: invalid value `{value}` for protocol `{protocol}`"
                )
            }
        }
    }
}

impl std::error::Error for AddressError {}

impl Address {
    /// Empty Address - useful as a "no destination" sentinel.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Append a segment. Internal - external callers use the typed
    /// `.p2p()` / `.site()` / `.component()` / `.op()` builders so the
    /// allowed segment shapes can't drift outside this module.
    fn with(mut self, segment: Protocol) -> Self {
        self.segments.push(segment);
        self
    }

    /// Builder - append P2P peer id.
    pub fn p2p(self, peer: PeerId) -> Self {
        self.with(Protocol::P2p(peer))
    }
    /// Builder - append data-plane Site segment.
    pub fn site(self, site: NodeSiteId) -> Self {
        self.with(Protocol::Site(site))
    }
    /// Builder - append control-plane Component segment.
    pub fn component(self, c: ComponentRef) -> Self {
        self.with(Protocol::Component(c))
    }
    /// Builder - append control-plane Op segment.
    pub fn op(self, name: impl Into<String>) -> Self {
        self.with(Protocol::Op(name.into()))
    }

    /// Read-only view of the segments.
    pub fn segments(&self) -> &[Protocol] {
        &self.segments
    }

    /// First `P2p` segment's peer id, if any.
    pub fn peer_id(&self) -> Option<PeerId> {
        self.segments.iter().find_map(|p| match p {
            Protocol::P2p(id) => Some(*id),
            _ => None,
        })
    }

    /// First `Site` segment, if any.
    pub fn site_id(&self) -> Option<NodeSiteId> {
        self.segments.iter().find_map(|p| match p {
            Protocol::Site(id) => Some(*id),
            _ => None,
        })
    }

    /// First `Component` segment, if any.
    pub fn component_ref(&self) -> Option<ComponentRef> {
        self.segments.iter().find_map(|p| match p {
            Protocol::Component(c) => Some(*c),
            _ => None,
        })
    }

    /// First `Op` segment, if any (control-plane op name).
    pub fn op_name(&self) -> Option<&str> {
        self.segments.iter().find_map(|p| match p {
            Protocol::Op(name) => Some(name.as_str()),
            _ => None,
        })
    }

    /// Encode to canonical binary form.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for seg in &self.segments {
            seg.write_to(&mut buf);
        }
        buf
    }

    /// Decode from canonical binary form.
    pub fn from_bytes(mut bytes: &[u8]) -> Result<Self, AddressError> {
        let mut segments: Vec<Protocol> = Vec::new();
        while !bytes.is_empty() {
            let (seg, consumed) = Protocol::read_from(bytes)?;
            segments.push(seg);
            bytes = &bytes[consumed..];
        }
        Ok(Address { segments })
    }

    /// Parse a `/protocol/value/...` string form. Only the four
    /// BB-internal protocols (`p2p`, `site`, `component`, `op`) are
    /// recognized - transport-layer segments (ip4/tcp/udp/etc.)
    /// belong to host adapters and never reach this parser.
    ///
    /// Example: `/p2p/12/site/17` or `/p2p/12/component/5/op/FindNode`.
    pub fn parse_str(s: &str) -> Result<Self, AddressError> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(Self::empty());
        }
        if !trimmed.starts_with('/') {
            return Err(AddressError::MalformedString(s.to_string()));
        }
        let mut parts = trimmed[1..].split('/').peekable();
        let mut segments: Vec<Protocol> = Vec::new();
        while parts.peek().is_some() {
            let protocol = parts
                .next()
                .ok_or_else(|| AddressError::MalformedString(s.to_string()))?;
            let value = parts
                .next()
                .ok_or_else(|| AddressError::MalformedString(s.to_string()))?;
            let seg = match protocol {
                "p2p" => {
                    // Value is base58btc-encoded multihash bytes,
                    // matches libp2p's `/p2p/Qm…` string form.
                    let mh_bytes = bs58::decode(value)
                        .into_vec()
                        .map_err(|_| invalid(protocol, value))?;
                    Protocol::P2p(
                        PeerId::from_bytes(&mh_bytes).map_err(|_| invalid(protocol, value))?,
                    )
                }
                "site" => Protocol::Site(NodeSiteId::from(
                    value.parse::<u64>().map_err(|_| invalid(protocol, value))?,
                )),
                "component" => Protocol::Component(ComponentRef::from(
                    value.parse::<u32>().map_err(|_| invalid(protocol, value))?,
                )),
                "op" => Protocol::Op(value.to_string()),
                other => return Err(invalid(other, value)),
            };
            segments.push(seg);
        }
        Ok(Address { segments })
    }
}

fn invalid(protocol: &str, value: &str) -> AddressError {
    AddressError::InvalidValue {
        protocol: protocol.into(),
        value: value.into(),
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for seg in &self.segments {
            match seg {
                Protocol::P2p(id) => write!(f, "/p2p/{id}")?,
                Protocol::Site(id) => write!(f, "/site/{}", id.as_u64())?,
                Protocol::Component(c) => write!(f, "/component/{}", c.as_u32())?,
                Protocol::Op(name) => write!(f, "/op/{}", name)?,
            }
        }
        Ok(())
    }
}

impl std::str::FromStr for Address {
    type Err = AddressError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_str(s)
    }
}

// `Address` is `Sized` (it owns `Vec<Protocol>`), so the `Storage`
// impl applies to the owned type — distinct from the tensor leaves
// in `bb-ir`, which impl `Storage` for `?Sized` slice types. The
// associated `Storage::TYPE` lets custom ops declare `Multiaddress`
// ports and have the type solver narrow on them.
impl Storage for Address {
    const TYPE: &'static TypeNode = &TYPE_MULTIADDRESS;
}

// `Address` itself is never a slot carrier in production code —
// the typed wrapper `crate::syscall::values::AddressValue` is what
// flows through the slot table, and that's where the
// `register_type_node!(AddressValue, &TYPE_MULTIADDRESS)` binding
// lives. The `Storage for Address` impl above is what custom ops
// reach for at compile time when declaring `Multiaddress` ports.

/// Errors surfaced by `AddressBook` mutation methods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AddressBookError {
    /// `register_address` / `forget_address` / `drop_peer` called
    /// for a `PeerId` with no entry in the book.
    UnknownPeer(PeerId),
    /// `add_peer` called with an empty address vector. An entry
    /// with zero addresses can't be looked up successfully, so
    /// creating one is meaningless - reject up front.
    EmptyAddressList,
    /// `add_peer` for a NEW peer when the book is already at its
    /// configured cap. Adversarial peer-discovery floods can't
    /// grow the book without bound.
    Full {
        /// Current cap.
        cap: usize,
    },
    /// `add_peer`'s internal dedup buffer could not be reserved.
    /// `Vec::try_reserve_exact` returned `TryReserveError` - the
    /// host's allocator has no headroom for `requested` addresses.
    /// The engine maps this to `WireReceiveErrorKind::AllocationFailed`
    /// so the receiver-side address-book hint is best-effort under
    /// allocator pressure (the envelope still routes).
    AllocationFailed {
        /// Address count the dedup buffer attempted to reserve.
        requested: usize,
    },
}

impl fmt::Display for AddressBookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressBookError::UnknownPeer(p) => {
                write!(f, "AddressBook: peer {p} not registered")
            }
            AddressBookError::EmptyAddressList => {
                write!(f, "AddressBook: add_peer requires a non-empty address list")
            }
            AddressBookError::Full { cap } => {
                write!(f, "AddressBook: at cap {cap}, new peer rejected")
            }
            AddressBookError::AllocationFailed { requested } => {
                write!(
                    f,
                    "AddressBook: dedup reservation for {requested} addresses failed"
                )
            }
        }
    }
}

impl std::error::Error for AddressBookError {}

/// Per-peer storage: ordered address list + reference count.
struct AddressEntry {
    addresses: Vec<Address>,
    ref_count: u64,
}

/// Default cap on tracked peer entries. Adversarial peer-discovery
/// floods (gossip announcements, peer-list responses, multi-address
/// chatter) would otherwise grow `entries` without bound.
pub const DEFAULT_ADDRESS_BOOK_CAP: usize = 16_384;

/// Ref-counted `PeerId → Vec<Address>` registry. Single source of
/// truth for "where can I reach this peer."
pub struct AddressBook {
    entries: HashMap<PeerId, AddressEntry>,
    /// Maximum permitted entry count. `add_peer` for a NEW peer
    /// fails with `AddressBookError::Full` once `entries.len() >=
    /// cap`. Tunable via [`Self::set_cap`].
    cap: usize,
}

impl Default for AddressBook {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            cap: DEFAULT_ADDRESS_BOOK_CAP,
        }
    }
}

impl AddressBook {
    /// Fresh, empty address book.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the entry cap. Production hosts can match the cap
    /// to their topology size (a small cluster + 2× headroom).
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
    }

    /// Announce a peer with one or more addresses.
    ///
    /// - **New peer** → entry created with `ref_count = 1`,
    ///   addresses inserted in the given order.
    /// - **Known peer** → `ref_count += 1`. New addresses appended;
    ///   duplicates dropped, existing entries preserve their
    ///   position (first caller's preference wins).
    ///
    /// Errors with [`AddressBookError::EmptyAddressList`] if
    /// `addresses` is empty.
    ///
    /// Components MUST pair every `add_peer` with an eventual
    /// [`Self::drop_peer`] so `ref_count` converges to zero.
    pub fn add_peer(
        &mut self,
        peer: PeerId,
        addresses: Vec<Address>,
    ) -> Result<(), AddressBookError> {
        if addresses.is_empty() {
            return Err(AddressBookError::EmptyAddressList);
        }
        match self.entries.get_mut(&peer) {
            Some(entry) => {
                entry.ref_count = entry.ref_count.saturating_add(1);
                for addr in addresses {
                    if !entry.addresses.contains(&addr) {
                        entry.addresses.push(addr);
                    }
                }
            }
            None => {
                if self.entries.len() >= self.cap {
                    return Err(AddressBookError::Full { cap: self.cap });
                }
                let mut dedup: Vec<Address> = Vec::new();
                crate::fallible::try_reserve_exact(&mut dedup, addresses.len()).map_err(|_| {
                    AddressBookError::AllocationFailed {
                        requested: addresses.len(),
                    }
                })?;
                for addr in addresses {
                    if !dedup.contains(&addr) {
                        dedup.push(addr);
                    }
                }
                self.entries.insert(
                    peer,
                    AddressEntry {
                        addresses: dedup,
                        ref_count: 1,
                    },
                );
            }
        }
        Ok(())
    }

    /// Release one reference to `peer`. Decrements `ref_count`;
    /// removes the entry (and all addresses) when the count
    /// reaches zero. Errors with
    /// [`AddressBookError::UnknownPeer`] if no entry exists.
    pub fn drop_peer(&mut self, peer: PeerId) -> Result<(), AddressBookError> {
        let Some(entry) = self.entries.get_mut(&peer) else {
            return Err(AddressBookError::UnknownPeer(peer));
        };
        entry.ref_count = entry.ref_count.saturating_sub(1);
        if entry.ref_count == 0 {
            self.entries.remove(&peer);
        }
        Ok(())
    }

    /// Append `address` to the peer's list. Idempotent - duplicates
    /// are dropped. Does NOT change `ref_count`. Errors with
    /// [`AddressBookError::UnknownPeer`] if the peer has no entry.
    pub fn register_address(
        &mut self,
        peer: PeerId,
        address: Address,
    ) -> Result<(), AddressBookError> {
        let Some(entry) = self.entries.get_mut(&peer) else {
            return Err(AddressBookError::UnknownPeer(peer));
        };
        if !entry.addresses.contains(&address) {
            entry.addresses.push(address);
        }
        Ok(())
    }

    /// Prune one unreachable `address` from the peer's list.
    /// Transport adapters call this after observing a
    /// transport-level failure on a specific address. Does NOT
    /// change `ref_count` and does NOT remove the entry even if
    /// pruning leaves the address list empty - [`Self::drop_peer`]
    /// is the only path that removes entries. Errors with
    /// [`AddressBookError::UnknownPeer`] if no entry exists.
    pub fn forget_address(
        &mut self,
        peer: PeerId,
        address: &Address,
    ) -> Result<(), AddressBookError> {
        let Some(entry) = self.entries.get_mut(&peer) else {
            return Err(AddressBookError::UnknownPeer(peer));
        };
        entry.addresses.retain(|a| a != address);
        Ok(())
    }

    /// Ordered slice of every address bound to `peer`. Returns
    /// `None` for an unknown peer OR a peer whose address list is
    /// empty (e.g. all addresses pruned via `forget_address` and
    /// nothing re-added). Both cases mean "can't route" to the
    /// wire syscall.
    pub fn lookup(&self, peer: PeerId) -> Option<&[Address]> {
        let entry = self.entries.get(&peer)?;
        if entry.addresses.is_empty() {
            None
        } else {
            Some(entry.addresses.as_slice())
        }
    }

    /// Convenience: the first (highest-preference) address.
    pub fn lookup_first(&self, peer: PeerId) -> Option<&Address> {
        self.lookup(peer).and_then(|addrs| addrs.first())
    }

    /// Current reference count for `peer`, or `0` if unregistered.
    pub fn ref_count(&self, peer: PeerId) -> u64 {
        self.entries.get(&peer).map(|e| e.ref_count).unwrap_or(0)
    }

    /// Number of registered peers (regardless of ref_count).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no peers are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate `(peer, &[Address], ref_count)` triples for every
    /// registered peer. Used by snapshot capture.
    pub fn iter(&self) -> impl Iterator<Item = (PeerId, &[Address], u64)> {
        self.entries
            .iter()
            .map(|(p, e)| (*p, e.addresses.as_slice(), e.ref_count))
    }

    /// Snapshot-restore setter - reconstructs an entry with the
    /// recorded addresses + ref_count without going through
    /// [`Self::add_peer`] (which would force `ref_count = 1`).
    /// Used exclusively by `Node::restore`.
    pub fn restore_entry(&mut self, peer: PeerId, addresses: Vec<Address>, ref_count: u64) {
        self.entries.insert(
            peer,
            AddressEntry {
                addresses,
                ref_count,
            },
        );
    }
}

