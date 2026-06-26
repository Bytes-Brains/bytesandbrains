//! `GlobalRegistry` — federation membership protocol with two halves.
//!
//! - [`GlobalRegistryClient`]: bound on every client Node. Records an
//!   `Announce` op whose `server_peer` input wires the bootstrap-server
//!   identity through the graph (no struct field). On dispatch the
//!   client ships an envelope carrying `ctx.current.self_peer` to the
//!   server's well-known [`GLOBAL_REGISTRY_SERVER_CREF`] component, then
//!   refreshes its TTL/heartbeat state from the server's `Handshake`
//!   reply.
//!
//! - [`GlobalRegistryServer`]: bound on the server Node. Accepts
//!   inbound `Announce` envelopes, registers the announcing peer in
//!   the runtime [`AddressBook`](bb_runtime::framework::AddressBook)
//!   under a server-assigned TTL, and replies with a `Handshake`
//!   carrying `(assigned_ttl_ns, heartbeat_interval_ns)`. Lazy
//!   eviction runs at the top of every `Sample` / `CurrentView` read.
//!
//! The protocol carries no static peer identity on either side: the
//! client wires `server_peer` via graph input, and the server reads
//! the announcing peer from the inbound payload.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use bb_runtime::atomic::{AtomicOpDecl, AtomicOpKind, AtomicOpsetDecl, DispatchResult};
use bb_runtime::bus::OpError;
use bb_runtime::completion::{CompletionHandle, ContractResponse};
use bb_runtime::envelope::{SlotFill, WireEnvelope};
use bb_runtime::framework::Address;
use bb_runtime::ids::{ComponentRef, PeerId};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::BytesValue;

use bb_ir::types::{TYPE_BYTES, TYPE_PEER_ID, TYPE_PEER_ID_VEC, TYPE_SCALAR_I32, TYPE_TRIGGER};

/// Well-known `ComponentRef` the client addresses the server's
/// [`GlobalRegistryServer`] component at. Pinned on both sides; a
/// production deployment would resolve via discovery.
pub const GLOBAL_REGISTRY_SERVER_CREF: u32 = 0;

/// Well-known `ComponentRef` the server addresses the client's
/// [`GlobalRegistryClient`] component at when delivering Handshake
/// replies. Pinned on both sides.
pub const GLOBAL_REGISTRY_CLIENT_CREF: u32 = 1;

/// Atomic-op opset domain shared by both halves.
pub const GLOBAL_REGISTRY_DOMAIN: &str = "ai.bytesandbrains.protocol.global_registry";

/// Handshake payload the server replies with on a successful
/// `Announce`. The client decodes TTL/heartbeat state and merges
/// `server_addresses` into its [`AddressBook`] entry for the server
/// peer so subsequent dial attempts can pick any reachable endpoint.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handshake {
    /// TTL the server assigned this client. The client must
    /// re-`Announce` before this elapses or the server lazily evicts
    /// the entry on its next `Sample` / `CurrentView` read.
    pub assigned_ttl_ns: u64,
    /// Server-derived heartbeat interval (`assigned_ttl_ns / 3`).
    /// The client throttles outgoing `Announce` calls to no more
    /// than one per interval.
    pub heartbeat_interval_ns: u64,
    /// Full multi-address bag the server advertises for itself. The
    /// client lands these in its [`AddressBook`] entry for the
    /// server peer so the dialer can pick any reachable endpoint.
    /// Empty means the server failed to populate its own
    /// `local_addresses()` — a deployment error the server rejects
    /// before reaching this struct.
    pub server_addresses: Vec<Address>,
}

// ─── Client ────────────────────────────────────────────────────────

/// Client half of GlobalRegistry. Holds TTL / heartbeat state echoed
/// from the server's most recent Handshake; the server peer is wired
/// through the graph at every `Announce` and is not a struct field.
#[derive(Clone, Debug, Default, Serialize, Deserialize, bb_derive::Concrete)]
pub struct GlobalRegistryClient {
    /// TTL the server assigned in its last Handshake response.
    /// Stamped into outgoing `Announce` envelopes as
    /// `remaining_deadline_ns` so receivers can age out stale entries
    /// without per-peer state.
    pub last_assigned_ttl_ns: u64,

    /// Heartbeat interval the server computed in its last Handshake
    /// (`assigned_ttl_ns / 3`). The client respects this interval when
    /// throttling subsequent Announces.
    pub last_heartbeat_interval_ns: u64,

    /// Monotonic-ns timestamp of the most recent Announce dispatch.
    /// Skipped on snapshot/restore so bootstrap re-seeds the cadence
    /// on resume.
    #[serde(skip)]
    pub last_announce_ts_ns: u64,
}

impl GlobalRegistryClient {
    /// Construct with default TTL state. The server peer is wired
    /// through the graph at every `Announce` op and is not a field.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Atomic-op declarations for the client half. `Announce` carries
/// the server peer as a graph input; `Handshake` delivers the
/// server's TTL/heartbeat reply back into the client.
static GLOBAL_REGISTRY_CLIENT_OPS: &[AtomicOpDecl] = &[
    AtomicOpDecl {
        name: "Announce",
        inputs: &[("server_peer", &TYPE_PEER_ID)],
        outputs: &[("wakeup", &TYPE_TRIGGER)],
        kind: AtomicOpKind::Immediate,
        type_relations: &[],
    },
    AtomicOpDecl {
        name: "Handshake",
        inputs: &[],
        outputs: &[("wakeup", &TYPE_TRIGGER)],
        kind: AtomicOpKind::Immediate,
        type_relations: &[],
    },
];

impl bb_runtime::roles::ProtocolRuntime for GlobalRegistryClient {
    type Error = OpError;

    fn atomic_opset(&self) -> AtomicOpsetDecl {
        AtomicOpsetDecl {
            domain: GLOBAL_REGISTRY_DOMAIN,
            version: 1,
            ops: GLOBAL_REGISTRY_CLIENT_OPS,
        }
    }

    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, OpError> {
        match op_type {
            "Announce" => {
                let now = ctx.time.scheduler.now_ns();
                // Heartbeat throttle. Sub-interval calls are silent
                // no-ops; the first Announce always fires because
                // both `last_announce_ts_ns` and
                // `last_heartbeat_interval_ns` start at zero.
                if self.last_announce_ts_ns != 0
                    && self.last_heartbeat_interval_ns != 0
                    && now.saturating_sub(self.last_announce_ts_ns)
                        < self.last_heartbeat_interval_ns
                {
                    return Ok(DispatchResult::Immediate(Vec::new()));
                }

                let server_peer = downcast_peer_id(inputs, "server_peer")?;

                // The Announce payload carries the client's full
                // address bag so the server's AddressBook learns
                // every reachable endpoint on first contact. No
                // synthesis fallback: a client with no local
                // addresses is a deployment error caught at the
                // bottom of the bootstrap path, not papered over
                // with a `/p2p/<PeerId>` placeholder here.
                let local_addresses = ctx.local_addresses().to_vec();
                if local_addresses.is_empty() {
                    return Err(OpError {
                        detail: "GlobalRegistryClient::Announce: no local addresses; \
                                 configure via install(...) or node.add_local_address()"
                            .to_string(),
                        ..Default::default()
                    });
                }
                let payload = bincode::serialize(&(ctx.current.self_peer, local_addresses))
                    .map_err(|e| OpError {
                        detail: format!("Announce: serialize (self_peer, addresses): {e}"),
                        ..Default::default()
                    })?;

                let dest_suffix = Address::empty()
                    .component(ComponentRef::from(GLOBAL_REGISTRY_SERVER_CREF))
                    .op("Announce")
                    .to_bytes();
                let dest_peer_addr = Address::empty().p2p(server_peer).to_bytes();

                let env = WireEnvelope {
                    dest_peer_addresses: vec![dest_peer_addr],
                    fills: vec![SlotFill {
                        dest_suffix,
                        payload,
                        trigger_only: false,
                        ..Default::default()
                    }],
                    correlation: None,
                    remaining_deadline_ns: self.last_assigned_ttl_ns,
                    edge_rtt_reports: Vec::new(),
                    ..Default::default()
                };
                ctx.net.outbound.push(env);

                self.last_announce_ts_ns = now;
                Ok(DispatchResult::Immediate(Vec::new()))
            }
            "Handshake" => {
                let payload = inputs
                    .iter()
                    .find_map(|(_, v)| v.as_any().downcast_ref::<BytesValue>().map(|b| b.0.clone()))
                    .ok_or_else(|| OpError {
                        detail: "Handshake: missing BytesValue payload".to_string(),
                        ..Default::default()
                    })?;
                let handshake: Handshake = bincode::deserialize(&payload).map_err(|e| OpError {
                    detail: format!("Handshake: decode: {e}"),
                    ..Default::default()
                })?;
                self.last_assigned_ttl_ns = handshake.assigned_ttl_ns;
                self.last_heartbeat_interval_ns = handshake.heartbeat_interval_ns;

                // Merge the server's advertised address bag into the
                // client's AddressBook. A handshake with an empty
                // `server_addresses` is malformed (the server's
                // dispatch rejects that case before serializing) so
                // a defensive skip beats a crash here. AddressBook
                // failures surface as a bus event rather than a
                // hard return so a transient cap collision cannot
                // tip the client into a fatal-error spiral.
                if !handshake.server_addresses.is_empty() {
                    if let Some(server_peer) = ctx.current.inbound.src_peer {
                        if let Err(e) = ctx
                            .peers
                            .addresses
                            .add_peer(server_peer, handshake.server_addresses)
                        {
                            ctx.bus.publish(bb_runtime::bus::NodeEvent::Infra(
                                bb_runtime::bus::InfraEvent::OpFailure {
                                    op_ref: ctx.current.op_ref,
                                    error: OpError {
                                        detail: format!(
                                            "Handshake: address_book.add_peer({server_peer:?}): {e}"
                                        ),
                                        ..Default::default()
                                    },
                                },
                            ));
                        }
                    }
                }
                Ok(DispatchResult::Immediate(Vec::new()))
            }
            other => Err(OpError {
                detail: format!("unknown op for GlobalRegistryClient: {other}"),
                ..Default::default()
            }),
        }
    }
}

// ─── Server ────────────────────────────────────────────────────────

/// Per-deployment knobs for [`GlobalRegistryServer`]. All durations
/// are in nanoseconds. Defaults follow the deep-research bound:
/// 90 s TTL, 30 s floor, 5 min ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalRegistryServerConfig {
    /// TTL the server stamps onto fresh registrations. Clients
    /// receive this verbatim in their Handshake reply.
    pub default_ttl_ns: u64,
    /// Floor for aggressive eviction policy bumps. Not enforced
    /// directly today; reserved as the documented lower bound for
    /// future server-side cohort tuning.
    pub min_ttl_ns: u64,
    /// Ceiling on stale entries. Not enforced directly today;
    /// reserved as the documented upper bound for future
    /// server-side cohort tuning.
    pub max_ttl_ns: u64,
}

impl Default for GlobalRegistryServerConfig {
    fn default() -> Self {
        Self {
            default_ttl_ns: 90_000_000_000,
            min_ttl_ns: 30_000_000_000,
            max_ttl_ns: 300_000_000_000,
        }
    }
}

/// Server half of GlobalRegistry. Maintains
/// `PeerId → (expires_at_ns, source_address)` over the announced
/// cohort and exposes the cohort as a
/// [`bb_runtime::contracts::PeerSelector`] source. Eviction is lazy:
/// `Sample` / `CurrentView` drop entries whose `expires_at_ns` has
/// elapsed before consulting the registry.
#[derive(Debug, Serialize, Deserialize, bb_derive::Concrete, bb_derive::PeerSelector)]
pub struct GlobalRegistryServer {
    /// Configurable TTL bounds + heartbeat policy.
    pub config: GlobalRegistryServerConfig,

    /// RNG seed for deterministic `Sample` selection. Persisted so a
    /// restored server samples consistently.
    pub seed: u64,

    /// Registry: `PeerId → (expires_at_ns, source_address)`. Restored
    /// alongside `seed` so cohort continuity survives snapshot/restore.
    pub entries: HashMap<PeerId, (u64, Address)>,

    /// Per-call counter the seedable RNG mixes in so successive
    /// `sample(n)` returns vary even on a constant seed. Reset to
    /// zero on snapshot/restore — restored servers replay sampling
    /// from a known starting point. `AtomicU64` is the minimal
    /// lock-free shape that fits the Contract's `&self`-only `select`
    /// signature.
    #[serde(skip)]
    sample_counter: AtomicU64,
}

impl Default for GlobalRegistryServer {
    fn default() -> Self {
        Self {
            config: GlobalRegistryServerConfig::default(),
            seed: 0,
            entries: HashMap::new(),
            sample_counter: AtomicU64::new(0),
        }
    }
}

impl Clone for GlobalRegistryServer {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            seed: self.seed,
            entries: self.entries.clone(),
            sample_counter: AtomicU64::new(0),
        }
    }
}

impl GlobalRegistryServer {
    /// Construct a fresh server with the default TTL/heartbeat policy
    /// and `seed` driving deterministic `Sample` selection.
    pub fn new(seed: u64) -> Self {
        Self {
            config: GlobalRegistryServerConfig::default(),
            seed,
            entries: HashMap::new(),
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Construct with an explicit `config` override.
    pub fn with_config(seed: u64, config: GlobalRegistryServerConfig) -> Self {
        Self {
            config,
            seed,
            entries: HashMap::new(),
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Server-derived heartbeat interval. Integer division on
    /// `default_ttl_ns / 3` gives the client three windows to refresh
    /// before eviction.
    pub fn heartbeat_interval_ns(&self) -> u64 {
        self.config.default_ttl_ns / 3
    }

    /// Drop registry entries whose `expires_at_ns < now_ns` and
    /// release their address-book references. Called at the top of
    /// every `Sample` / `CurrentView` read.
    fn evict_expired(&mut self, now_ns: u64, addresses: &mut bb_runtime::framework::AddressBook) {
        let expired: Vec<PeerId> = self
            .entries
            .iter()
            .filter_map(|(peer, (expires, _))| (now_ns >= *expires).then_some(*peer))
            .collect();
        for peer in expired {
            self.entries.remove(&peer);
            let _ = addresses.drop_peer(peer);
        }
    }

    /// Live cohort (post-eviction) as a `Vec<PeerId>` in insertion
    /// order. Helper used by both `Sample` and `CurrentView`.
    fn live_peers(&self) -> Vec<PeerId> {
        self.entries.keys().copied().collect()
    }
}

impl bb_runtime::contracts::PeerSelector for GlobalRegistryServer {
    type Error = OpError;

    fn select(
        &mut self,
        ctx: &mut bb_runtime::runtime::RuntimeResourceRef<'_>,
        params: bb_runtime::contracts::peer_selector::SelectParams,
        _completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error> {
        use bb_runtime::contracts::peer_selector::SelectParams;
        let now = ctx.time.scheduler.now_ns();
        self.evict_expired(now, ctx.peers.addresses);
        let known = self.live_peers();
        let out = match params {
            SelectParams::All => known,
            SelectParams::Random { n } => {
                sample_n(&known, n as usize, self.seed, &self.sample_counter)
            }
            SelectParams::NearKey { key: _, n } => {
                let take = (n as usize).min(known.len());
                known[..take].to_vec()
            }
        };
        ContractResponse::Now(Ok(out))
    }

    fn current_view(
        &mut self,
        ctx: &mut bb_runtime::runtime::RuntimeResourceRef<'_>,
        _completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error> {
        let now = ctx.time.scheduler.now_ns();
        self.evict_expired(now, ctx.peers.addresses);
        ContractResponse::Now(Ok(self.live_peers()))
    }
}

/// Atomic-op declarations for the server half. `Sample` and
/// `CurrentView` carry an opaque cookie (libp2p-style incremental
/// discovery). v1 ships full cookies — `next_cookie` is reserved
/// state the client echoes back unchanged on the next read.
static GLOBAL_REGISTRY_SERVER_OPS: &[AtomicOpDecl] = &[
    AtomicOpDecl {
        name: "Sample",
        inputs: &[("count", &TYPE_SCALAR_I32), ("cookie", &TYPE_BYTES)],
        outputs: &[("peers", &TYPE_PEER_ID_VEC), ("next_cookie", &TYPE_BYTES)],
        kind: AtomicOpKind::Immediate,
        type_relations: &[],
    },
    AtomicOpDecl {
        name: "CurrentView",
        inputs: &[("cookie", &TYPE_BYTES)],
        outputs: &[("peers", &TYPE_PEER_ID_VEC), ("next_cookie", &TYPE_BYTES)],
        kind: AtomicOpKind::Immediate,
        type_relations: &[],
    },
    AtomicOpDecl {
        name: "Announce",
        inputs: &[],
        outputs: &[],
        kind: AtomicOpKind::Immediate,
        type_relations: &[],
    },
];

impl bb_runtime::roles::ProtocolRuntime for GlobalRegistryServer {
    type Error = OpError;

    fn atomic_opset(&self) -> AtomicOpsetDecl {
        AtomicOpsetDecl {
            domain: GLOBAL_REGISTRY_DOMAIN,
            version: 1,
            ops: GLOBAL_REGISTRY_SERVER_OPS,
        }
    }

    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, OpError> {
        match op_type {
            "Announce" => {
                let payload = inputs
                    .iter()
                    .find_map(|(_, v)| v.as_any().downcast_ref::<BytesValue>().map(|b| b.0.clone()))
                    .ok_or_else(|| OpError {
                        detail: "Announce: missing BytesValue payload".to_string(),
                        ..Default::default()
                    })?;
                let (announcing_peer, announced_addresses): (PeerId, Vec<Address>) =
                    bincode::deserialize(&payload).map_err(|e| OpError {
                        detail: format!("Announce: decode (peer, addresses): {e}"),
                        ..Default::default()
                    })?;
                if announced_addresses.is_empty() {
                    return Err(OpError {
                        detail:
                            "GlobalRegistryServer::Announce: client supplied empty address list"
                                .to_string(),
                        ..Default::default()
                    });
                }

                let now = ctx.time.scheduler.now_ns();
                let ttl = self.config.default_ttl_ns;
                let heartbeat = ttl / 3;
                // First-address-wins for the registry's source-address
                // slot. The full bag still lands in the AddressBook
                // for dial fan-out.
                let source_addr = announced_addresses[0].clone();

                // Idempotent registration: a re-Announce from a known
                // peer overwrites `expires_at_ns` without bumping the
                // address-book `ref_count` (add_peer is a no-op for
                // duplicate addresses on an existing entry, and the
                // ref_count bump is what would otherwise leak).
                let is_new = !self.entries.contains_key(&announcing_peer);
                self.entries.insert(
                    announcing_peer,
                    (now.saturating_add(ttl), source_addr.clone()),
                );
                if is_new {
                    ctx.peers
                        .addresses
                        .add_peer(announcing_peer, announced_addresses)
                        .map_err(|e| OpError {
                            detail: format!("Announce: address_book.add_peer: {e}"),
                            ..Default::default()
                        })?;
                }

                // Server's own addresses ride back on the Handshake
                // so the client's dialer can pick any reachable
                // endpoint. No synthesis fallback.
                let server_addresses = ctx.local_addresses().to_vec();
                if server_addresses.is_empty() {
                    return Err(OpError {
                        detail: "GlobalRegistryServer::Announce: no local addresses to advertise; \
                                 configure via install(...) or node.add_local_address()"
                            .to_string(),
                        ..Default::default()
                    });
                }
                let handshake = Handshake {
                    assigned_ttl_ns: ttl,
                    heartbeat_interval_ns: heartbeat,
                    server_addresses,
                };
                let handshake_payload = bincode::serialize(&handshake).map_err(|e| OpError {
                    detail: format!("Announce: serialize handshake: {e}"),
                    ..Default::default()
                })?;
                let reply_suffix = Address::empty()
                    .component(ComponentRef::from(GLOBAL_REGISTRY_CLIENT_CREF))
                    .op("Handshake")
                    .to_bytes();
                let reply_env = WireEnvelope {
                    dest_peer_addresses: vec![Address::empty().p2p(announcing_peer).to_bytes()],
                    fills: vec![SlotFill {
                        dest_suffix: reply_suffix,
                        payload: handshake_payload,
                        trigger_only: false,
                        ..Default::default()
                    }],
                    correlation: None,
                    remaining_deadline_ns: 0,
                    edge_rtt_reports: Vec::new(),
                    ..Default::default()
                };
                ctx.net.outbound.push(reply_env);

                Ok(DispatchResult::Immediate(Vec::new()))
            }
            "Sample" => {
                let now = ctx.time.scheduler.now_ns();
                self.evict_expired(now, ctx.peers.addresses);
                let n = inputs
                    .iter()
                    .find_map(|(name, v)| {
                        (*name == "count").then(|| v.as_any().downcast_ref::<u32>().copied())
                    })
                    .flatten()
                    .unwrap_or(0) as usize;
                let known = self.live_peers();
                let picked = sample_n(&known, n, self.seed, &self.sample_counter);
                let next_cookie = next_cookie_from(inputs);
                Ok(DispatchResult::Immediate(vec![
                    ("peers".to_string(), Box::new(picked) as Box<dyn SlotValue>),
                    (
                        "next_cookie".to_string(),
                        Box::new(BytesValue(next_cookie)) as Box<dyn SlotValue>,
                    ),
                ]))
            }
            "CurrentView" => {
                let now = ctx.time.scheduler.now_ns();
                self.evict_expired(now, ctx.peers.addresses);
                let view = self.live_peers();
                let next_cookie = next_cookie_from(inputs);
                Ok(DispatchResult::Immediate(vec![
                    ("peers".to_string(), Box::new(view) as Box<dyn SlotValue>),
                    (
                        "next_cookie".to_string(),
                        Box::new(BytesValue(next_cookie)) as Box<dyn SlotValue>,
                    ),
                ]))
            }
            other => Err(OpError {
                detail: format!("unknown op for GlobalRegistryServer: {other}"),
                ..Default::default()
            }),
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────

/// Downcast `inputs[name]` to a [`PeerId`]. Handles both the typed
/// `PeerIdValue` carrier and a bare `PeerId` (constant ops route the
/// raw value when no carrier is required).
fn downcast_peer_id(inputs: &[(&str, &dyn SlotValue)], name: &str) -> Result<PeerId, OpError> {
    for (slot, v) in inputs {
        if *slot != name {
            continue;
        }
        if let Some(p) = v.as_any().downcast_ref::<PeerId>() {
            return Ok(*p);
        }
        if let Some(pv) = v
            .as_any()
            .downcast_ref::<bb_runtime::syscall::values::PeerIdValue>()
        {
            return Ok(pv.0);
        }
    }
    Err(OpError {
        detail: format!("missing `{name}` input (expected PeerId)"),
        ..Default::default()
    })
}

/// Read the inbound `cookie` (opaque bytes) and echo it back as the
/// `next_cookie`. v1 ships full cookies — pagination state stays
/// reserved so the surface is stable when v2 adds chunked discovery.
fn next_cookie_from(inputs: &[(&str, &dyn SlotValue)]) -> Vec<u8> {
    for (slot, v) in inputs {
        if *slot != "cookie" {
            continue;
        }
        if let Some(b) = v.as_any().downcast_ref::<BytesValue>() {
            return b.0.clone();
        }
    }
    Vec::new()
}

/// Deterministic `n`-subset of `peers` driven by an xorshift mixed
/// with `seed` and a per-call counter. Empty input yields an empty
/// result.
fn sample_n(peers: &[PeerId], n: usize, seed: u64, counter: &AtomicU64) -> Vec<PeerId> {
    if peers.is_empty() || n == 0 {
        return Vec::new();
    }
    let take = n.min(peers.len());
    let count = counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(count);
    let mut pool: Vec<PeerId> = peers.to_vec();
    for i in 0..take {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = i + (state as usize) % (pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(take);
    pool
}

