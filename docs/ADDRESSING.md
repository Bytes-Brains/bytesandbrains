# ADDRESSING.md — Addresses are Multiaddrs, and addresses route themselves

## Premise

In bytes-and-brains, an **address is a multiaddr** — a sequence of typed
protocol segments. The multiaddr fully describes the delivery path, so the
receiver does not consult any per-message subscription table, opset routing
map, or `recv_sites`-style HashMap. The address itself is the routing.

Sender:
1. Builds the destination multiaddr by walking the graph: the producer's
   compiler pass knows which Recv-op site or which Component-op the value
   needs to land at.
2. Packs each typed input into a `SlotFill` with its per-slot multiaddr
   suffix.
3. Resolves the destination `PeerId` to its ordered address list via
   the framework's `AddressBook` and packs that list into
   `WireEnvelope.dest_peer_addresses`. Sends one envelope per
   `(producer_op, dest_peer_addresses_first)` group.
4. Snapshots its own local-address bag (`ctx.local_addresses()` at
   `bb-runtime/src/runtime.rs:269`) and stamps it onto
   `WireEnvelope.src_peer_addresses` (`bb-ops/src/network/wire/mod.rs:196-220`)
   so the receiver learns every interface the sender currently binds.

Receiver:
1. Parses each fill's `dest_suffix`.
2. The suffix's trailing segments tell the receiver exactly which slot
   or component-op to deliver to.
3. Dispatches directly — no subscription lookup.
4. Merges `envelope.src_peer_addresses` and the transport-observed
   `IngressEvent::EnvelopeFrom.src_observed_address` into the
   receiver's AddressBook entry for the source peer
   (`bb-runtime/src/engine/poll.rs:1005-1062`). Claimed addresses land
   first; observed wins for the NAT-translated case the sender's
   snapshot cannot know.

The DAG / IR / DSL still key destinations by `PeerId`. The wire
syscall is the resolution boundary: it consults the
[`AddressBook`](#address-book-semantics) at dispatch time, packs the
resolved list into the envelope, and surfaces
[`EngineStep::PeerResolveFailed`](#peer-resolution-failure) on miss.

## Protocol enum

The framework carries the per-slot `dest_suffix` in each `SlotFill`
for intra-node routing AND the resolved ordered address list in
`WireEnvelope.dest_peer_addresses` for transport selection. Peer
resolution (`PeerId → Vec<Address>`) happens inside the wire syscall
via the `AddressBook`; the host's transport adapter just picks one
of the supplied addresses based on its networking capabilities.

```rust
pub enum Protocol {
    P2p(PeerId),            // varint 421 — libp2p-standard /p2p/<multihash>
    Site(NodeSiteId),       // varint 0xE2 — data-plane slot fill target
    Component(ComponentRef),// varint 0xE3 — control-plane component
    Op(String),             // varint 0xE4 — control-plane op discriminator
}
```

`NodeSiteId` is globally unique within a Node (allocated from a
global `AtomicU64` counter), so `/site/<id>` uniquely identifies a
data-plane slot. `Component + Op` together identify a control-plane
dispatch target.

### Why so narrow?

The engine's routing dispatch (`Engine::deliver_fill` in
`src/engine/poll.rs`) ONLY matches on `addr.site_id()` and
`addr.component_ref() + addr.op_name()`. Every other multiaddr
segment was historically supported for libp2p round-trip symmetry
(Ip4/Ip6/Tcp/Udp), logs (Graph), or test fixtures (Memory) — none
of which the framework ever read for routing. They were
maintenance burden with zero internal benefit.

**Host adapter responsibility.** When a host (libp2p, in-process
sim, anything else) ships an outbound envelope, it reads
`dest_peer_addresses` (a `repeated bytes` already resolved by the
wire syscall via the `AddressBook`) and picks one entry based on
its networking capabilities (IPv4 reachability, QUIC support,
relay preference, etc.). The adapter then constructs whatever
transport-level bytes it needs and ships the envelope. On receipt,
the adapter passes the envelope into the framework along with the
source `PeerId` (via `IngressEvent::EnvelopeFrom { src_peer,
envelope }`) — the framework dispatches purely by `dest_suffix` on
each fill, ignoring `dest_peer_addresses` on the inbound side.

## Binary encoding

`Address` encodes as `varint(code) || payload`. Protocol codes are
unsigned LEB128 varints — matches libp2p's multiaddr wire format.
- `/p2p/`: payload is `varint(len) || multihash bytes`, identical
  to libp2p byte-for-byte. The multihash itself carries an
  algorithm code (identity, sha2-256, sha3, blake2b, …) so
  different overlay protocols pick their own digest function
  without coordination.
- `/site/`: payload is a fixed 8-byte big-endian `NodeSiteId`.
- `/component/`: payload is a fixed 4-byte big-endian
  `ComponentRef`.
- `/op/`: payload is `varint(len) || utf8 bytes`.

Concatenated segments form the full byte sequence. Empty Address
= empty buffer.

`Address::from_bytes` rejects any varint code outside the
framework's set with `AddressError::UnknownCode`. Standard libp2p
transport codes (`ip4 = 4`, `tcp = 6`, etc.) fail to parse —
they're host-adapter concerns that should never reach the
framework. The `/p2p/` code (421) is recognized so a multiaddr
suffix produced by libp2p's `Multiaddr` round-trips through our
codec at the wire level.

## String form

`/protocol/value/protocol/value/...` — segments separated by `/`,
each protocol's value parsed per its type. The `/p2p/<peerid>`
form uses base58btc-encoded multihash bytes, matching libp2p's
`Qm…` / `12D3…` peer-id strings exactly:

```
/p2p/12D3KooWBh.../site/17
    → data-plane fill on peer 12D3KooWBh..., slot 17

/p2p/12D3KooWBh.../component/7/op/FindNode
    → control-plane component-7's "FindNode" op on peer 12D3KooWBh...
```

`Address::from_str` parses the string form; `Display` round-trips
back. Unknown protocols (including `ip4`, `tcp`, `memory`, etc.)
return `AddressError::InvalidValue`.

## Wire envelope

The wire envelope ships one **resolved destination address list**,
the sender's **claimed local-address bag**, and one or more
`SlotFill`s, batched per the compiler's `analyze_wire_edges` pass.

```proto
message WireEnvelope {
  // Ordered destination address list — the framework's wire
  // syscall populates this from the AddressBook at dispatch time.
  // The host's transport adapter picks one based on its
  // networking capabilities. Each entry is `Address::to_bytes()`.
  repeated bytes dest_peer_addresses = 1;

  repeated SlotFill fills = 2;
  WireCorrelation correlation = 3;

  // ... deadline + RTT piggyback + src_peer_bytes + schema_version ...

  // Sender-claimed local-address bag — snapshot of the sender's
  // AddressBook entry for its own PeerId at envelope-mint time.
  // Receiver merges into its own AddressBook entry for the sender
  // (see Receiver-side address ingest). Each entry is
  // `Address::to_bytes()`; bounded at decode time by
  // `EnvelopeCaps.max_src_peer_addresses` +
  // `max_src_peer_address_bytes`.
  repeated bytes src_peer_addresses = 8;
}

message SlotFill {
  bytes  dest_suffix = 1;   // per-slot multiaddr suffix (intra-node)
  bytes  payload     = 2;
  bool   trigger_only = 3;
}
```

Peer routing and intra-node slot routing are decoupled:

- **`dest_peer_addresses`** is the resolved snapshot of
  `AddressBook::lookup(peer)` at dispatch time — an ordered slice
  by peer-stated preference. The host's transport adapter picks
  one of the entries based on its networking capabilities (IPv4
  reachability, QUIC support, relay preference, etc.). The DAG /
  IR / DSL still address destinations by `PeerId` — only the
  envelope-on-the-wire carries the resolved list.
- **`src_peer_addresses`** (proto field 8, `bb-ir/proto/bb_core.proto:135`)
  is the snapshot of the sender's `local_addresses()` at envelope-mint
  time. The wire syscall stamps it on every Send
  (`bb-ops/src/network/wire/mod.rs:196-220`); receivers feed it into
  the AddressBook merge described below. Empty stamps an empty
  `repeated bytes` so the receiver leaves its existing entry alone.
- **`dest_suffix`** is the per-slot multiaddr suffix the receiver
  parses to dispatch — `/site/<NodeSiteId>` for data-plane fills or
  `/component/<cref>/op/<name>` for control-plane fills.

`dest_peer_addresses` routes to a peer; the suffix routes within
the receiver; `src_peer_addresses` keeps both peers' AddressBooks
in sync without an out-of-band exchange.

`EnvelopeCaps.max_src_peer_addresses` (default 8;
`bb-runtime/src/envelope.rs:44`) and
`max_src_peer_address_bytes` (default 256; `bb-runtime/src/envelope.rs:48`)
cap the receiver's pre-allocation so an adversarial sender cannot
balloon the AddressBook through inflated claims. `EnvelopeCodec::decode_capped`
checks both caps before any prost allocation
(`bb-runtime/src/envelope.rs:236-246`).

## Receiver dispatch

For each fill in an inbound envelope, the engine parses `dest_suffix`
and routes by the trailing-segment shape:

| Suffix shape | Routing |
|---|---|
| ends in `/site/<id>` | Data-plane: charge `fill.payload.len()` against `Engine::ingress_byte_budget`; resolve typed `SlotValue` via `decode_typed_fill` (`bb-runtime/src/engine/poll.rs:996-1083`); write to slot at `NodeSiteId`; push consumers. Backend-bound slots route through `Backend::materialize_from_wire`; non-backend slots run the global `wire_decoder_registry`. |
| ends in `/component/<cref>/op/<name>` | Control-plane: call `components[cref].dispatch_atomic(name, [(payload, correlation)], ctx)` |
| anything else | Silent drop |

For data-plane fills with `trigger_only = true`, the receiver writes a
`TriggerValue` instead of decoding the payload (which is empty).

**Self-addressed wire ops with multi-target Nodes.** When a peer
hosts more than one target (e.g. `&["Client", "Server"]`), the
wire syscall continues to address destinations by `PeerId` and the
receiver dispatches by `dest_suffix` exactly as in the single-target
case. The receiver's site name → `NodeSiteId` table is **global
across every installed target's graph** (`Engine::install_graph`
populates a single `site_names` map for every installed graph), so
a `wire.Send` from `Client`'s partition addressed at
`self.peer_id()` lands at the correct `wire.Recv` site even when
that site lives inside `Server`'s partition graph. The site_id is
unique per Node, not per target, so the partition the suffix
resolves into is determined entirely by which graph owns the
`NodeSiteId` — no extra target-awareness is required in the
envelope or the dispatch table. The same holds for control-plane
`/component/<cref>/op/<name>` fills: the deduped `ComponentRef` is
shared across every target sharing the slot, so a control-plane
fill from `Client`'s partition reaches the same component instance
the `Server` partition would dispatch to.

**The receiver does not allocate envelope-receive storage outside
the fallibility line.** `fill.payload` arrives from prost
already framework-owned (envelope decode allocated it under the
`EnvelopeCaps::max_total_bytes` cap, `bb-runtime/src/envelope.rs:197-207`).
Per-fill failures (`AllocationFailed`, `BudgetExceeded`,
`BackendMaterializeFailed`, decode failures) drop the offending
fill and continue iterating sibling fills — partial-delivery
semantics per [WIRE.md §5.4](WIRE.md#per-fill-failure-modes).
The receiver-side address-book hint is best-effort under the
byte budget: an `AddressBook` add that returns
`AddressBookError::AllocationFailed` is swallowed so the
envelope still routes; the routing decision never depends on
the address-book write succeeding.

## Local address bag

A Node binds to `Vec<Address>` at install time, not a single
`Address`. Real deployments bind to several interfaces at once: an
IPv4 public address, an IPv6 public address, a tailscale virtual,
an HTTP endpoint, a libp2p relay. Every interface is registered
against the Node's own `PeerId` in the `AddressBook` and rides
unmodified on every outbound envelope.

`bb::install(peer_id, addresses, model, targets, config)` at
`src/install.rs:235-241` takes the bag verbatim. An empty vec
skips self-registration so the Node renders all outbound
identity-protocol ops as "no addresses" failures at the protocol
level (the golden way: no synthesized `/p2p/<PeerId>` fallback).
The `addresses` bag is shared across every entry in the
`targets: &[&str]` slice — every target the install path registers
sees the same `local_addresses()` view, since the bag is owned
once by the engine's `AddressBook` keyed on the Node's
`self_peer`.

Three accessor methods sit on `Node` for runtime mutation
(`bb-runtime/src/node/mod.rs:625-660`):

```rust
pub fn local_addresses(&self) -> &[Address];
pub fn add_local_address(&mut self, addr: Address) -> Result<(), AddressBookError>;
pub fn forget_local_address(&mut self, addr: &Address) -> Result<(), AddressBookError>;
```

Each is a one-line wrapper around the AddressBook entry keyed by
`self.engine.self_peer`. `peer_address()` returns the first entry
or `Address::empty()` for source compatibility with single-address
call sites (`bb-runtime/src/node/mod.rs:611-619`).

Every dispatch context exposes `ctx.local_addresses()`
(`bb-runtime/src/runtime.rs:269-274`): every Contract impl that
populates "here are my addresses" reads through this accessor so
mutations via `node.add_local_address` propagate without rebinding.

## Receiver-side address ingest

Phase 1 of `Engine::poll` reads both sender-claimed and
transport-observed addresses for every inbound envelope. The two
sources land via separate paths into the same merge code at
`bb-runtime/src/engine/poll.rs:1005-1062`:

- **Claimed (sender)** — `envelope.src_peer_addresses` decodes into
  a `Vec<Address>`. Empty list is a no-op (sender did not advertise).
  Skip-on-unchanged guard: slice equality against the existing
  AddressBook entry elides the rewrite, capping cost at one
  AddressBook write per real change.
  (`merge_src_peer_addresses` at lines 1013-1039.)
- **Observed (transport)** — `IngressEvent::EnvelopeFrom.src_observed_address`
  carries the dialer endpoint the adapter actually saw. `None` means
  the adapter cannot surface a reflexive address. Containment check
  against the existing entry skips the rewrite; missing entry
  bootstraps a fresh one via `add_peer`. (`merge_src_observed_address`
  at lines 1049-1062.)

Claimed addresses merge first so the entry exists when the observed
merge tries to append; observed wins for the NAT-translated case
because the sender's snapshot cannot know its post-NAT endpoint.

The in-process router defaults `src_observed_address` to
`Some(Address::empty().p2p(src_peer))` via
`IngressEvent::from_in_process` (`bb-runtime/src/ingress.rs:143-152`)
so the test surface exercises the merge path real transports
populate.

## Address book semantics

The framework's `AddressBook` is the single source of truth for
every `PeerId → Vec<Address>` mapping. Overlay protocols, transport
adapters, and the application share it; no Component duplicates
address bytes in its own state.

### Ref-counted multi-address entries

Each entry is `(addresses: Vec<Address>, ref_count: u64)`:

- **`addresses`** is ordered by insertion. Earlier entries are
  higher-preference; new addresses appended to a known peer
  preserve the existing order with dedupe (the first caller's
  preference wins).
- **`ref_count`** tracks how many independent owners (overlay
  protocols, transport adapters, the application) hold a grip on
  the peer. An entry survives until its last reference drops.

### Mutations

| Call | Effect on ref_count | Effect on address list |
|---|---|---|
| `add_peer(peer, Vec<Address>)` | `+1` (new entry starts at 1) | merge-append with dedupe; preserves order |
| `drop_peer(peer)` | `-1`; entry removed at 0 | (entry removal also drops all bound addresses) |
| `register_address(peer, addr)` | unchanged | append `addr` with dedupe |
| `forget_address(peer, addr)` | unchanged | prune `addr`; entry stays even if list becomes empty |

`add_peer` rejects an empty `Vec<Address>` with
`AddressBookError::EmptyAddressList`. `drop_peer` /
`register_address` / `forget_address` error with
`AddressBookError::UnknownPeer` when the peer has no entry.

`AddressBook::lookup(peer)` returns `Option<&[Address]>`. It yields
`None` for both an unknown peer AND a peer whose address list is
empty (e.g., all addresses pruned via `forget_address` and nothing
re-added). Both are treated as "can't route" by the wire syscall.

## DAG-mutable address book — `address_book` syscall ops

The `ai.bytesandbrains.address_book` opset exposes two primitives
recorded graphs use to seed the `AddressBook` from the data plane.
Both are niche but load-bearing for discovery protocols (mDNS,
gossip-overlay address propagation, relay-discovered peer
advertisement): they let those protocols compile into a Graph that
announces and resolves peers without out-of-band host calls.

| Op | DSL helper | Underlying call | Errors on |
|---|---|---|---|
| `Insert(peer, address) → ()` | (runtime-internal; recorded directly) | new peer → `add_peer(peer, vec![addr])`; known peer → `register_address(peer, addr)` | empty input, `Full` |
| `InsertMany(peer, addresses) → ()` | `bb_dsl::syscalls::address_book_insert_many(g, peer, addrs)` (`bb-dsl/src/syscalls.rs:55-66`) | new peer → `add_peer(peer, addrs)`; known peer → one `register_address` per address | empty input, `Full` |
| `Lookup(peer) → addresses: Vec<Address>` | `bb_dsl::syscalls::address_book_lookup(g, peer)` (`bb-dsl/src/syscalls.rs:72-83`) | full ordered slice via `AddressBook::lookup` | unknown peer, empty list |

`Insert` (`bb-ops/src/syscalls/peers/insert.rs`) is the single-address
path the receiver-side merge inside the engine reaches for.
`InsertMany` (`bb-ops/src/syscalls/peers/insert_many.rs`) is the
batched form discovery protocols use to record an entire peer bag
in one NodeProto. `Lookup` returns the full ordered slice on the new
`TYPE_ADDRESS_VEC` carrier (`bb-ir/src/types/builtins.rs:311-318`);
callers that need a single entry pick one explicitly.

Address forwarding scenarios (relay, mDNS, DHT address advertisement)
are graph-level concerns: a forwarding Component records a graph
that consumes a `(PeerId, Vec<Address>)` payload + calls
`InsertMany`. Address bytes ride in the payload, not in the envelope
header.

### Carriers

`bb-runtime/src/syscall/values.rs:67-68` defines `AddressVecValue` as
the wire-eligible carrier for the ordered bag. The lattice node
`TYPE_ADDRESS_VEC` (`bb-ir/src/types/builtins.rs:311-318`) registers
under `ai.bytesandbrains.address_vec` with wire-hash 0x0303 so a
single `Lookup` output and an `InsertMany` input share one decoder
across the wire.

## Peer resolution failure

`wire::Send` cannot resolve its destination in three cases:

- the `peer` input didn't carry a parseable `PeerId`;
- the `PeerId` is unknown to the `AddressBook`;
- the `PeerId`'s entry exists but its address list is empty.

In each case the syscall produces NO envelope, pushes a record onto
the framework's `pending_peer_resolve_failures`, and publishes
`InfraEvent::PeerResolveFailure` on the bus. The engine's Phase 8
drains the queue and surfaces each entry as
`EngineStep::PeerResolveFailed { peer, op_ref, exec_id }`.

This is a first-class lifecycle event in the same family as
`PeerBlocked` / `PeerDown` / `PeerUp` from Production-Readiness
Stage 4 — not an `OpFailed`. Telemetry / dashboards can surface
resolution failures alongside other peer-lifecycle signals.

## Address construction at install time

Two-step resolution per producer Send NodeProto:

1. **Compiler** (`analyze_wire_edges`, [COMPILER.md §9](COMPILER.md))
   — for each cross-Node edge, stamps the destination multiaddr's
   recv-side symbolic value name as
   `metadata_props["ai.bytesandbrains.dest_site_name.<input>"] = "<recv_output_name>"`.
   The name is symbolic because runtime `NodeSiteId`s aren't allocated
   until install.

2. **Node install** — after all `ModelProto` graphs are
   installed, walks each Send NodeProto, looks up every
   `dest_site_name.<input>` value against the global `site_names`
   map (spanning every installed graph), and rewrites it as a
   canonical `dest_suffix.<input>` `AttributeProto` carrying
   `Address::empty().site(NodeSiteId).to_bytes()`.

At dispatch time the wire syscall's `collect_fills` reads each
fill's suffix from the resolved `dest_suffix.<input>` attribute
via `ctx.current_node_attributes`. A `<name>_suffix` companion
input is supported as a fallback for sim harnesses that construct
envelopes manually outside the compilation pipeline.

## What this replaces

| Legacy mechanism | Replaced by |
|---|---|
| `Address(Vec<u8>)` opaque bytes | typed `Protocol` segment sequence |
| `Engine.recv_sites: HashMap<String, Vec<(OpRef, NodeSiteId)>>` | parse `Address::site_id()` from each fill |
| `Engine.event_subscriptions: HashMap<String, Vec<OpRef>>` | `HashMap<String, Vec<NodeSiteId>>` — Phase 3 writes a `TriggerValue` to each subscribed site and pushes consumers (same dispatch path as wire fills) |
| `Engine.routing_table: RoutingTable` for `(opset, version) → DataPlane/Control` | parse `dest_suffix` segments |
| `WireEnvelope.opset / message_type / peer_id / hash` | encoded as multiaddr segments |
| `WireEnvelope.dest_peer: uint64` (single PeerId on the wire) | `WireEnvelope.dest_peer_addresses: repeated bytes` — resolved address list; transport picks one |
| `AddressBook: PeerId → Address` (single address per peer) | `AddressBook: PeerId → (Vec<Address>, ref_count)` — ordered list + ref-counting |
| `PeerAddressBook: PeerId → (identity, Address)` (duplicated address) | `PeerAddressBook: PeerId → identity` only — addresses live solely in `AddressBook` |
| `ModelProto.recv_index` / `RecvSiteRef` | gone — compiler stamps `dest_site_name.<input>` on each producer Send NodeProto; Node install resolves to `dest_suffix.<input>` Address bytes via the global `site_names` map |
| `Engine.wire_inbox` (fallback for unaddressed payloads) | every fill is addressed; malformed addresses drop silently |
| `Engine.framework.address_book` / `peer_address_book` were transient-only | snapshot/restore via `FrameworkSnapshot.address_book` + `peer_address_book` — peer registries persist across restarts |

## Code map

| Concern | Source |
|---|---|
| `Address`, `Protocol`, `AddressBook`, `AddressBookError` | `bb-runtime/src/framework/address_book.rs` |
| `PeerAddressBook` (identity only) | `bb-runtime/src/framework/peer_address_book.rs` |
| Wire envelope proto (`src_peer_addresses` at field 8) | `bb-ir/proto/bb_core.proto:135` |
| Envelope encode/decode + `EnvelopeCaps` (claim caps) | `bb-runtime/src/envelope.rs:42-48,236-246` |
| `AddressVecValue` typed carrier | `bb-runtime/src/syscall/values.rs:59-65,101` |
| `TYPE_ADDRESS_VEC` lattice node | `bb-ir/src/types/builtins.rs:306-318,369,409` |
| Outbound `wire.Send` (stamps `src_peer_addresses`) | `bb-ops/src/network/wire/mod.rs:196-220` |
| `address_book/` syscall ops (`Insert`, `InsertMany`, `Lookup`) | `bb-ops/src/syscalls/peers/{insert,insert_many,lookup}.rs` |
| DSL helpers (`address_book_insert_many`, `address_book_lookup`) | `bb-dsl/src/syscalls.rs:50-83` |
| `EngineStep::PeerResolveFailed` | `bb-runtime/src/engine/step.rs` |
| `InfraEvent::PeerResolveFailure` | `bb-runtime/src/bus.rs` |
| Inbound parse + dispatch + ingest merge | `bb-runtime/src/engine/poll.rs:473-514,1005-1062` |
| `IngressEvent::EnvelopeFrom.src_observed_address` | `bb-runtime/src/ingress.rs:47-60,143-152` |
| `Node::install(peer_id, Vec<Address>, ...)` entry | `src/install.rs:236-242` |
| `Node` local-address accessors | `bb-runtime/src/node/mod.rs:611-660` |
| `ctx.local_addresses()` | `bb-runtime/src/runtime.rs:262-274` |
| `GlobalRegistry::Announce` / `Handshake` carry full bags | `bb-ops/src/protocols/global_registry/mod.rs:56-72,162-186,477-543` |
| Tests: address round-trips + ref-counting | `bb-runtime/src/framework/address_book_tests.rs` |
| Tests: `address_book` syscall ops | `bb-ops/src/syscalls/peers/{insert,insert_many,lookup}_tests.rs` |
| Tests: envelope shapes + `src_peer_addresses` caps | `bb-runtime/src/envelope_tests.rs` |
| Tests: ingress merge (claimed + observed) | `bb-runtime/src/engine/poll_src_peer_addresses_tests.rs`, `bb-runtime/src/engine/poll_observed_address_tests.rs` |
| Tests: end-to-end multi-address delivery | `tests/local_multi_address_e2e.rs`, `tests/wire_envelope_src_addresses.rs` |
