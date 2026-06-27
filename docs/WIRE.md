# WIRE.md — envelope spec + custom messaging

The wire is the single byte channel between BB Nodes. **Every byte
the framework hands to transport, and every byte transport hands
back, rides as one `WireEnvelope`.** The framework owns no sockets;
the envelope is the contract with whatever transport the host wired
up (libp2p, HTTP, QUIC, in-process simulator).

This document specifies:

- The canonical `WireEnvelope` proto schema, field by field.
- How data-plane messaging works for graph edges crossing Node
  boundaries (the `ai.bytesandbrains.wire v1` opset).
- How control-plane messaging works for protocol components shipping
  their own message types (any other opset).
- How application authors and component authors define **custom
  message types that just work** in the runtime without touching
  transport, encoding, or routing.
- How triggers, tensors, and arbitrary user types cross the wire
  identically through the same machinery.
- The transport-adapter contract.

Pairs with ENGINE.md (the runtime that drives the wire),
IR_AND_DSL.md (how wire ops appear in the graph), and ENGINE.md
(broader runtime spec).

---

## Part 1 — Overview

**Addresses route themselves.** Every delivery target — whether a
data-plane slot or a control-plane component op — is named by a
multiaddr per [`ADDRESSING.md`](ADDRESSING.md). The receiver parses
the per-fill `dest_suffix` to find the target; there is no opset
routing table, no `recv_sites` HashMap, no `message_type` keyed
lookup.

### Principle 1a — External byte payloads cross as `&[u8]`

**Load-bearing architectural invariant.** External byte payloads
crossing the engine boundary do so as **borrowed slices**
(`&[u8]`), never owned `Vec<u8>` taken by value. The framework
copies the bytes into its own fallibly-allocated memory inside the
boundary call — that copy IS the ownership transition, not an
inefficiency.

Rationale: many transport stacks (libp2p reads, QUIC / io_uring
completion buffers, kernel ring buffers, statically-allocated DPDK
queues) hand the framework a pointer into THEIR owned memory. The
framework must NOT keep a reference past the call (the transport
reclaims the buffer immediately on return) and must NOT take
ownership (the transport's allocator owns the buffer, not the
framework's).

Boundary surfaces that obey Principle 1a today:

- `CompletionSink::complete(&self, cmd_id, result_bytes: &[u8])`
  + `fail(&self, cmd_id, detail: &str)` —
  `bb-runtime/src/runtime.rs:29-97`. The `IngressQueue` impl
  reads the borrowed slice, charges against the byte budget,
  fallibly reserves a framework-owned `Vec<u8>`, copies in,
  and routes through the `Completion` / `CompletionFailed`
  `IngressEvent` variants. `detail` is truncated to
  `COMPLETION_DETAIL_CAP` (4 KiB) at a UTF-8 character boundary
  rather than rejected so the host's display message always
  lands.
- `Node::deliver_event(&mut self, module, input, value_bytes: &[u8])`
  — `bb-runtime/src/node/mod.rs:765-835`. The framework caps
  against `NodeConfig::max_app_event_bytes`
  (`bb-runtime/src/node/config.rs:167`), charges against
  `ingress_byte_budget`, and copies via `try_reserve_exact` +
  `extend_from_slice`.
- `Node::invoke(&mut self, module, inputs: &[(&str, &[u8])])`
  — `bb-runtime/src/node/mod.rs:845-952`. Caps against
  `NodeConfig::max_invoke_inputs` (count) +
  `max_invoke_bytes` (cumulative); fallibly reserves the outer
  `Vec` and each per-input `Vec<u8>`.

Small metadata (`PeerId`, `CommandId`, `type_hash`, name strings
under a fixed cap) crosses by value or on the stack; Principle 1a
applies specifically to large byte payloads. Inside the framework,
byte ownership flows normally (`Vec<u8>` moved between
framework-owned slots, `Arc<...>` for backend buffers); the
external-to-internal copy is the one mandatory transition.

**Framework-to-backend handoff is NOT a Principle 1a boundary.**
`Backend::materialize_from_wire(&self, type_hash, bytes: Vec<u8>)`
(`bb-runtime/src/contracts/backend.rs:497`) takes ownership by
value because the backend lives inside the framework ecosystem
and plays by the runtime contract. The framework copied or owned
the bytes already; it hands them to the backend with full
ownership transfer so the backend can adopt them zero-copy via
`ArrayD::from_shape_vec` (when alignment permits), pull a buffer
from a pool and copy in, or fresh-allocate.

Two delivery shapes share the one envelope format, distinguished by
the trailing segments of each fill's `dest_suffix`:

**Data plane** — suffix ends in `/site/<NodeSiteId>`. Originates
from the single user-authored `wire_send(g, payload, peer_id)` DSL
free function in user code; lands at the matching slot on the
receiver, which pushes the slot's downstream consumers onto the
frontier. The receiver-side `wire.recv` NodeProto is synthesized
by the compiler's `synthesize_wire_recvs` pass — users never
author it. The framework owns the encoding (bincode via the
`SlotValue` blanket — anything `Clone + Serialize + Deserialize`
rides the wire), the routing (peer_id resolved to multiaddr via
`address_book`), and the correlation (a freshly minted
`wire_req_id` per send). The app sees only `wire_send`.

**Control plane** — suffix ends in `/component/<ComponentRef>/op/<name>`.
Carries the component's own message types (gossip pushes, FindNode
queries, etc.). The component owns payload encoding and decoding;
the framework parses the suffix and dispatches to
`components[cref].dispatch_atomic(op_name, ...)`.

**Both shapes use the same `WireEnvelope` wire format.** A Node's
view: bytes in via `Node::deliver_inbound(envelope)`, bytes out via
`EngineStep::SendEnvelope(envelope)`. A transport adapter's view:
a stream of envelopes to ship + a stream of envelopes to push back.

---

## Part 2 — The canonical envelope schema

The envelope ships a peer-level multiaddr destination plus one or
more `SlotFill`s. The compiler batches all cross-Node edges from
the same producer Send op into one envelope; each input value
contributes one `SlotFill`.

```proto
syntax = "proto3";
package bb.core;

// bb-ir/proto/bb_core.proto

enum CorrelationKind {
  NONE = 0;
  REQUEST = 1;
  RESPONSE = 2;
}

message WireCorrelation {
  CorrelationKind kind = 1;
  // Sender-allocated id; receiver echoes the same id in the
  // matching RESPONSE envelope.
  uint64 wire_req_id = 2;
}

message WireEnvelope {
  // Ordered destination address list — the framework's wire syscall
  // populates this from the AddressBook at dispatch time (the
  // resolved snapshot of `AddressBook::lookup(peer)`). The host's
  // transport adapter picks one of the entries based on its
  // networking capabilities (IPv4 reachability, QUIC support,
  // relay preference, etc.). Each entry is `Address::to_bytes()`.
  // Misses (peer unknown or empty list) surface
  // `EngineStep::PeerResolveFailed` instead of producing an envelope.
  repeated bytes dest_peer_addresses = 1;

  // One or more typed slot fills batched in this envelope. Each
  // fill carries its own per-slot multiaddr suffix; receivers parse
  // the suffix to route (intra-node only).
  repeated SlotFill fills = 2;

  // Correlation header. Request/response pairing for
  // send_req_batched ↔ send_resp; ignored for fire-and-forget sends.
  WireCorrelation correlation = 3;

  // ... deadline_ns + edge_rtt_reports + src_peer_bytes + schema_version ...

  // Sender-claimed local-address bag — snapshot of the sender's
  // AddressBook entry for its own PeerId at envelope-mint time. The
  // receiver merges into its own AddressBook entry for `src_peer` so
  // future replies can dial back on every reachable interface. Empty
  // means the sender chose not to advertise (e.g. `local_addresses()`
  // was empty); the receiver leaves its existing entry untouched.
  // Bounded at decode time by `EnvelopeCaps.max_src_peer_addresses`
  // (default 8) and `max_src_peer_address_bytes` (default 256).
  // (`bb-ir/proto/bb_core.proto:135`.)
  repeated bytes src_peer_addresses = 8;
}

message SlotFill {
  // Per-slot multiaddr suffix, appended to the receiver's
  // self-identity to form the full destination address. Two shapes
  // per ADDRESSING.md:
  //   /site/<NodeSiteId>             — data-plane slot fill
  //   /component/<cref>/op/<name>    — control-plane component op
  bytes dest_suffix = 1;

  // Wire-encoded payload bytes. Decoder for data-plane fills is
  // resolved via the destination Site's declared TypeNode (looked
  // up from the installed graph). Control-plane fills hand the
  // bytes directly to the component's dispatch_atomic.
  bytes payload = 2;

  // True when the receiver only needs the firing signal. The
  // receiver writes a TriggerValue to the slot; payload bytes are
  // empty.
  bool trigger_only = 3;
}
```

**Compatibility rule:** envelope fields are append-only with stable
field numbers. New routing kinds do NOT require an envelope-schema
change — they're new multiaddr segment codes (see
[ADDRESSING.md](ADDRESSING.md)).

---

## Part 3 — Encoding + wire format

`WireEnvelope::encode_framed()` produces varint-length-prefixed
protobuf — the standard "framed protobuf" wire format every
protobuf-aware tool recognizes. The transport adapter writes the
framed bytes to its socket; the peer reads them off and calls
`WireEnvelope::decode_framed(&bytes)`.

```rust
// Framework provides:
impl WireEnvelope {
    pub fn encode_framed(&self) -> Vec<u8>;
    pub fn decode_framed(bytes: &[u8]) -> Result<Self, EnvelopeError>;
    // Plus unframed encode_to_vec/decode_bytes for use inside other
    // proto messages (e.g. nested in a snapshot envelope).
}
```

**Endianness:** protobuf's encoding is endianness-independent;
varints and fixed-size integers are decoded the same on any
platform.

**Maximum size:** the framework imposes no envelope-size cap; the
transport adapter SHOULD impose its own (e.g. libp2p typically
limits frames to 16 MiB). Oversize payloads should split via the
data-plane's batching mechanism (one SlotFillBatch per envelope,
multiple envelopes per cycle).

---

## Part 4 — Routing

`Node::deliver_inbound(envelope)` iterates `envelope.fills` and
dispatches each fill by parsing its `dest_suffix` per
[ADDRESSING.md](ADDRESSING.md). There is no routing table — the
trailing multiaddr segments are the routing decision.

### 4.1 Routing decisions

| `fill.dest_suffix` shape | Routing |
|---|---|
| ends in `/site/<NodeSiteId>` | Data-plane: materialise the fill into a typed `SlotValue` via the shared `wire_decoder_registry` (or `TriggerValue` if `trigger_only`) per §5.4; write into the slot at a fresh `ExecId`; push consumers from `installed_graph.consumers[site]` onto the frontier. |
| ends in `/component/<ComponentRef>/op/<name>` | Control-plane: borrow the bound component; synthesize atomic-dispatch inputs (`payload`, `correlation` as opaque entries); call `component.dispatch_atomic(name, inputs, ctx)`. |
| Any other shape (no Site, no Component+Op) | Drop silently. |

### 4.2 Address construction at install time

The compiler's `analyze_wire_edges` pass — when run end-to-end
across peer partitions — stamps each producer Send NodeProto's
metadata with the destination multiaddr suffix derived from the
matching Recv site on the consumer's installed graph. At dispatch
time, the wire syscall populates each `SlotFill.dest_suffix` from
the stamped metadata.

### 4.3 No routing-table snapshot

No routing table to snapshot. The installed graphs and their
`consumers` maps are reconstructed from the snapshotted
`ModelProto`s on restore; address-routed delivery resumes
unchanged.

---

## Part 5 — The data plane: `ai.bytesandbrains.wire v1`

The data plane carries graph-edge values across Node boundaries.
The framework owns it entirely: encoding, batching, correlation,
decoding, routing to recv ops. **Application authors interact via
the DSL only.**

> **Wire ops and the hoist optimization.** Wire ops are NOT subject
> to `hoist_pure_subgraphs` (COMPILER.md §11a, planned). Any
> `module_instance` scope chain whose nodes include a wire op stays
> inlined in the partition body so `partition_by_wire_ops` can slice
> across it. Only pure (no-wire) sub-Module bodies are hoisted into
> `ModelProto.functions[]`.

### 5.1 The app's view

To send any value over the wire to a destination peer:

```rust
// In a Module's op(&self, g: &mut Graph, _: &[Output]):
wire_send(g, my_value, my_dest);
```

That's the entire app surface for fire-and-forget. The framework:

1. The compiler's `analyze_wire_edges` pass identifies the
   cross-Node edge and stamps
   `metadata_props["ai.bytesandbrains.dest_site_name.<input>"]` on
   the producer Send NodeProto. Node install resolves it to
   a `dest_suffix.<input>` `AttributeProto` carrying
   `/site/<NodeSiteId>` Address bytes per
   [ADDRESSING.md](ADDRESSING.md).
2. At dispatch, the wire syscall reads the resolved suffix from
   `ctx.current_node_attributes`, calls
   `SlotValue::to_wire_bytes` (bincode) on the value, and builds
   one `SlotFill { dest_suffix, payload, trigger_only }`.
3. Resolves the destination `PeerId` via the framework's
   `AddressBook` and builds a `WireEnvelope { dest_peer_addresses:
   vec![/* AddressBook::lookup(peer) bytes */], fills: [...],
   correlation: None }`. Lookup miss (peer unknown or empty list)
   surfaces `EngineStep::PeerResolveFailed` + bus
   `InfraEvent::PeerResolveFailure` instead of producing an
   envelope (see [ADDRESSING.md §Peer resolution failure](ADDRESSING.md#peer-resolution-failure)).
4. Snapshots `ctx.local_addresses()` once per Send and stamps
   `env.src_peer_addresses` on every envelope minted in the fan-out
   (`bb-ops/src/network/wire/mod.rs:196-220`). The snapshot is the
   sender's `AddressBook` entry for its own `PeerId` at dispatch
   time; receivers merge it into their own AddressBook so future
   replies can dial back on every reachable interface.
5. Pushes the envelope onto `outbound_queue`.
6. The engine drains the outbound queue at end of cycle into
   `EngineStep::SendEnvelope`s.
7. The host transport reads `env.dest_peer_addresses` (ordered by
   peer-stated preference), picks one based on its networking
   capabilities, and ships the bytes.

On the receiver:

1. Host transport's read path calls `node.deliver_inbound(env)` or
   pushes `IngressEvent::EnvelopeFrom { src_peer, envelope,
   src_observed_address }` directly. The optional observed-address
   field carries the dialer endpoint the adapter saw (the
   in-process router uses `IngressEvent::from_in_process` at
   `bb-runtime/src/ingress.rs:143-152`).
2. Phase 1 of `Engine::poll` merges every advertised address into
   the receiver's AddressBook (`bb-runtime/src/engine/poll.rs:497-505,1005-1062`):
   sender-claimed `envelope.src_peer_addresses` first, then the
   transport-observed `src_observed_address` if present. Slice
   equality and containment checks elide rewrites when nothing
   changed, capping the cost at one AddressBook write per real
   change under sustained traffic.
3. The engine iterates `env.fills` and for each one calls
   `Address::from_bytes(&fill.dest_suffix)`.
4. Suffix ends in `/site/<NodeSiteId>`: materialise the payload
   bytes into a typed `SlotValue` via the shared
   `wire_decoder_registry` (see §5.4 for the failure modes), or
   `TriggerValue` if `trigger_only`. Write the typed carrier to
   that slot at a fresh `ExecId`; push the slot's downstream
   consumers. Per-fill failures emit one
   `InfraEvent::WireReceiveError` + matching
   `EngineStep::WireReceiveFailed` and continue iterating sibling
   fills (partial-delivery semantics — see §5.4).
5. The consumer Recv op fires; downstream Ops read the value.

The app never sees envelopes, never touches transport, never
encodes bytes. The app declared "I send a u64 to this dest" via
the DSL; the runtime delivered "a u64 landed at the addressed
slot."

### 5.2 Request / response

> **v1 design** — The wire surface is a single
> `wire.send(payload, peer_id) → (data, handle)` op. Request/response
> correlation rides on the implicit dataflow connection between two
> sends, not on explicit cohort handles. A typical request/response
> pattern fans out via a `send` to the responder, and the responder's
> reply is another `send` back; the compiler's `synthesize_wire_recvs`
> pass cuts the receiver-side `wire.recv` NodeProtos on whichever
> partition consumes each send's output. The `handle` output of each
> send carries a freshly minted `wire_req_id` for future explicit
> correlation work; v1 does not require threading it.

Back at the original sender:

3. Inbound envelope's `correlation = Response(wid)` routes to the
   Wire's in-flight cohort
4. When N responses collected, the Wire signals the suspended
   `WireSendReqBatched` Op's CommandId
5. Engine resumes; downstream of `responses` output fires

All of this is invisible to the user code. The user wrote a
DSL call; the framework delivered the cohort.

### 5.3 Slot-fill batching

When N graph edges from one producer Node target the same receiver
in the same cycle, the framework packs them into **one envelope**
carrying multiple `SlotFill`s. The compiler's `analyze_wire_edges`
pass groups edges by `(producer_role, destination_peer)` (the
`batch_group_id` metadata) and stamps per-input
`dest_site_name.<input>` entries on the producer Send NodeProto;
Node install resolves each into a `/site/<id>` Address
suffix.

```
Producer cycle:
  Op A fires, output → graph edge to Receiver, slot bb.peer_id
  Op B fires, output → graph edge to Receiver, slot bb.u64
  Op C fires, output → graph edge to Receiver, slot bb.trigger (trigger-only)

Outbound:
  ONE envelope, dest_peer_addresses = [
                  /* Address::to_bytes() for each AddressBook entry */
                  <addr_0_bytes>, <addr_1_bytes>, ...
                ],
                fills = [
        SlotFill { dest_suffix=/site/<id_A>, payload=<peer_id bytes>,
                   trigger_only=false },
        SlotFill { dest_suffix=/site/<id_B>, payload=<u64 bytes>,
                   trigger_only=false },
        SlotFill { dest_suffix=/site/<id_C>, payload=[],
                   trigger_only=true },
      ]
```

The wire syscall populates `dest_peer_addresses` from
`AddressBook::lookup(peer)` at dispatch time — the host transport
adapter just picks one of the entries based on its networking
capabilities (libp2p, in-process channel, relay, etc.). The
framework only constructs + parses BB-internal `dest_suffix`
routing segments (`/site`, `/component`, `/op`) inside each fill.

On the receiver, `deliver_inbound` iterates each fill, parses its
`dest_suffix`, writes the payload to the addressed slot (or a
`TriggerValue` if `trigger_only`), and pushes the slot's
consumers. Three values delivered as one envelope; one TCP send,
one parse, three slot writes.

A single-fill envelope is just the degenerate case (one fill in
the `fills` array). There's no special unbatched path.

### 5.4 Wire-eligibility and typed-receive

<a id="typed-receive"></a>

Wire-eligibility is the `SlotValue` blanket: any type satisfying

```rust
T: Any + Send + Sync + Clone
   + serde::Serialize + serde::de::DeserializeOwned
```

is a `SlotValue` by construction (`bb-ir/src/slot_value.rs`).
Encoding on the producer side is `SlotValue::to_wire_bytes`,
which is `bincode::serialize(self)` for the universal blanket;
`wire.Send` also stamps `SlotFill.type_hash` from
`SlotValue::type_hash()` so the receiver can route the bytes back
to the original carrier without coordinating with the sender
(`bb-ops/src/network/wire/mod.rs:433-438`).

#### Decoder registry

Every `register_type_node!(T, &TYPE_X)` invocation emits a
`WireDecoderBinding` alongside the `RuntimeTypeBinding`
(`bb-ir/src/slot_value.rs:178-256`); the global
`wire_decoder_registry()` is a `HashMap<u64, WireDecodeFn>` keyed
by `type_hash`. The CompositeValue cross-wire codec already
consults this registry to rebuild typed children
(`bb-runtime/src/syscall/values.rs:114-165`); the single-fill
receive path is symmetric with Bundle/Unbundle so authoring one
new carrier participates in both surfaces with no extra
registration.

#### Inbound flow

`Engine::deliver_data_plane_fill` (`bb-runtime/src/engine/poll.rs:828-901`)
resolves the typed `SlotValue` BEFORE allocating an `ExecId` or
mutating the slot table:

1. Trigger fills bypass the registry and install `TriggerValue`
   directly.
2. Non-trigger fills go through `decode_typed_fill`
   (`bb-runtime/src/engine/poll.rs:996-1083`). The step resolves
   the destination slot's binding via
   `GraphSlot::recv_site_to_slot_id`
   (`bb-runtime/src/engine/graph_slot.rs`) and
   `Engine::slot_id_to_role_ref`
   (`bb-runtime/src/engine/core.rs:236`), validates the
   wire-type assertion in `GraphSlot::recv_wire_type_hash`
   (`bb-runtime/src/engine/graph_slot.rs:63-73`), pre-charges
   `fill.payload.len()` against
   `Engine::ingress_byte_budget`
   (`bb-runtime/src/engine/core.rs:540-552`), then branches:
   - **Backend-mediated path** when the destination slot binds a
     `Backend` role: `materialize_via_backend`
     (`bb-runtime/src/engine/poll.rs:1085-1206`) `mem::take`s
     `fill.payload` (already framework-owned from envelope
     decode), hands the `Vec<u8>` to the backend via the
     per-`T` `BackendRuntime::materialize_from_wire` dispatcher,
     and wraps the result in a
     `BackendTensorCarrier`
     (`bb-runtime/src/slot_value.rs:43-174`) whose
     `charged_bytes` + `backend_ref` accounting fields the engine
     stamps post-dispatch.
   - **Framework-carrier path** otherwise: the global
     `wire_decoder_registry()` decoder runs on `&fill.payload`
     and returns a typed `Box<dyn SlotValue>`.
3. The materialised `SlotValue` is written to the slot; downstream
   consumers fire. Slot-table overwrite / eviction calls
   `SlotValue::charged_bytes()` and releases the count via
   `Engine::release` (`bb-runtime/src/engine/core.rs:554-`).

No `BytesValue` fallback. Authors who want raw bytes ship a
`BytesValue` from the sender; the decoder registry round-trips it
to `BytesValue` on the receiver. Anything else lands as the
typed carrier the sender stamped — downstream consumers downcast
via `as_any().downcast_ref::<T>()` against the declared
denotation; the previous "decode bytes via bincode against the
graph's TypeNode" hop is gone.

#### Per-fill failure modes

Five per-fill failures surface as
`InfraEvent::WireReceiveError` on the bus + matching
`EngineStep::WireReceiveFailed` (`bb-runtime/src/bus.rs:103-114`,
`bb-runtime/src/bus.rs:205-263`,
`bb-runtime/src/engine/step.rs`):

- **TypeMismatch** — destination slot declares an expected
  `type_hash` via `GraphSlot::recv_wire_type_hash`
  (`bb-runtime/src/engine/graph_slot.rs:63-73`) and the inbound
  fill's `type_hash` does not match. Checked before the decoder
  lookup so a mis-typed payload never reaches the decoder.
- **UnknownTypeHash** — no decoder is registered for the stamped
  hash (version skew between sender and receiver, or a fuzzed
  envelope). The slot stays empty; the bus event surfaces the
  drop.
- **DecodeFailed** — the registered decoder ran and returned
  `Err`. The variant carries `error_summary: String` (typically
  the underlying `bincode::Error::to_string`) so subscribers
  attribute the drop without re-parsing the bytes.
- **AllocationFailed** — a `try_reserve_exact` reservation along
  the per-fill ingress path returned `TryReserveError`, or a
  caller-side cap rejected before allocating
  (`bb-runtime/src/bus.rs:233-241,265-283`). The variant carries
  `byte_count: usize` (the bytes requested) plus
  `reason: AllocFailReason::{HeapExhausted, PerItemCapExceeded { cap }}`.
- **BudgetExceeded** — the pre-decode `try_charge` against
  `Engine::ingress_byte_budget` overflowed
  (`bb-runtime/src/engine/poll.rs:1035-1046`,
  `bb-runtime/src/bus.rs:242-253`). The variant carries
  `byte_count` (the fill's payload length) and
  `budget_remaining: usize` (bytes still available when the
  charge attempt fired). The fill is dropped before any decoder
  runs; siblings in the same envelope still deliver.
- **BackendMaterializeFailed** — the bound backend's
  `materialize_from_wire` returned `Err`, or the engine could
  not borrow the backend component
  (`bb-runtime/src/engine/poll.rs:1113-1142,1192-1204`,
  `bb-runtime/src/bus.rs:254-263`). The variant carries
  `backend_ref: ComponentRef` and
  `backend_error_summary: String`. The byte charge is released
  before the event surfaces.

Each event carries `src_peer: Option<PeerId>`, `fill_index: u32`
(0-based position within the envelope), `actual_hash: u64`, and
`payload_size: usize`. The sub-kind discriminator lets
subscribers route on the top-level topic (`"WireReceiveError"`)
and match on the cause from variant fields — the
`PeerSuspect`/`PeerDown`/`PeerLive` triple split distinct
lifecycle events into separate variants; the three wire-receive
failures share one lifecycle stage (one fill, one decode step)
and one audience (anyone watching wire-payload integrity), so a
single variant with an enum sub-kind keeps the bus-topic count
down.

#### Partial-delivery semantics

`Engine::deliver_envelope` (`bb-runtime/src/engine/poll.rs:715-735`)
iterates fills sequentially. A per-fill failure emits its
`WireReceiveError` event and `WireReceiveFailed` step, then the
loop continues to the next fill — failures do not short-circuit
the envelope. Other fills with valid `type_hash`es still
deliver. The `fill_index` field on every failure event lets
subscribers identify which fill failed when an envelope partial-
delivers.

#### Destination metadata

`GraphSlot::recv_wire_type_hash: HashMap<NodeSiteId, u64>`
(`bb-runtime/src/engine/graph_slot.rs:63-73`) holds the expected
`type_hash` per `wire.Recv` payload site. Populated at install
time alongside `recv_sender_sites`. Slots without an entry are
treated as dynamic / Any: the registry lookup proceeds without
the mismatch check.

The compiler does not yet stamp `ValueInfoProto.type_node` on
Recv payload outputs with a hash that matches the producer-side
`SlotValue::type_hash()` derivation, so the map stays empty at
install time in production
(`bb-runtime/src/engine/graph_slot.rs:190-202`); the
TypeMismatch check is dormant until that compiler follow-up
lands (per
`docs/internal/superpowers/specs/2026-06-24-wire-recv-typed-receive-and-bundle-bench.md`
§7). Tests populate the map manually to exercise the path
(`bb-runtime/src/engine/poll_typed_receive_tests.rs`).

---

## Part 6 — The control plane

Any opset that is NOT `ai.bytesandbrains.wire v1` is control plane.
A `ProtocolRuntime` component owns the opset; the framework
provides routing + lifecycle but does NOT touch the payload bytes.

### 6.1 The component contract

`ProtocolRuntime` follows the universal runtime-trait contract from
[ROLES.md §2](ROLES.md). It has no fixed set of role methods —
protocols don't share a standard verb catalog. The trait surface is
the universal `atomic_opset()` + `dispatch_atomic()` pair; the
declared atomic opset's op names are the protocol's wire-protocol
vocabulary. Inbound `SlotFill`s addressed via
`/component/<cref>/op/<name>` per
[ADDRESSING.md](ADDRESSING.md) route into `dispatch_atomic`
exactly like any other op.

For the canonical long-form example — a Kademlia-style component
that simultaneously implements `ProtocolRuntime` + `IndexRuntime` +
`PeerSelectorRuntime` on one struct, with the protocol's
async-networking machinery transparent to the data plane — see
[AUTHORING_COMPONENTS.md Parts 5–10](AUTHORING_COMPONENTS.md).

```rust
pub trait ProtocolRuntime: AnyComponent + Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    /// The atomic opset this protocol declares. Both inbound envelope
    /// routing AND user-graph DSL ops register here. Domain matches
    /// the wire envelope's opset field.
    fn atomic_opset(&self) -> AtomicOpsetDecl;

    /// Single dispatch entry. For inbound envelopes addressed to
    /// this component via `/component/<cref>/op/<name>` per
    /// [ADDRESSING.md](ADDRESSING.md), the framework synthesizes
    /// inputs from the matching SlotFill:
    ///   - `payload`:     Opaque<Bytes>  (the fill's payload bytes)
    ///   - `correlation`: Opaque<u64>    (the envelope's wire_req_id)
    /// `op_type` is the `Op` segment from the multiaddr suffix
    /// (e.g. "Push", "FindNode"). For user-graph DSL ops the inputs
    /// come from upstream slot values exactly like any role op.
    /// The impl decodes payload bytes per `op_type` and mutates
    /// state.
    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error>;

    /// Optional: component-scheduled timer maturity hook. Same
    /// behavior as ENGINE.md §29.5; orthogonal to atomic dispatch.
    fn on_timer(
        &mut self,
        kind: ComponentTimerKind,
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<(), Self::Error> { Ok(()) }
}
```

A component declares its atomic opset once; the framework routes
inbound envelopes whose opset matches; the component decodes payload
bytes however it likes (bincode, hand-rolled proto, JSON, raw structs)
and reacts inside `dispatch_atomic`.

### 6.2 The component's view of outbound

To send a control-plane envelope, the component pushes onto
`ctx.outbound_queue` from inside `dispatch_atomic` or `on_timer`:

```rust
impl ProtocolRuntime for MyGossipProtocol {
    type Error = ProtocolError;

    fn atomic_opset(&self) -> AtomicOpsetDecl {
        AtomicOpsetDecl {
            domain: "myapp.gossip",
            version: 1,
            ops: &[
                AtomicOpDecl { name: "Push", /* IO */ kind: AtomicOpKind::Async },
                // ... additional gossip message types ...
            ],
        }
    }

    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        _ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error> {
        match op_type {
            "Push" => {
                // Inbound fill: framework synthesizes payload + correlation
                // inputs (per Part 6.1). Source attribution comes from
                // the transport adapter via `IngressEvent::EnvelopeFrom`'s
                // `src_peer` (PeerId).
                let payload: &[u8] = inputs.get_opaque("payload")?;
                let msg: GossipPush = bincode::deserialize(payload)?;
                self.merge_view(msg.view);
                Ok(DispatchResult::Immediate(vec![]))
            }
            _ => Err(ProtocolError::UnknownOp(op_type.to_string())),
        }
    }

    fn on_timer(&mut self, _kind: ComponentTimerKind, ctx: &mut RuntimeResourceRef<'_>)
        -> Result<(), Self::Error>
    {
        // Periodic gossip push: pick a random peer + ship our view.
        let peer = self.pick_random_peer()?;
        let push_msg = GossipPush { view: self.view.clone() };
        let payload = bincode::serialize(&push_msg)?;

        // Address the fill at the recipient's gossip component op
        // via /component/<cref>/op/Push per ADDRESSING.md. The
        // component_ref is the recipient's binding for this opset
        // (known at install time via the routing handshake).
        let dest_suffix = Address::empty()
            .component(self.peer_gossip_cref(peer))
            .op("Push")
            .to_bytes();

        // Resolve the peer's address list via the AddressBook +
        // pack it into the envelope. Lookup miss surfaces as
        // EngineStep::PeerResolveFailed (handled by the framework's
        // wire syscall in user-authored sends); when a Component
        // pushes envelopes directly, it must perform the lookup
        // itself.
        let Some(addrs) = ctx.address_book.lookup(peer) else {
            // No entry — record + skip (or publish PeerResolveFailure).
            return Ok(());
        };
        let dest_peer_addresses: Vec<Vec<u8>> =
            addrs.iter().map(|a| a.to_bytes()).collect();
        ctx.outbound_queue.push(WireEnvelope {
            dest_peer_addresses,
            fills: vec![SlotFill {
                dest_suffix,
                payload,
                trigger_only: false,
            }],
            correlation: Some(WireCorrelation::none()),
        });

        // Re-schedule.
        ctx.scheduler.schedule_component_timer(
            self.component_ref(),
            ComponentTimerKind(0),
            now_ns() + 1_000_000_000,
        );

        Ok(())
    }
}
```

The component:
- Defines its own message types (`GossipPush`, etc.) as plain Rust
  structs with `Serialize + Deserialize`.
- Picks its own encoding (bincode is canonical; any protobuf /
  json / hand-rolled choice works).
- Decodes inbound payload bytes per `op_type` dispatch.
- Pushes outbound envelopes via `ctx.outbound_queue` addressed
  via `/component/<cref>/op/<name>` multiaddr suffixes.

**The framework dispatches, the component speaks.** A new protocol
ships as: one `ProtocolRuntime` impl + a few message structs.
Transport adapter unchanged; user graphs unchanged unless they want
to use the protocol's DSL surface.

### 6.3 Request / response across control plane

Control-plane components MAY use the envelope's `correlation` field
for request/response pairing, identically to data plane:

```rust
// In an outbound FindNode:
let dest_suffix = Address::empty()
    .component(self.peer_kademlia_cref(peer))
    .op("FindNode")
    .to_bytes();
let wid = self.next_wire_req_id();
let Some(addrs) = ctx.address_book.lookup(peer) else { return; };
let dest_peer_addresses: Vec<Vec<u8>> =
    addrs.iter().map(|a| a.to_bytes()).collect();
ctx.outbound_queue.push(WireEnvelope {
    dest_peer_addresses,
    fills: vec![SlotFill {
        dest_suffix,
        payload: bincode::serialize(&FindNodeReq { target_id })?,
        trigger_only: false,
    }],
    correlation: Some(WireCorrelation::request(wid)),
});

// In dispatch_atomic for op_type == "FindNodeReply":
let wid = inputs.get_opaque::<u64>("correlation")?;
self.fulfill_find_node_request(wid, payload)?;
```

The framework doesn't interpret control-plane correlation — that's
the component's choice — but the field exists in the envelope so
components don't have to invent their own correlation token system.

### 6.4 Surfacing CommandIds to user graphs

When the user writes:

```rust
let cmd = self.kademlia.put(g, key, value);
```

the `Kademlia` DSL handle records a NodeProto under the
`bb.kademlia v1` (or whatever) opset. At runtime the engine
dispatches via the bound `ProtocolRuntime::dispatch`:

```rust
fn dispatch(&mut self, op_type: &str, inputs: &TypedInputMap, ctx: &mut RuntimeResourceRef)
    -> OpResult
{
    match op_type {
        "Put" => {
            let cmd_id = CommandId::allocate();
            let lookup_state = self.start_lookup(...);
            self.pending_lookups.insert(cmd_id, lookup_state);
            // Push the initial FindNode envelopes (see 6.2).
            // Mark this CommandId as in-flight.
            OpResult::Async(cmd_id)
        }
        _ => OpResult::Failed(OpError::Custom(format!("unknown op {op_type}"))),
    }
}
```

Later (possibly many cycles later), from inside `dispatch_atomic`
the component fulfills:

```rust
if self.lookup_complete(lookup_state) {
    let result_value: StoreResult = self.finalize_lookup(&lookup_state);
    let trigger: Box<dyn SlotValue> = Box::new(TriggerValue);
    let result: Box<dyn SlotValue> = Box::new(result_value);
    ctx.complete_command(cmd_id, vec![trigger, result]);
}
```

The engine drains `pending_completions` after the hook returns +
calls `handle_completion(cmd_id, [trigger, result])` — the user's
graph cascade downstream of `kademlia.put` resumes with the result
value.

Six FindNode round-trips + N STORE messages happen invisibly
inside the component. The user's graph saw exactly one Op
(`Put`) that took time to complete and produced a result.

---

## Part 7 — Custom message types

The framework requires zero special handling for new message types
beyond a thin contract for each plane.

### 7.1 Data-plane custom types

To send a custom type over the data plane, the user defines a
struct that derives serde + Clone:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub epoch: u32,
    pub loss: f32,
    pub accuracy: f32,
}
```

That's the entire wire-eligibility contract — the
`impl<T: Clone + Serialize + DeserializeOwned + ...> SlotValue for T`
blanket in `src/slot_value.rs` covers the rest.

Use it in a Module:

```rust
let report = self.compute_report.compute(g, gradients);
// ... down the graph ...
wire_send(g, report, supervisor_peer);
```

That's all the user writes. The framework:
- Encodes via `bincode::serialize(self)` at the wire boundary.
- The receiver's downstream-consumer Op decodes via
  `bincode::deserialize::<TrainingProgress>(bytes)` against the
  type the graph contract guarantees at that slot.
- Ships envelopes with `message_type: "user.training_progress"`,
  `hash: TrainingProgress::META.hash`.
- Decodes on the receiver via the registered entry.
- Routes to the receiver's `WireRecv` Op declared with the matching
  TypeNode.

### 7.2 Control-plane custom messages

A `ProtocolRuntime` component defines its own message struct(s) and
serializes / deserializes them however it likes. No `WireType` impl
needed; the component is the only thing that touches the bytes.

```rust
#[derive(Serialize, Deserialize)]
struct FindNodeReq { target_id: [u8; 32] }

#[derive(Serialize, Deserialize)]
struct FindNodeReply { nearest_peers: Vec<(PeerId, [u8; 32])> }

#[derive(Serialize, Deserialize)]
struct PingReq;

#[derive(Serialize, Deserialize)]
struct PingReply;
```

The component owns the message_type → struct mapping internally
via `match` in `dispatch_atomic`. The framework does not need to know.

### 7.3 Reserved denotations vs user namespace

The framework reserves the `bb.*` denotation namespace for built-in
`TypeNode` statics (`TRIGGER_TYPE`, `PEER_ID_TYPE`, the primitive
type metas in `src/wire.rs`, etc.) and the `ai.bytesandbrains.*`
opset domain namespace for framework opsets. User custom types
belong in the `user.*` or `<vendor>.*` namespace by convention:

- `user.training_progress` — fine
- `myapp.checkpoint_summary` — fine
- `bb.*` — reserved for the framework.

---

## Part 8 — Triggers cross the wire too

The framework's `Trigger` type is a unit / zero-byte signal. When
a graph edge crossing a Node boundary is **trigger-only**, the
envelope still ships — receiver synthesis depends on it — but the
payload bytes are empty.

### 8.1 The compiler's TriggerOnly classification

Per COMPILER.md §9.1: the L21 partition pass classifies every
cross-Node edge as `Data` or `TriggerOnly` based on what the
receiver's downstream consumers actually read:

- **Data** — at least one downstream consumer of the edge reads
  the typed value. Envelope ships the full encoded payload bytes;
  receiver decodes per the declared TypeNode; trigger-expecting
  downstream Ops coerce on read.
- **TriggerOnly** — every downstream consumer's declared input
  port has `TypeNode.denotation == "bb.trigger"`. Envelope ships
  with zero payload bytes; receiver synthesizes Trigger; saves
  bandwidth.

Classification rule (single edge `producer → V → receiver_graph`):

> The edge is **TriggerOnly** iff every downstream consumer of V in
> the receiver graph declares its input port type as
> `bb.trigger`. Any other downstream port (including primitives,
> peer ids, tensors, custom types) → the edge is **Data**.

Classification is per-edge, recorded at install time via
`metadata_props["ai.bytesandbrains.wire_transport"] = "data" | "trigger_only"`.

### 8.2 The on-wire encoding

A trigger-only fill in a `SlotFillBatch`:

```protobuf
SlotFill {
  graph_name: "ServerModule",
  input_name: "barrier_input",
  value_bytes: [],        // empty
  trigger_only: true
}
```

A trigger-only single-fill envelope (un-batched case):

```
WireEnvelope {
  opset: ai.bytesandbrains.wire v1,
  message_type: "bb.trigger",
  payload: [],
  correlation: None,
  peer_id: <peer>,
  hash: <Trigger::META.hash>
}
```

Both encode to ~30 bytes on the wire (protobuf framing overhead +
small field tags + no payload). The receiver decodes, sees
`trigger_only: true` or `message_type: "bb.trigger"`, synthesizes a
zero-byte Trigger value, writes to the slot.

### 8.3 Why send a trigger if it has no data

Distributed coordination. The classic "barrier" pattern: producer
on Node A finishes some computation; receiver on Node B has work
that depends on "Node A finished" but doesn't care about the
value. Without a wire envelope, Node B has no way to know.
TriggerOnly = explicit "I happened" signal at minimum bandwidth.

The whole point is the same value flow the user authored stays
intact: the producer's DSL call wrote a `TriggerValue`-carrying
`Output` to its output port; the compiler noticed the downstream is
trigger-only;
the runtime ships zero bytes; the receiver gets `Trigger` in its
slot; the cascade continues.

---

## Part 9 — Arbitrary data baked into the graph

The graph IR carries the full type information for every value
crossing the wire. The runtime doesn't negotiate types
out-of-band; everything needed for the receiver to decode is in
the graph the receiver loaded.

### 9.1 Type information is in the IR

A `WireRecv` Op's `payload_type` attribute is a `TypeProto`
(ONNX canonical, §11 of IR_AND_DSL.md):

```protobuf
NodeProto {
  op_type: "Recv",
  domain: "ai.bytesandbrains.wire",
  output: ["trigger_site", "payload_site"],
  attribute: [
    AttributeProto {
      name: "payload_type",
      type: TYPE_PROTO,
      tp: TypeProto {
        opaque_type: Opaque {
          domain: "ai.bytesandbrains",
          name: "PeerId"
        }
      }
    }
  ]
}
```

Or for tensors:

```protobuf
attribute: [
  AttributeProto {
    name: "payload_type",
    type: TYPE_PROTO,
    tp: TypeProto {
      tensor_type: Tensor {
        elem_type: FLOAT,
        shape: { dim: [Dimension { dim_param: "batch" }] }
      }
    }
  }
]
```

Or for arbitrary user types:

```protobuf
attribute: [
  AttributeProto {
    name: "payload_type",
    type: TYPE_PROTO,
    tp: TypeProto {
      opaque_type: Opaque {
        domain: "user",
        name: "training_progress"
      }
    }
  }
]
```

The user authored `self.wire.recv(g, &TRAINING_PROGRESS_META)`; the
DSL recorded the TypeProto on the Op; the loaded graph
**self-describes** every value's type. The receiver's decoder
table is keyed by the hash derived from the denotation; the graph
has the denotation; everything aligns automatically.

### 9.2 The graph is the contract

Two Nodes communicating must agree on the type each edge carries.
The contract IS the receiver's graph: the `WireRecv`'s declared
`payload_type` is the truth; the sender's `WireSend` must produce
a value whose denotation matches.

Validation at install:
- Both ends of a cross-Node edge declare the same `TypeNode`
  (via the receiver's Recv attribute matching the sender's Send
  type).
- Receiver's decoder table has an entry for the declared
  denotation.
- Sender's encoder (the WireType impl) produces bytes the
  receiver's decoder can read.

These constraints are checked at Node `build()` time, surfaced
as `LoadError` variants, never as runtime decode failures.

### 9.3 Data + triggers are uniform

Whether the value crossing is a Trigger (zero-byte), a u64
(8 bytes), a tensor (megabytes), or a user-defined opaque type
(arbitrary serde-derivable struct), **the machinery is identical**:

- Author declares the type on both ends via the DSL.
- Compiler classifies edges (trigger-only optimization for triggers).
- Framework encodes via the WireType impl.
- Envelope ships.
- Framework decodes via the per-Node decoder table.
- Receiver's slot gets the typed value.

No special-casing per type. Adding a new wire-traversable user
type is: define struct + serde derive + `WireType` impl +
register at Node. ~15 lines of code per type.

---

## Part 10 — Cross-Node correlation

The `WireCorrelation` field on the envelope handles
request/response semantics. Three states (matching the proto enum):

- `None` — fire-and-forget. Receiver routes by `(opset, message_type,
  hash)` to a matching Recv Op.
- `Request(wire_req_id)` — sender opens a request. Receiver records
  `(wire_req_id → peer_id)` on its Wire component; routes payload
  to a `WireRecvReq` Op; later, a matching `WireSendResp` consults
  the inbound map to send back to `peer_id` with
  `Response(wire_req_id)` correlation.
- `Response(wire_req_id)` — receiver matches against its in-flight
  request map; on match, signals the suspended
  `WireSendReqBatched` Op's CommandId; on N-of-N completion fires
  the `responses` output downstream.

`WireRequestId` is a Node-scoped sender-allocated u64. Different
senders may use overlapping ids (the receiver's in-flight map is
per-peer-keyed implicitly through which Wire component records
it). The framework's `Wire` component handles allocation +
tracking; users see only the DSL.

The same `WireCorrelation` field is available for control-plane
opsets — protocols MAY pair their own messages by it (Kademlia
FindNode/FindNodeReply, Ping/Pong) — without inventing a separate
correlation mechanism.

---

## Part 11 — Slot-fill batching deep-dive

Per cycle, the compiler's L21 partition + the runtime's outbound
collation work together to minimize envelope count.

### 11.1 Grouping rule

Outbound envelopes are grouped by `(destination_peer_id,
opset, message_type=="bb.slot_fill_batch")`. Multiple slot fills
to the same `(peer_id, message_type=batch)` pack into one
`SlotFillBatch` payload, one envelope, one transport send.

```
Cycle drains outbound_queue → 7 pending envelopes:
  3 → peer 100, all single-fill data plane
  2 → peer 100, multi-fill batch
  1 → peer 200, single trigger-only
  1 → peer 100, control-plane to "user.gossip"

Final EngineSteps:
  SendEnvelope(batch to peer 100, message_type=bb.slot_fill_batch,
               5 fills inside)
  SendEnvelope(single envelope to peer 200, trigger-only)
  SendEnvelope(control envelope to peer 100, opset=user.gossip v1)
```

Three transport sends instead of seven.

### 11.2 Cap on batch size

Default: 64 fills per batch. Configurable via
`NodeConfig::with_max_batch_fills(usize)`. Exceeding the cap
spills to a second batch envelope (same `(peer, opset)`, second
SlotFillBatch payload).

### 11.3 Trigger-only fills are tiny

A trigger-only fill is ~3 protobuf bytes (the name strings dominate
size). Packing 64 trigger-only fills into one batch is ~200 bytes
of envelope payload + ~80 bytes of envelope framing = ~280 bytes
on the wire for 64 distributed-barrier signals. Without batching,
that'd be ~64 × 30 bytes = ~1.9 KiB across 64 separate sends.

### 11.4 Batching never crosses opsets

Different opsets never share an envelope. Each opset's payload
encoding is owned by either the framework (data plane) or the
controlling component (control plane); mixing would require
multi-opset envelopes which the framework explicitly does not
support.

---

## Part 12 — Encryption + authentication

**Out of scope for the framework.** The wire envelope is
plaintext-by-default; the transport adapter is responsible for
encryption + authentication.

Typical stacks:

- **libp2p transport** wraps the envelope bytes inside Noise / TLS
  streams over QUIC or TCP. Authentication via libp2p's peer-id
  signing.
- **HTTP transport** wraps envelope bytes inside TLS over TCP. Peer
  identity via mutual TLS or app-layer auth tokens.
- **In-process simulator** ships raw bytes between Node instances
  in the same process; no crypto needed.

The framework's `PeerId` is a [multihash][mh] — wire-format
compatible with `libp2p_identity::PeerId`. Different overlay
protocols pick their own digest function (identity-encoded keys
for ≤42-byte public keys, sha2-256 for larger keys, sha3,
blake2b, …) without coordination. The framework itself does no
key verification — transports that need verified identities
(libp2p, federation deployments) derive the PeerId from a public
key and verify on each envelope receipt before calling
`Node::deliver_inbound`. The multihash byte format means a
PeerId minted by libp2p round-trips through our `PeerId::from_bytes`
unchanged, and our `PeerId::to_bytes` produces bytes libp2p parses
without translation.

This separation keeps the framework crypto-agnostic. Deployments
choose their threat model + transport accordingly.

[mh]: https://github.com/multiformats/multihash

---

## Part 13 — Snapshot semantics for in-flight wire state

Per Rule 12 (full-state snapshots), the framework preserves enough
wire state to resume mid-cohort after restore.

### 13.1 What's snapshotted (data plane)

For each `Wire` component bound on the Node:

- `in_flight: HashMap<CommandId, InFlightRequest>` — outstanding
  send_req_batched cohorts. Includes `cohort_n`, accumulated
  responses, response_type_id, the wire_req_ids cycled.
- `pending_recv: VecDeque<PendingRecv>` — decoded inbound values
  waiting for the matching Recv Op to consume.
- `inbound_requests: HashMap<WireRequestId, PeerId>` — for inbound
  requests this Node is preparing to respond to.

Each Wire component's `snapshot() -> Vec<u8>` encodes these via
its own prost schema; restore deserializes.

### 13.2 What's snapshotted (control plane)

For each `ProtocolRuntime` component:

- The component's own internal state (k-buckets, peer views, age
  counters, pending_lookups, etc.) — encoded by the component's
  `snapshot()` impl.
- Scheduled `ComponentTimer` entries on the framework Scheduler —
  serialize alongside framework timers; restore reattaches.

### 13.3 What's NOT snapshotted

- The transport adapter's state (sockets, connection pools) —
  the host's responsibility on restart.
- In-progress envelope encoding/decoding work (purely runtime,
  intra-cycle).
- The waker (runtime-only).
- `Engine::ingress_bytes_in_flight`
  (`bb-runtime/src/engine/core.rs:274`) — the budget counter is
  derived state. Slot-table payloads survive the snapshot; their
  byte charges are recomputed at restore from
  `SlotValue::charged_bytes()` on each live entry, then the
  counter resumes as bytes drain through normal slot overwrite /
  eviction. In-flight ingress events that hadn't yet reached the
  slot table are not preserved (the host's transport adapter
  replays them).

### 13.4 Post-restore behavior

After `Node::restore(snapshot)`:

- Outstanding `send_req_batched` cohorts that haven't completed
  remain in the Wire's `in_flight` map. The matching responses,
  if they arrive post-restore (transport reconnects), fulfill
  them normally.
- Responses that arrived AT the peer but were in flight during
  snapshot are lost. Components MUST tolerate retry / timeout —
  this is application protocol design, not framework
  responsibility.
- `Kademlia`-style protocols handle the loss naturally via their
  parallel lookup retry logic.
- Stateful protocols without retry semantics SHOULD layer their
  own resend / acknowledgment mechanism.

---

## Part 14 — Worked examples

### 14.1 Sending a custom user value

```rust
// Define the type. The single derive line is the entire wire-
// eligibility contract — the `SlotValue` blanket covers serialize,
// deserialize, clone, and dyn-safe slot residency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDelta {
    pub round: u32,
    pub weights: Vec<f32>,
}

// Module sending it.
struct TrainerModule {
    model: BurnModel,
    aggregator_peers: Constant<Vec<PeerId>>,
}

impl Module for TrainerModule {
    fn name(&self) -> &str { "Trainer" }
    fn op(&self, g: &mut Graph, _inputs: &[Output]) -> Vec<Output> {
        // ... training graph recording ...
        let delta = self.model.compute_delta(g);
        let peers = self.aggregator_peers.value(g);
        wire_send(g, delta, peers);  // ships ModelDelta via bincode
        vec![]  // no top-level outputs
    }
}

// Application entry point.
let compiled = Compiler::new()
    .bind_backend::<BurnBackend>("compute")
    .compile(TrainerModule { /* ... */ })?;

let node = bb::install(
    my_peer_id,
    my_addr,
    compiled,
    "TrainerModule",
    Config::new().with("compute", burn_config),
)?;
```

The user wrote one thing related to the wire: the struct + serde
derives (3 lines). The whole serialization / batching / routing /
decoding pipeline is the framework's. Wire is engine-native; no
binding step.

### 14.2 Implementing a custom Gossip protocol

```rust
// Message types.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct GossipPush {
    view: Vec<(PeerId, u64)>,  // (peer, last_seen_ns)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GossipPull;

// The protocol.
pub struct MyGossip {
    view: HashMap<PeerId, u64>,
    period_ns: u64,
}

impl MyGossip {
    pub fn new(period_ns: u64) -> Self {
        Self { view: HashMap::new(), period_ns }
    }
}

impl ProtocolRuntime for MyGossip {
    type Error = ProtocolError;

    fn atomic_opset(&self) -> AtomicOpsetDecl {
        AtomicOpsetDecl {
            domain: "myapp.gossip",
            version: 1,
            ops: &[
                AtomicOpDecl { name: "Sample", /* IO */ kind: AtomicOpKind::Immediate },
                AtomicOpDecl { name: "Push",   /* IO */ kind: AtomicOpKind::Immediate },
                AtomicOpDecl { name: "Pull",   /* IO */ kind: AtomicOpKind::Immediate },
            ],
        }
    }

    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error> {
        match op_type {
            // User-graph DSL op: pick a random known peer.
            "Sample" => {
                let peer = self.pick_random_peer();
                Ok(DispatchResult::Immediate(vec![
                    ("peer".into(), Box::new(peer) as Box<dyn SlotValue>)
                ]))
            }
            // Inbound envelope: framework synthesizes peer_id + payload.
            "Push" => {
                let peer_id: PeerId = inputs.get_opaque("peer_id")?;
                let payload: &[u8] = inputs.get_opaque("payload")?;
                let msg: GossipPush = bincode::deserialize(payload)?;
                self.merge_view(peer_id, msg.view);
                Ok(DispatchResult::Immediate(vec![]))
            }
            "Pull" => {
                // Inbound pull request — reply with our view.
                let peer_id: PeerId = inputs.get_opaque("peer_id")?;
                let reply = GossipPush { view: self.view_as_vec() };
                ctx.outbound_queue.push(WireEnvelope::control(
                    OpsetId::new("myapp.gossip", 1),
                    "Push",
                    bincode::serialize(&reply).unwrap(),
                    WireCorrelation::None,
                    peer_id,
                ));
                Ok(DispatchResult::Immediate(vec![]))
            }
            _ => Err(ProtocolError::UnknownMessageType { /* … */ }),
        }
    }

    fn on_timer(&mut self, _kind: ComponentTimerKind, ctx: &mut RuntimeResourceRef<'_>)
        -> Result<(), Self::Error>
    {
        let peer = self.pick_random_peer();
        let push = GossipPush { view: self.view_as_vec() };
        ctx.outbound_queue.push(WireEnvelope::control(
            OpsetId::new("myapp.gossip", 1),
            "Push",
            bincode::serialize(&push).unwrap(),
            WireCorrelation::None,
            peer.0,
        ));

        ctx.scheduler.schedule_component_timer(
            ctx.current_component_ref,
            ComponentTimerKind(0),
            now_ns() + self.period_ns,
        );
        Ok(())
    }
}

// Compile + install via the canonical three-phase chain.
let compiled = Compiler::new()
    .bind_protocol::<MyGossip>("gossip")
    .compile(ModuleHostingGossip { /* ... */ })?;

let node = bb::install(
    my_peer_id,
    my_addr,
    compiled,
    "ModuleHostingGossip",
    Config::new().with("gossip", MyGossipConfig { interval_ns: 1_000_000_000 }),
)?;
```

The component is ~50 lines of real protocol logic; the framework
provides everything else (routing, scheduling, persistent state,
CommandId completion if needed). New protocols ship as crates
without touching framework internals.

### 14.3 A cross-Node trigger barrier

```rust
struct Coordinator {
    wire: Wire,
    syscall: Syscall,
}

impl Module for Coordinator {
    fn name(&self) -> &str { "Coordinator" }
    fn op(&self, g: &mut Graph, _inputs: &[Output]) -> Vec<Output> {
        // 5 workers' "I am done" signals arrive as triggers via wire.
        let (worker_done, _) = self.wire.recv(g, &TRIGGER_META);

        // Threshold: when 5 triggers accumulated, fire.
        let all_done = self.syscall.threshold(g, worker_done, 5);

        // Fan out "go ahead" trigger to all workers.
        let workers: Output = /* known peer list */;
        let go_signal = self.syscall.passthrough(g, all_done);
        self.wire.send(g, go_signal, workers);
        vec![]
    }
    // ... components() ...
}
```

On the wire: each "done" envelope is ~30 bytes (trigger-only).
Each "go" envelope is also ~30 bytes. 10 envelopes total per
barrier (5 incoming, 5 outgoing). The graph carries the type
information; the receiver synthesizes Trigger on inbound; the
framework batches multiple "go" sends to the same destination
into one envelope where possible. No user code touches transport.

---

## Part 15 — The transport adapter contract

A transport adapter sits between the framework and the network. It
runs on host threads (potentially many) and bridges bytes:

```rust
// 1. Outbound: pull EngineStep::SendEnvelope from poll() results
//    and ship them via the transport.
loop {
    let steps = node.poll(&mut cx).await;
    for step in steps {
        if let EngineStep::SendEnvelope(env) = step {
            let bytes = env.encode_framed();
            transport.send(env.peer(), bytes).await?;
        }
    }
}

// 2. Inbound: receive bytes from the transport, decode, push.
loop {
    let (peer_id, bytes) = transport.recv().await?;
    let env = WireEnvelope::decode_framed(&bytes)?;
    let ingress = node.ingress_handle();
    ingress.push(IngressEvent::Envelope(env));
}
```

The transport adapter:

- Owns sockets, connection management, TLS / Noise / etc.
- Maps the framework's `PeerId` (a `Multihash<64>`) to its own
  transport-layer addressing (TCP sockets, QUIC connection IDs,
  etc.). The PeerId byte format is libp2p-compatible, so a libp2p
  adapter passes the bytes through without translation.
- Decides retry / timeout / reconnect semantics per its protocol.
- Pushes inbound envelopes through `IngressQueue::push` (the
  framework's only thread-safe seam).
- Pulls outbound envelopes from `EngineStep::SendEnvelope` and
  ships their `encode_framed()` bytes.

**Envelope-decode allocation.** `WireEnvelope::decode_framed`
runs `prost::Message::decode` after the adapter passes the
borrowed byte slice in. Prost allocates per-field storage
internally; `EnvelopeCaps::max_total_bytes`
(`bb-runtime/src/envelope.rs:197-207`, default 16 MiB, edge
preset 256 KiB) gates the decode-side budget. Oversize frames
fail decode before allocation, so the adapter sees a typed
`EnvelopeError` instead of an allocator panic. Once decoded,
`fill.payload` is a framework-owned `Vec<u8>` that the engine
either hands to `Backend::materialize_from_wire` via
`mem::take` (zero memcpy in the tensor path) or borrows
`&fill.payload` to the framework-carrier decoder.

The framework provides:

- A canonical envelope encoding (`encode_framed` / `decode_framed`).
- A guarantee that envelopes from the same Node are well-formed
  + framable.
- The `IngressQueue` for thread-safe inbound submission.

That's the entire interface. The transport adapter knows nothing
about graphs, slots, opsets, decoders, components, or Modules; it
ships bytes between PeerIds. Multiple transport adapters can
coexist on one Node (one libp2p adapter + one in-process simulator
adapter on the same Node, each handling a subset of peers).

---

That's the wire spec. One canonical envelope format; two semantic
planes routed by opset; framework-owned data-plane encoding for
graph edges; component-owned control-plane encoding for protocol
messages; triggers and arbitrary user types pass identically
through the same machinery; the app sees DSL methods, the
component sees `outbound_queue` + `dispatch_atomic`, and
the transport adapter sees bytes.
