# Toolkit reference

> CHECK THIS BEFORE WRITING NEW CODE.
>
> If an op solves your problem, USE it by recording a NodeProto with
> its literal (domain, op_type) strings. NEVER hand-roll behavior the
> framework already provides.
>
> When you add a new op / Component / DSL helper / slot method, you
> MUST add it to this file in the same commit. The per-commit gate
> runs `scripts/check_toolkit_currency.py` which fails on drift.

## Principles (don't fight these)

- **Bootstrap is a graph.** The user composes their initialization
  sequence inside `Module::bootstrap` exactly the way they compose
  `Module::body`. Components contribute ops the user records
  (`GlobalRegistryClient::Announce`, `Index::train`,
  `BackendSlot::prime`, …); the Module's bootstrap function
  orchestrates them. `Node::run_bootstrap(&[])` fires every
  install-order target; `Node::run_bootstrap(&[BootstrapInput,
  ...])` re-fires a named target with host-supplied input bytes.
  No separate Component bootstrap dispatch path — initialization
  is just the bootstrap graph reaching the relevant ops.

## Auto-handled behaviors (don't reinvent these)

- **Wire receive is compiler-synthesized.** `bb_compiler::synthesize_wire_recvs()` plants `ai.bytesandbrains.wire::Recv` placeholders on receiver partitions; data-plane delivery seeds the slot directly via `deliver_fill`. Never hand-roll `wire.Recv` in a graph.
- **GlobalRegistry Announce throttles itself.** `GlobalRegistryClient` compares `last_announce_ts_ns` vs `last_heartbeat_interval_ns` inside `dispatch_atomic`. Calling Announce in a tight loop is safe; sub-interval calls are silent no-ops. Don't build a periodic Heartbeat op.
- **GlobalRegistry Handshake reply is automatic.** `bind_protocol::<GlobalRegistryServer>` causes Announce on the server to emit a Handshake envelope back to `GLOBAL_REGISTRY_CLIENT_CREF`. The client `Handshake` op is framework-routed on inbound delivery.
- **Local addresses are framework-supplied.** `ctx.local_addresses()` is auto-populated from install. Never `Constant(local_addresses)` then feed into Announce.
- **Inbound src_peer is framework-supplied.** RX gates and aggregator `Contribute` read `ctx.current.inbound.src_peer` — never carry the source peer as a NodeProto attribute.
- **Gate ops are compiler-inserted.** `DedupGateRx`, `PeerHealthGateTx`, `BackoffGateTx`, `PeerHealthGateRx`, `BackoffGateRx`, and `DeadlineCheck` are all stamped by compiler passes (`insert_dedup_gate_rx`, `insert_peer_health_gate_tx`, `insert_backoff_gate_tx`, `insert_peer_health_gate_rx`, `insert_backoff_gate_rx`, `insert_async_deadlines`). Don't author them in user graphs.
- **All-inputs-ready is built into engine dispatch.** Op bodies see every input present. `Gate` and `GateDispatch` are pure forwarders — engine readiness does the work.
- **`EventSource` subscriptions are framework-routed.** `Engine::register_event_subscription` wires the op onto the bus by parsing the `kind` attribute at install time. Body just emits a Trigger on arrival.
- **`LifecyclePhase` is phase-gated by the engine.** `Engine::register_lifecycle_op` parses the `phase` attribute at install. Ops only land on the frontier when `fire_lifecycle(phase)` runs.
- **Async timer completions reroute automatically.** `After`, `Sleep`, and `BootstrapOutput` return `Async(cmd)`; the engine parks them in `pending_async[cmd]` and routes timer/host completion via `handle_completion`. Don't poll.
- **`Any` and `DeadlineMatch` latch winners.** Per-(group / OpRef, ExecId) latch absorbs repeat arrivals via `ctx.syscall.any_fired_groups` / `ctx.syscall.deadline_match_fired`. Don't add your own dedup wrapper.
- **Backend tensor ops have placeholder slot methods.** Don't hand-record `ai.onnx::Add` — call `BackendSlot::add(g, a, b)` so the right backend lands at install.
- **FedAvg buffer is per-round + deterministic.** `BTreeMap<PeerId, …>` gives lexical-peer-id order; duplicate `Contribute` from the same peer in the same round REPLACES the prior entry. Don't add per-peer dedup upstream.
- **Address book ref-counting is automatic.** `AddressBook::InsertMany` dedupe-appends; ref-counted entries; lazy eviction in `GlobalRegistryServer` Sample / CurrentView reads. Don't hand-manage TTLs.

## Before-you-write checklist

1. Does it match a placeholder slot method? Use the slot.
2. Does it match a registered syscall op? Use the op_type literal.
3. Does it match a wire op? Use `g.net_out` (Send) or `g.input` (Recv).
4. Does it look like discovery / heartbeats? GlobalRegistry has it.
5. None of the above? Justify a new op in the PR description.

## Components (bind_* targets)

<!-- toolkit:components -->

| Role | bind_method | Type | Ships with framework | Code | Notes |
|---|---|---|---|---|---|
| Aggregator | `bind_aggregator` | `FedAvg<B>` | yes | `bb-ops/src/aggregators/fedavg/mod.rs:97` | Federated-averaging aggregator generic over a Backend B. Composes the weighted-mean reduction from B's Mul + Add primitives so the 30-op floor stays unchanged. Per-round `BTreeMap<PeerId, (B::Tensor, u64)>` buffer gives deterministic lexical-peer-id walk; duplicate contributions from the same peer in the same round REPLACE the prior entry. Aggregate emits cumulative `num_samples` for hierarchical weighting. Buffer is `#[serde(skip)]` — structural identity is snapshot, transient round state is not. Trust model: NaN/Inf poisons the round per IEEE 754 — defenses belong at the contribution boundary. |
| Backend | `bind_backend` | `CpuBackend` | yes | `bb-ops/src/backends/cpu/mod.rs:155` | Pure-Rust reference CPU backend. Implements `bb::Backend` for the ai.onnx v1 51-op subset over ndarray. Storage is `ArrayD<f32>` end-to-end (`CpuTensor` wraps `Arc<CpuBackendBuffer>`). Single allocation seam via `alloc_tensor` / `wrap_array` for future pool/arena strategies. Gated by the `cpu-backend` feature. Uses `#[derive(bb_derive::Backend)]` for the RoleRuntime bridge. |
| PeerSelector | `bind_peer_selector` | `ConstantView` | yes | `bb-ops/src/protocols/constant_view/mod.rs:52` | Fixed-list peer selector. Authors construct with the full peer set at install time; queries return slices per `SelectParams` (All / Random{n} / NearKey). Random sampling uses a small xorshift seeded by `seed` so the same seed over the same peer set produces the same sequence — useful for tests and tiny fixed-size deployments where the peer set is known at install time and never changes. Returns `ConstantViewError::Empty` if asked to sample from an empty view. Uses `#[derive(bb_derive::PeerSelector)]` for the RoleRuntime bridge. |
| Index | `bind_index` | `KademliaHand` | yes | `bb-ops/src/protocols/kademlia_hand/mod.rs:51` | Test-only Index Contract impl. Stub add/search/remove that return deterministic values derived from `seed` so tests can assert exact return values. Part of a multi-role test component re-exported under `bb_ops::test_components::KademliaHand`. Used by integration tests as a stand-in for a real Kademlia DHT peer. Uses `#[derive(bb_derive::Index)]` for the RoleRuntime bridge. |
| PeerSelector | `bind_peer_selector` | `KademliaHand` | yes | `bb-ops/src/protocols/kademlia_hand/mod.rs:81` | Test-only PeerSelector Contract impl on the same `KademliaHand` struct. `select()` and `current_view()` return empty `Vec<PeerId>` — structural stubs whose only job is to satisfy the Contract surface so `dispatch_atomic` calls can be recorded by tests. Multi-role binding: the same `KademliaHand` type satisfies Index, PeerSelector, and ProtocolRuntime simultaneously. Uses `#[derive(bb_derive::PeerSelector)]` for the RoleRuntime bridge. |
| Protocol | `bind_protocol` | `KademliaHand` | yes | `bb-ops/src/protocols/kademlia_hand/mod.rs:142` | Hand-written ProtocolRuntime impl — skips `register_protocol!{}` because that macro would conflict with the `#[derive(bb::Concrete)]` above by re-emitting the same struct + ConcreteComponent. Declares opset `test.Kademlia.atomic` with FindNode (Async), Sample (Immediate), Search (Async). `dispatch_atomic` appends each `op_type` name to a shared `Arc<Mutex<Vec<String>>>` calls log so tests can assert dispatch ordering. Inventory submits `ComponentRoleBinding{Protocol}` manually since the derive doesn't cover ProtocolRuntime. |
| Protocol | `bind_protocol` | `GlobalRegistryClient` | yes | `bb-ops/src/protocols/global_registry/mod.rs:128` | Client half of the GlobalRegistry federation-membership protocol. Bound on every client Node. Declares Announce (with `server_peer` PeerId input) and Handshake atomic ops in the `ai.bytesandbrains.protocol.global_registry` domain. On Announce dispatch ships an envelope carrying `ctx.current.self_peer` to the server's well-known `GLOBAL_REGISTRY_SERVER_CREF` (=0). Heartbeat throttle — sub-interval calls are silent no-ops. Handshake reply refreshes `last_assigned_ttl_ns` / `last_heartbeat_interval_ns`; `last_announce_ts_ns` is `#[serde(skip)]` so bootstrap re-seeds the cadence on resume. The protocol carries no static peer identity — `server_peer` is wired through the graph at every Announce. |
| PeerSelector | `bind_peer_selector` | `GlobalRegistryServer` | yes | `bb-ops/src/protocols/global_registry/mod.rs:394` | Server half of GlobalRegistry exposed as a PeerSelector source over the announced cohort. `select()` supports All / Random{n} / NearKey{key,n}. Eviction is lazy — `evict_expired()` drops entries whose `expires_at_ns` has elapsed at the top of every Sample / CurrentView read, releasing AddressBook references. Random selection mixes a per-call `AtomicU64 sample_counter` with the persisted `seed` so successive `sample(n)` returns vary on a constant seed; counter is `#[serde(skip)]` and reset to zero on snapshot/restore so restored servers replay sampling from a known starting point. `AtomicU64` is the minimal lock-free shape that fits the Contract's `&self`-only select signature. |
| Protocol | `bind_protocol` | `GlobalRegistryServer` | yes | `bb-ops/src/protocols/global_registry/mod.rs:459` | Server half of GlobalRegistry as a Protocol. Declares Sample, CurrentView (both with opaque cookie for libp2p-style incremental discovery; v1 ships full cookies), and Announce in the `ai.bytesandbrains.protocol.global_registry` domain. Announce accepts inbound envelopes, registers the announcing peer in the runtime AddressBook under a server-assigned TTL bounded by `GlobalRegistryServerConfig` (default 90s, min 30s, max 300s), and replies with a Handshake carrying `(assigned_ttl_ns, heartbeat_interval_ns=ttl/3, server_addresses)`. Reads announcing peer from inbound payload — carries no static peer identity. Pairs with `GlobalRegistryServer` PeerSelector impl — dual binding sites (`bind_protocol` + `bind_peer_selector`). |

<!-- /toolkit:components -->

## Atomic ops by domain

### `ai.bytesandbrains.syscall`

<!-- toolkit:ops domain="ai.bytesandbrains.syscall" -->

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| Interval | | tick: Timestamp | period_ns: i64 | Immediate | `bb-ops/src/syscalls/triggers/interval.rs:54` | Reads `period_ns` attr (default 1_000_000_000); schedules next firing via `TimerKind::Interval` on `ctx.time.scheduler`; engine scheduler maturity drain re-arms next firing. Emits current timestamp. |
| OnTrigger | trigger: Trigger | trigger: Trigger | | Immediate | `bb-ops/src/syscalls/triggers/on_trigger.rs:38` | Identity pass-through for a Trigger; requires at least one input. |
| EventSource | | event: Trigger | kind: string | Immediate | `bb-ops/src/syscalls/triggers/event_source.rs:39` | Framework-routed subscription: `Engine::register_event_subscription` wires the op onto the bus by parsing the `kind` attr at install time. Body fires Trigger on each event arrival. |
| After | | trigger: Trigger (via Async cmd) | delay_ns: i64 | Async | `bb-ops/src/syscalls/triggers/after.rs:45` | Allocates `CommandId`; schedules `TimerKind::After`; returns `Async(cmd)` so engine `handle_completion` delivers the delayed Trigger after maturity. |
| Pulse | | trigger: Trigger | | Immediate | `bb-ops/src/syscalls/triggers/pulse.rs:32` | One-shot bootstrap trigger; emits a single Trigger. |
| Limit.Acquire | trigger: Trigger | trigger: Trigger (or empty on deny) | name: string, n: int | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:268` | Calls `ctx.peers.gate.acquire(name, n)`; emits Trigger on acquire, empty Immediate on deny. |
| Limit.Release | trigger: Trigger | | name: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:277` | Calls `ctx.peers.gate.release(name)`. |
| Any | variadic | value: polymorphic | group: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:286` | First-arrival semantics; per-Op latch on group name absorbs repeat arrivals via `ctx.syscall.any_fired_groups`. Empty group disables latch. |
| Gate | value: polymorphic, trigger: Trigger | value: polymorphic | | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:295` | Engine all-inputs-ready already gates dispatch; body forwards value via `clone_boxed`. |
| Serialize.Enqueue | value: Bytes | trigger: Trigger | queue: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:304` | Pushes bytes onto `ctx.syscall.serialize_queue` under named queue. |
| Serialize.Dequeue | trigger: Trigger | value: Bytes | queue: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:313` | Pops bytes from `ctx.syscall.serialize_queue`; empty Immediate when queue empty. |
| CorrelateTag | trigger: Trigger | token: CorrelationToken | | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:322` | Mints a fresh token via `ctx.net.requests.mint_token()`. |
| Hold.Stash | value: Bytes | | slot: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:331` | Stashes bytes into `ctx.syscall.hold_table` under named slot. |
| Hold.Flush | trigger: Trigger | value: Bytes | slot: string | Immediate | `bb-ops/src/syscalls/coordination/mod.rs:340` | Flushes `ctx.syscall.hold_table`; empty Immediate when slot empty. |
| DeadlineCheck | trigger: Trigger | trigger: Trigger | deadline_ns: i64 (required) | Immediate | `bb-ops/src/syscalls/coordination/deadline_check.rs:67` | Compiler-inserted (`insert_async_deadlines` pass) upstream of every Async op carrying `deadline_ns`. Fails op with deadline-exceeded when `now_ns >= deadline_ns`. Complements engine Phase-5 `PendingAsync.deadline_ns` post-suspension scan. |
| DedupGateRx | value: polymorphic | value: polymorphic | | Immediate | `bb-ops/src/syscalls/gates/dedup_rx.rs:76` | Compiler-inserted (`insert_dedup_gate_rx` pass) at head of RX gate chain. FNV-1a hashes input wire bytes; consults `ctx.net.dedup.record`; drops envelope with duplicate reason on repeat. Framework-routed at compile time. |
| PeerHealthGateTx | trigger: Trigger | trigger: Trigger | peer: bytes (multihash, ATTR_PEER) | Immediate | `bb-ops/src/syscalls/gates/peer_health_tx.rs:87` | Compiler-inserted (`insert_peer_health_gate_tx` pass) upstream of every `wire::Send`. Consults `ctx.peers.governor.check_outbound`; emits Trigger on Allow, fails op with stable labels (blocklisted/not_allowlisted/cooldown) on Deny. |
| BackoffGateTx | trigger: Trigger | trigger: Trigger | peer: bytes (multihash, ATTR_PEER) | Immediate | `bb-ops/src/syscalls/gates/backoff_tx.rs:70` | Compiler-inserted (`insert_backoff_gate_tx` pass) between `PeerHealthGateTx` and `wire::Send`. Consults `ctx.peers.backoff.should_retry`; emits Trigger when retry-eligible, fails op with cooldown otherwise. |
| PeerHealthGateRx | value: polymorphic | value: polymorphic | | Immediate | `bb-ops/src/syscalls/gates/peer_health_rx.rs:81` | Compiler-inserted (`insert_peer_health_gate_rx` pass) between `wire::Recv` and consumers. Reads `ctx.current.inbound.src_peer` (framework-supplied per inbound envelope, never NodeProto attr). Consults `check_inbound`; forwards on Allow via `clone_boxed`, fails op with stable Deny label. |
| BackoffGateRx | value: polymorphic | value: polymorphic | | Immediate | `bb-ops/src/syscalls/gates/backoff_rx.rs:68` | Compiler-inserted (`insert_backoff_gate_rx` pass) in RX chain. Reads `ctx.current.inbound.src_peer`; consults `ctx.peers.backoff.should_retry`; forwards on retry-eligible, fails with cooldown otherwise. |
| LifecyclePhase | | trigger: Trigger | phase: string | Immediate | `bb-ops/src/syscalls/lifecycle/mod.rs:104` | Phase-gated firing: engine only pushes ops enrolled in `Engine.lifecycle_table[phase]` onto frontier when `fire_lifecycle(phase)` runs. `Engine::register_lifecycle_op` parses `phase` attr at install time. Body just emits Trigger. |
| BootstrapDispatch | | cmd: CommandId | | Immediate | `bb-ops/src/syscalls/lifecycle/mod.rs:113` | Mints `CommandId` via `ctx.allocate_command_id()` for downstream `BootstrapOutput` pairing. |
| BootstrapOutput | cmd: CommandId | trigger: Trigger (via Async cmd) | | Async | `bb-ops/src/syscalls/lifecycle/mod.rs:122` | Returns `Async(cmd)` so engine parks the op in `pending_async[cmd]` until host completes the command via ingress queue. |
| AppEmit | value: Bytes | | name: string | Immediate | `bb-ops/src/syscalls/telemetry/mod.rs:125` | Pushes onto `ctx.syscall.pending_app_events`; engine drains into `EngineStep::AppEvent`. Validates name against reserved prefixes (`bb.`, `ai.bytesandbrains.`) — collision returns `BadInput`. |
| AppNotify | trigger: Trigger | | name: string | Immediate | `bb-ops/src/syscalls/telemetry/mod.rs:134` | Pushes `AppEvent::notify` onto `ctx.syscall.pending_app_events`; same reserved-prefix validation as `AppEmit`. |
| Record | value: Bytes | | name: string | Immediate | `bb-ops/src/syscalls/telemetry/mod.rs:143` | Writes to `ctx.syscall.record_buffer` under named slot. |
| IncrMetric | trigger: Trigger | | name: string, delta: int | Immediate | `bb-ops/src/syscalls/telemetry/mod.rs:152` | Bumps `ctx.syscall.counters[name]` by delta (default 1). |
| Clock | | now: Timestamp | | Immediate | `bb-ops/src/syscalls/clock_rng/clock.rs:33` | Reads `ctx.time.scheduler.now_ns()`. |
| RngU64 | | value: u64 | | Immediate | `bb-ops/src/syscalls/clock_rng/rng_u64.rs:33` | Draws from `ctx.syscall.rng.next_u64()` (framework-seeded RNG). |
| DeadlineMatch | then: Trigger, timeout: Trigger | winner: Trigger | | Immediate | `bb-ops/src/syscalls/clock_rng/deadline_match.rs:50` | First-arrival selector; per-(OpRef, ExecId) latch via `ctx.syscall.deadline_match_fired` so subsequent invocations are absorbed once a winner is determined. |
| Sleep | | trigger: Trigger (via Async cmd) | duration_ns: i64 | Async | `bb-ops/src/syscalls/clock_rng/sleep.rs:42` | Schedules `TimerKind::Sleep(cmd)` on `ctx.time.scheduler` and returns `Async(cmd)`; engine routes completion via `handle_completion`. |
| GateDispatch | variadic | out: Trigger | | Immediate | `bb-ops/src/syscalls/sync/gate_dispatch.rs:47` | Generic N-input synchronization barrier. Engine all-inputs-ready check guarantees every input arrived before dispatch; body just emits Trigger. |
| PassThrough | value: polymorphic | value: polymorphic | | Immediate | `bb-ops/src/syscalls/structural/pass_through/mod.rs:52` | Structural identity syscall; forwards via `clone_boxed` preserving concrete type. (domain, op_type) re-exported from `bb_ir::syscall_ids`. |
| Tee | value: polymorphic | out_0, out_1, ... out_{fanout-1}: polymorphic | fanout: int (default 2) | Immediate | `bb-ops/src/syscalls/structural/tee/mod.rs:45` | Duplicates single input into `fanout` outputs via `clone_boxed`. (domain, op_type) re-exported from `bb_ir::syscall_ids`. |
| Constant | | value: Bytes (encoded TensorProto) | value: TensorProto | Immediate | `bb-ops/src/syscalls/structural/constant/mod.rs:50` | Emits the `value` attribute TensorProto bytes as `BytesValue`. Downstream tensor consumers decode via `Tensor::from_proto`. Distinct from `ai.onnx::Constant` primitive. |

<!-- /toolkit:ops -->

### `ai.bytesandbrains.wire`

<!-- toolkit:ops domain="ai.bytesandbrains.wire" -->

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| Send | data: polymorphic, peers: PeerIdVec | data: Trigger (structural placeholder), handle: WireReqId | `ai.bytesandbrains.dest_suffix.<input>`: bytes, `ai.bytesandbrains.trigger_only.<input>`: bytes, `deadline_ns`: int (optional), `ai.bytesandbrains.wire.chain_id` (metadata_props) | Immediate | `bb-ops/src/network/wire/mod.rs:441` | Wire opset (`ai.bytesandbrains.wire` v1) — engine-native, not user role. Resolves PeerId via `ctx.peers.addresses`; builds one SlotFill per non-peer input from compiler-stamped `dest_suffix`; supports ORIGINATOR/FORWARDER modes via inbound `wire_req_id` reuse; derives Dapper-style `remaining_deadline_ns`; stamps `src_peer_addresses`; registers in-flight via `ctx.net.requests`; failures emit `InfraEvent::PeerResolveFailure` rather than aborting fan-out. |
| Recv | | | | Immediate | `bb-ops/src/network/wire/mod.rs:450` | Framework-synthesized by `bb_compiler::synthesize_wire_recvs()` on receiver partitions. Pure structural placeholder — never lands on the frontier in normal flow; inbound data-plane delivery seeds slot directly via `deliver_fill`. Dispatch returns `Immediate(vec![])`. |

<!-- /toolkit:ops -->

### `ai.bytesandbrains.address_book`

<!-- toolkit:ops domain="ai.bytesandbrains.address_book" -->

Address-book ops are recorded via DSL helpers (`address_book_insert_many`, `address_book_lookup`) — see the DSL helpers table. Engine-native custom ops; no separate `bb-ops` entry beyond the helper-recorded `NodeProto`.

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| InsertMany | peer, addresses | trigger: Trigger | | Immediate | `bb-dsl/src/syscalls.rs:55` | Recorded via `address_book_insert_many` helper. New peer creates `ref_count=1` entry; known peer dedupe-appends; empty addresses vec is a dispatch-time `OpError`. |
| Lookup | peer | addresses: AddressVec | | Immediate | `bb-dsl/src/syscalls.rs:72` | Recorded via `address_book_lookup` helper. Returns full ordered vec; unknown / empty-address peer surfaces as dispatch-time `OpError`. |

<!-- /toolkit:ops -->

### `ai.bytesandbrains.protocol.global_registry`

<!-- toolkit:ops domain="ai.bytesandbrains.protocol.global_registry" -->

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| Announce (client) | server_peer: PeerId | wakeup: Trigger | | Immediate | `bb-ops/src/protocols/global_registry/mod.rs:111` | Client-side `AtomicOpDecl` static slice `GLOBAL_REGISTRY_CLIENT_OPS`. Heartbeat throttle built into `dispatch_atomic` (compares `last_announce_ts_ns` vs `last_heartbeat_interval_ns`). Snapshots local addresses into payload; ships envelope to GLOBAL_REGISTRY_SERVER_CREF; framework-routed via `ProtocolRuntime` dispatch. Inventory binding via `#[derive(bb::Concrete)]` on `GlobalRegistryClient`. |
| Handshake (client) | | wakeup: Trigger | | Immediate | `bb-ops/src/protocols/global_registry/mod.rs:119` | Client-side reply handler. Framework-routed: inbound envelope delivered to `GLOBAL_REGISTRY_CLIENT_CREF.Handshake`. Decodes server reply (`assigned_ttl_ns`, `heartbeat_interval_ns`, `server_addresses`) and merges addresses into `ctx.peers.addresses` for `src_peer`. |
| Sample (server) | count: i32, cookie: Bytes | peers: PeerIdVec, next_cookie: Bytes | | Immediate | `bb-ops/src/protocols/global_registry/mod.rs:436` | Server-side `AtomicOpDecl` static slice `GLOBAL_REGISTRY_SERVER_OPS`. Libp2p-style incremental discovery; routed via `#[derive(bb::PeerSelector)]` (Contract bridge). Lazy eviction runs at top of every Sample read. |
| CurrentView (server) | cookie: Bytes | peers: PeerIdVec, next_cookie: Bytes | | Immediate | `bb-ops/src/protocols/global_registry/mod.rs:443` | Server-side full cohort view. Routed via `#[derive(bb::PeerSelector)]`; lazy eviction at top of every read. |
| Announce (server) | | | | Immediate | `bb-ops/src/protocols/global_registry/mod.rs:450` | Server-side inbound handler. Framework-routed via `ProtocolRuntime` dispatch on inbound envelope. Decodes `(announcing_peer, announced_addresses)`; registers in entries map under `default_ttl_ns`; updates AddressBook; replies with Handshake envelope to `GLOBAL_REGISTRY_CLIENT_CREF`. |

<!-- /toolkit:ops -->

### `test.Kademlia.atomic`

<!-- toolkit:ops domain="test.Kademlia.atomic" -->

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| FindNode | | | | Async | `bb-ops/src/protocols/kademlia_hand/mod.rs:110` | `KademliaHand` test-component `AtomicOpDecl` in `PROTO_OPSET_OPS`. Hand-written `ProtocolRuntime` impl (no `register_protocol!` to avoid clash with `#[derive(bb::Concrete)]`). Inventory `ComponentRoleBinding` submitted directly at `kademlia_hand/mod.rs:135`. Dispatch records call name in shared `Vec` for test assertions. |
| Sample | | | | Immediate | `bb-ops/src/protocols/kademlia_hand/mod.rs:117` | KademliaHand stub op; `ComponentRoleBinding` inventory submission at `kademlia_hand/mod.rs:135` routes via PeerSelector derive. |
| Search | | | | Async | `bb-ops/src/protocols/kademlia_hand/mod.rs:124` | KademliaHand stub op routed via Index derive; returns `Immediate(empty)` in `dispatch_atomic`. |

<!-- /toolkit:ops -->

### `ai.bytesandbrains.role.aggregator`

<!-- toolkit:ops domain="ai.bytesandbrains.role.aggregator" -->

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| Contribute | tensor: `Box<B::Tensor>`, metadata: FedAvgMeta | | | Immediate | `bb-ops/src/aggregators/fedavg/mod.rs:277` | `FedAvg<B>` `AtomicOpDecl` in `FEDAVG_ATOMIC_OPS`. Framework-routed via `#[derive(bb::Aggregator)]` bridge — invokes `AggregatorContract::contribute` with src peer from `ctx.current.inbound.src_peer`; opens completion handle; returns `Immediate` or `Async(cmd)` depending on `ContractResponse`. Inventory carriers at `fedavg/mod.rs:452, 464, 472, 483` register monomorphization `FedAvg<CpuBackend>` under `cpu-backend` feature: `ConcreteComponentRegistration`, `ComponentRoleBinding`, `DispatcherRegistration`, `StorageTypeEntry`. |
| Aggregate | | params: `Box<B::Tensor>`, metadata: FedAvgMeta | | Immediate | `bb-ops/src/aggregators/fedavg/mod.rs:284` | `FedAvg<B>` `AtomicOpDecl`; framework-routed via `#[derive(bb::Aggregator)]`. `dispatch_atomic` opens completion handle; returns `Immediate` with `(params, metadata)` or `Async(cmd)` per `ContractResponse`. |

<!-- /toolkit:ops -->

### `ai.onnx`

<!-- toolkit:ops domain="ai.onnx" -->

Backend tensor primitives. Prefer the `BackendSlot::*` placeholder methods — don't hand-record these unless you are authoring a new Backend.

| op_type | inputs | outputs | attributes | kind | code | notes |
|---|---|---|---|---|---|---|
| Add | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:51` | CpuBackend primitive (`PRIMITIVE_OPS` slice); `ai.onnx` v1 floor. BROADCAST_BINARY type relation. Routed via `BackendRuntime::atomic_opset`; engine dispatches via `invoke_backend_subgraph` carrier path, not OpRegistration. |
| Sub | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:52` | CpuBackend primitive; BROADCAST_BINARY relation. |
| Mul | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:53` | CpuBackend primitive; BROADCAST_BINARY relation. |
| Div | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:54` | CpuBackend primitive; BROADCAST_BINARY relation. |
| Neg | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:55` | CpuBackend primitive; ELEMENTWISE relation. |
| Abs | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:56` | CpuBackend primitive; ELEMENTWISE relation. |
| Sqrt | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:58` | CpuBackend primitive; ELEMENTWISE relation. |
| Pow | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:59` | CpuBackend primitive; BROADCAST_BINARY relation. |
| Exp | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:60` | CpuBackend primitive; ELEMENTWISE relation. |
| Log | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:61` | CpuBackend primitive; ELEMENTWISE relation. |
| MatMul | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:63` | CpuBackend primitive; MATMUL_BINARY relation. |
| ReduceSum | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:65` | CpuBackend primitive; REDUCE_AXIS relation. |
| ReduceMean | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:66` | CpuBackend primitive; REDUCE_AXIS relation. |
| ReduceMax | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:67` | CpuBackend primitive; REDUCE_AXIS relation. |
| ReduceMin | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:68` | CpuBackend primitive; REDUCE_AXIS relation. |
| Reshape | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:73` | CpuBackend primitive; UNARY_SAME_ELEMENT relation. |
| Transpose | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:74` | CpuBackend primitive; UNARY_SAME_ELEMENT relation. |
| Concat | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:75` | CpuBackend primitive; NO_RELATIONS (variadic). |
| Slice | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:76` | CpuBackend primitive; UNARY_SAME_ELEMENT relation. |
| Split | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:77` | CpuBackend primitive; NO_RELATIONS (variadic). |
| Squeeze | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:78` | CpuBackend primitive; UNARY_SAME_ELEMENT relation. |
| Unsqueeze | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:79` | CpuBackend primitive; UNARY_SAME_ELEMENT relation. |
| Identity | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:80` | CpuBackend primitive; ELEMENTWISE relation (true pass-through). |
| Cast | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:81` | CpuBackend primitive; NO_RELATIONS (attribute-driven). |
| Equal | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:85` | CpuBackend primitive; NO_RELATIONS (output bool, unconstrained until bool tensor leaf lands). |
| Greater | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:86` | CpuBackend primitive; NO_RELATIONS. |
| Less | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:87` | CpuBackend primitive; NO_RELATIONS. |
| Where | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:89` | CpuBackend primitive; NO_RELATIONS (conditional). |
| Constant | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:92` | CpuBackend primitive; NO_RELATIONS (TensorProto attribute-driven). Distinct from syscall Constant in `bb-ops/src/syscalls/structural/constant`. |
| Gather | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:94` | CpuBackend primitive; NO_RELATIONS (tensor + index types). |
| Relu | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:105` | CpuBackend `EXTENSION_OPS` (`ai.onnx` `EXTENSION_VERSION=1`); ELEMENTWISE relation. Not in primitive floor; users with different backend may not get it. |
| Sigmoid | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:106` | CpuBackend extension; ELEMENTWISE relation. |
| Tanh | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:107` | CpuBackend extension; ELEMENTWISE relation. |
| Softmax | | | | Immediate | (truncated in inventory) | CpuBackend extension; routed via `BackendRuntime::atomic_opset`. Prefer `BackendSlot::softmax`. |
| LeakyRelu | | | alpha: f32 | Immediate | `bb-ops/src/backends/cpu/opset.rs:109` | CpuBackend extension; ELEMENTWISE relation. Prefer `BackendSlot::leaky_relu`. |
| Gelu | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:110` | CpuBackend extension; ELEMENTWISE relation. Prefer `BackendSlot::gelu`. |
| Dot | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:116` | CpuBackend extension; MATMUL_BINARY relation. Prefer `BackendSlot::dot`. |
| Gemm | | | alpha, beta, transA, transB | Immediate | `bb-ops/src/backends/cpu/opset.rs:117` | CpuBackend extension; MATMUL_BINARY relation. Prefer `BackendSlot::gemm`. |
| Zeros | | | dims | Immediate | `bb-ops/src/backends/cpu/opset.rs:120` | CpuBackend extension; constant tensor of zeros. Prefer `BackendSlot::zeros`. |
| Ones | | | dims | Immediate | `bb-ops/src/backends/cpu/opset.rs:121` | CpuBackend extension; constant tensor of ones. Prefer `BackendSlot::ones`. |
| GlobalAveragePool | | | | Immediate | `bb-ops/src/backends/cpu/opset.rs:124` | CpuBackend extension; reduce-spatial-axes relation. Prefer `BackendSlot::global_average_pool`. |

<!-- /toolkit:ops -->

## Placeholder slots (the high-level DSL)

<!-- toolkit:slots -->

| Slot | Method | Inputs | Outputs | Code |
|---|---|---|---|---|
| BackendSlot | `zeros` | `g: &mut Graph, dims: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:79` |
| BackendSlot | `ones` | `g: &mut Graph, dims: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:84` |
| BackendSlot | `constant` | `g: &mut Graph, value: TensorProto` | `Output` | `bb-ops/src/placeholders/mod.rs:89` |
| BackendSlot | `add` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:96` |
| BackendSlot | `sub` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:101` |
| BackendSlot | `mul` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:106` |
| BackendSlot | `div` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:111` |
| BackendSlot | `neg` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:116` |
| BackendSlot | `abs` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:121` |
| BackendSlot | `sqrt` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:126` |
| BackendSlot | `exp` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:131` |
| BackendSlot | `log` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:136` |
| BackendSlot | `pow` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:141` |
| BackendSlot | `matmul` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:148` |
| BackendSlot | `gemm` | `g: &mut Graph, a: Output, b: Output, c: Option<Output>, alpha: f32, beta: f32, trans_a: bool, trans_b: bool` | `Output` | `bb-ops/src/placeholders/mod.rs:154` |
| BackendSlot | `dot` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:183` |
| BackendSlot | `relu` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:190` |
| BackendSlot | `sigmoid` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:195` |
| BackendSlot | `tanh` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:200` |
| BackendSlot | `softmax` | `g: &mut Graph, t: Output, axis: i64` | `Output` | `bb-ops/src/placeholders/mod.rs:205` |
| BackendSlot | `leaky_relu` | `g: &mut Graph, t: Output, alpha: f32` | `Output` | `bb-ops/src/placeholders/mod.rs:210` |
| BackendSlot | `gelu` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:220` |
| BackendSlot | `reshape` | `g: &mut Graph, t: Output, dims: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:227` |
| BackendSlot | `transpose` | `g: &mut Graph, t: Output, perm: Option<Vec<i64>>` | `Output` | `bb-ops/src/placeholders/mod.rs:232` |
| BackendSlot | `concat` | `g: &mut Graph, tensors: Vec<Output>, axis: i64` | `Output` | `bb-ops/src/placeholders/mod.rs:241` |
| BackendSlot | `split` | `g: &mut Graph, t: Output, axis: i64, sizes: Vec<i64>` | `Vec<Output>` | `bb-ops/src/placeholders/mod.rs:248` |
| BackendSlot | `slice` | `g: &mut Graph, t: Output, starts: Vec<i64>, ends: Vec<i64>, axes: Option<Vec<i64>>, steps: Option<Vec<i64>>` | `Output` | `bb-ops/src/placeholders/mod.rs:260` |
| BackendSlot | `squeeze` | `g: &mut Graph, t: Output, axes: Option<Vec<i64>>` | `Output` | `bb-ops/src/placeholders/mod.rs:280` |
| BackendSlot | `unsqueeze` | `g: &mut Graph, t: Output, axes: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:289` |
| BackendSlot | `identity` | `g: &mut Graph, t: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:294` |
| BackendSlot | `cast` | `g: &mut Graph, t: Output, to_elem_type: i32` | `Output` | `bb-ops/src/placeholders/mod.rs:299` |
| BackendSlot | `reduce_sum` | `g: &mut Graph, t: Output, axes: Option<Vec<i64>>, keepdims: bool` | `Output` | `bb-ops/src/placeholders/mod.rs:326` |
| BackendSlot | `reduce_mean` | `g: &mut Graph, t: Output, axes: Option<Vec<i64>>, keepdims: bool` | `Output` | `bb-ops/src/placeholders/mod.rs:337` |
| BackendSlot | `reduce_max` | `g: &mut Graph, t: Output, axes: Option<Vec<i64>>, keepdims: bool` | `Output` | `bb-ops/src/placeholders/mod.rs:348` |
| BackendSlot | `reduce_min` | `g: &mut Graph, t: Output, axes: Option<Vec<i64>>, keepdims: bool` | `Output` | `bb-ops/src/placeholders/mod.rs:359` |
| BackendSlot | `equal` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:372` |
| BackendSlot | `greater` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:377` |
| BackendSlot | `less` | `g: &mut Graph, a: Output, b: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:382` |
| BackendSlot | `batch_normalization` | `g: &mut Graph, input: Output, scale: Output, bias: Output, mean: Output, variance: Output, epsilon: f32, momentum: f32` | `Output` | `bb-ops/src/placeholders/mod.rs:390` |
| BackendSlot | `layer_normalization` | `g: &mut Graph, input: Output, scale: Output, bias: Output, axis: i64, epsilon: f32` | `Output` | `bb-ops/src/placeholders/mod.rs:413` |
| BackendSlot | `conv` | `g: &mut Graph, input: Output, weight: Output, bias: Option<Output>, strides: Vec<i64>, padding: Vec<i64>, dilations: Vec<i64>, groups: i64` | `Output` | `bb-ops/src/placeholders/mod.rs:438` |
| BackendSlot | `max_pool` | `g: &mut Graph, input: Output, kernel_shape: Vec<i64>, strides: Vec<i64>, padding: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:469` |
| BackendSlot | `average_pool` | `g: &mut Graph, input: Output, kernel_shape: Vec<i64>, strides: Vec<i64>, padding: Vec<i64>` | `Output` | `bb-ops/src/placeholders/mod.rs:490` |
| BackendSlot | `global_average_pool` | `g: &mut Graph, input: Output` | `Output` | `bb-ops/src/placeholders/mod.rs:513` |
| BackendSlot | `gather` | `g: &mut Graph, data: Output, indices: Output, axis: i64` | `Output` | `bb-ops/src/placeholders/mod.rs:520` |
| BackendSlot | `scatter` | `g: &mut Graph, data: Output, indices: Output, updates: Output, axis: i64` | `Output` | `bb-ops/src/placeholders/mod.rs:530` |
| BackendSlot | `if_op` | `g: &mut Graph, cond: Output, then_branch: GraphProto, else_branch: GraphProto` | `Output` | `bb-ops/src/placeholders/mod.rs:551` |
| BackendSlot | `loop_op` | `g: &mut Graph, max_iter: Output, cond: Output, body: GraphProto` | `Output` | `bb-ops/src/placeholders/mod.rs:573` |
| ModelSlot | `forward` | `g: &mut Graph, input: Output` | `Output` (prediction) | `bb-ops/src/placeholders/mod.rs:673` |
| ModelSlot | `backward` | `g: &mut Graph, grad: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:686` |
| ModelSlot | `compute_loss` | `g: &mut Graph, prediction: Output, target: Output` | `Output` (loss scalar) | `bb-ops/src/placeholders/mod.rs:699` |
| ModelSlot | `apply_delta` | `g: &mut Graph, delta: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:712` |
| ModelSlot | `load_parameters` | `g: &mut Graph, params: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:725` |
| ModelSlot | `params` | `g: &mut Graph` | `Output` (params tensor) | `bb-ops/src/placeholders/mod.rs:738` |
| IndexSlot | `add` | `g: &mut Graph, vectors: Output, ids: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:764` |
| IndexSlot | `search` | `g: &mut Graph, query: Output, k: i64` | `(Output, Output)` (ids, scores) | `bb-ops/src/placeholders/mod.rs:778` |
| IndexSlot | `remove` | `g: &mut Graph, ids: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:791` |
| IndexSlot | `train` | `g: &mut Graph, vectors: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:808` |
| AggregatorSlot | `contribute` | `g: &mut Graph, tensor: Output, metadata: Output` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:852` |
| AggregatorSlot | `aggregate` | `g: &mut Graph` | `(Output, Output)` (params, metadata) | `bb-ops/src/placeholders/mod.rs:871` |
| CodecSlot | `train` | `g: &mut Graph, ...` | `Output` (updated codec) | `bb-ops/src/placeholders/mod.rs:906` |
| CodecSlot | `encode` | `g: &mut Graph, value: Output` | `Output` (encoded bytes) | `bb-ops/src/placeholders/mod.rs:929` |
| CodecSlot | `decode` | `g: &mut Graph, bytes: Output` | `Output` (decoded value) | `bb-ops/src/placeholders/mod.rs:950` |
| DataLoaderSlot | `next_batch` | `g: &mut Graph` | `(Output, Output)` (batch, labels) | `bb-ops/src/placeholders/mod.rs:986` |
| DataLoaderSlot | `reset` | `g: &mut Graph` | `Output` (ack trigger) | `bb-ops/src/placeholders/mod.rs:1003` |
| DataLoaderSlot | `on_data_loaded` | `g: &mut Graph` | `Output` (event trigger) | `bb-ops/src/placeholders/mod.rs:1017` |
| PeerSelectorSlot | `sample` | `g: &mut Graph, n: usize` | `Output` (peer ids vec) | `bb-ops/src/placeholders/mod.rs:1092` |
| PeerSelectorSlot | `current_view` | `g: &mut Graph` | `Output` (peer ids vec) | `bb-ops/src/placeholders/mod.rs:1097` |

<!-- /toolkit:slots -->

## DSL helpers (`bb-dsl/src/syscalls.rs`)

<!-- toolkit:dsl-helpers -->

| Helper | Signature | Records | Code |
|---|---|---|---|
| `pass_through` | `pub fn pass_through(g: &mut Graph, input: Output) -> Output` | `PassThrough` syscall NodeProto (`domain=SYSCALL_DOMAIN`, `op_type=OP_PASS_THROUGH`). Structural identity op — threads a value through a partition without compute; output type matches input's `type_node`. | `bb-dsl/src/syscalls.rs:57` |
| `address_book_insert_many` | `pub fn address_book_insert_many(g: &mut Graph, peer: Output, addresses: Output) -> Output` | `AddressBook::InsertMany` custom-op NodeProto (`domain=ai.bytesandbrains.address_book`, `op_type=InsertMany`). Inputs: peer + addresses; output typed as `TYPE_TRIGGER`. New peer creates `ref_count=1` entry; known peer dedupe-appends; empty addresses vec is a dispatch-time `OpError`. | `bb-dsl/src/syscalls.rs:75` |
| `address_book_lookup` | `pub fn address_book_lookup(g: &mut Graph, peer: Output) -> Output` | `AddressBook::Lookup` custom-op NodeProto (`domain=ai.bytesandbrains.address_book`, `op_type=Lookup`). Input: peer; output typed as `TYPE_ADDRESS_VEC` (full ordered vec). Unknown / empty-address peer surfaces as dispatch-time `OpError`. | `bb-dsl/src/syscalls.rs:92` |
| `gate_dispatch` | `pub fn gate_dispatch(g: &mut Graph, inputs: &[Output]) -> Output` | `GateDispatch` syscall NodeProto (`domain=SYSCALL_DOMAIN`, `op_type=OP_GATE_DISPATCH`). Multi-edge synchronization barrier; takes a slice of inputs, output typed as `TYPE_BYTES`. | `bb-dsl/src/syscalls.rs:107` |
| `constant` | `pub fn constant(g: &mut Graph, label: &'static str, output_type: &'static TypeNode, data_type: tensor_proto::DataType) -> Output` | `Constant` syscall NodeProto (`domain=SYSCALL_DOMAIN`, `op_type=OP_CONSTANT`). Empty `TensorProto` sized for `data_type` satisfies the compiler's `expand_constant` pass; `output_type` is the recorded `&'static TypeNode` consumers downcast on (`TYPE_PEER_ID`, `TYPE_ADDRESS_VEC`, …). `label` rides on `metadata_props` under `ai.bytesandbrains.bootstrap.seed` for diagnostics. | `bb-dsl/src/syscalls.rs:131` |
| `announce` | `pub fn announce(g: &mut Graph, server_peer: Output) -> Output` | `GlobalRegistryClient::Announce` NodeProto (`domain=ai.bytesandbrains.protocol.global_registry`, `op_type=Announce`). Input: server `PeerId`; output typed as `TYPE_TRIGGER`. Stamps `ai.bytesandbrains.input.server_peer` on `metadata_props` mapping the input to its logical name. Client reads `ctx.local_addresses()` automatically, throttles to the server's heartbeat interval, and merges the Handshake reply's address bag. | `bb-dsl/src/syscalls.rs:166` |

<!-- /toolkit:dsl-helpers -->

## When NOT to record a NodeProto by hand

- **Don't record `ai.bytesandbrains.wire::Recv`.** The compiler synthesizes it on receiver partitions (`bb_compiler::synthesize_wire_recvs`). Inbound delivery seeds the slot directly via `deliver_fill` — your hand-rolled Recv will never see a frontier.
- **Don't `Constant(local_addresses)` before Announce.** `ctx.local_addresses()` is auto-populated from install; the client snapshots them into the payload inside `dispatch_atomic`.
- **Don't record a periodic Heartbeat op.** `GlobalRegistryClient::Announce` self-throttles to `last_heartbeat_interval_ns`. Use an `Interval` or a tick source and let throttling no-op the extra calls.
- **Don't record the gate ops (`DedupGateRx`, `PeerHealthGateTx`, `BackoffGateTx`, `PeerHealthGateRx`, `BackoffGateRx`, `DeadlineCheck`).** Compiler passes (`insert_dedup_gate_rx`, `insert_peer_health_gate_tx`, `insert_backoff_gate_tx`, `insert_peer_health_gate_rx`, `insert_backoff_gate_rx`, `insert_async_deadlines`) stamp them around your wire ops automatically.
- **Don't carry inbound `src_peer` as a NodeProto attribute.** It's framework-supplied as `ctx.current.inbound.src_peer` per envelope. The RX gates and aggregator `Contribute` already read it from there.
- **Don't hand-record `ai.onnx::Add` (or any backend tensor primitive).** Call the matching `BackendSlot::*` method so the bound backend's monomorphization lands at install time. Hand-recording locks you to one backend and bypasses the placeholder rewrite.
- **Don't wrap multi-input merges in custom barriers.** Engine all-inputs-ready already gates dispatch — `Gate` and `GateDispatch` are pure forwarders. The right tool for "wait for N" is `GateDispatch`; the right tool for "first of N" is `Any`.
- **Don't add a dedup wrapper around `Any` or `DeadlineMatch`.** Both latch per-group / per-(OpRef, ExecId) inside `ctx.syscall.any_fired_groups` / `ctx.syscall.deadline_match_fired`.
- **Don't poll on async timers.** `After`, `Sleep`, `BootstrapOutput` return `Async(cmd)` and the engine routes the completion via `handle_completion`. Park the op, don't loop.
- **Don't use a reserved telemetry name.** `AppEmit` / `AppNotify` validate against `bb.` and `ai.bytesandbrains.` prefixes — collision returns `BadInput`. Pick your own namespace.
- **Don't author a new aggregator if you just need weighted mean.** `FedAvg<B>` composes its reduction from the bound backend's `Mul` + `Add` primitives, keeps deterministic lexical-peer-id order, replaces duplicate same-round contributions, and emits cumulative `num_samples` for hierarchical use.
- **Don't pre-dedup contributions before `FedAvg::Contribute`.** Same-round duplicates from the same peer REPLACE the prior entry inside the per-round buffer. NaN/Inf poisoning belongs at the contribution boundary, not in a wrapper op.
