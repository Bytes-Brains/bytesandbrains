//! Syscall identifier string constants — the IR-level contract
//! between compiler (gate emission) and runtime (dispatch).

/// Framework reverse-DNS root.
pub const FRAMEWORK_DOMAIN: &str = "ai.bytesandbrains";

/// Domain for every framework-emitted syscall (compiler gates,
/// `pass_through`, `gate_dispatch`).
pub const SYSCALL_DOMAIN: &str = "ai.bytesandbrains.syscall";

/// Framework syscall opset version. Bumps on dispatch-shape changes
/// downstream nodes cannot satisfy.
pub const SYSCALL_OPSET_VERSION: i64 = 1;

/// Domain for the wire opset (Send + Recv).
pub const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";

/// Wire opset version.
pub const WIRE_OPSET_VERSION: i64 = 1;

/// Domain for `service` / `module` lowering ops.
pub const SERVICE_DOMAIN: &str = "ai.bytesandbrains.service";

/// Service opset version.
pub const SERVICE_OPSET_VERSION: i64 = 1;

/// Domain for peer-sampling + gossip-substrate ops. Distinct from
/// [`WIRE_DOMAIN`].
pub const NETWORK_DOMAIN: &str = "ai.bytesandbrains.network";

/// Network opset version.
pub const NETWORK_OPSET_VERSION: i64 = 1;

// --- DSL-side syscall op_types ------------------------------------

/// Structural identity — threads a value through a partition.
pub const OP_PASS_THROUGH: &str = "PassThrough";

/// Wire send.
pub const OP_WIRE_SEND: &str = "Send";

/// Wire recv. Paired with `Send` by `WIRE_ID_KEY`.
pub const OP_WIRE_RECV: &str = "Recv";

/// Multi-edge synchronization barrier.
pub const OP_GATE_DISPATCH: &str = "GateDispatch";

/// Fan a single input to N outputs via `SlotValue::clone_boxed`.
pub const OP_TEE: &str = "Tee";

/// Emit a literal from a `TensorProto` / `BytesProto` attribute.
pub const OP_CONSTANT: &str = "Constant";

/// `i64` deadline in nanoseconds (reference-clock epoch).
pub const ATTR_DEADLINE_NS: &str = "deadline_ns";

/// `PeerId`-typed value name. Used by backoff + peer-health gates.
pub const ATTR_PEER: &str = "peer";

// --- Coordination -------------------------------------------------

/// Deadline gate on the protected op's first input.
pub const OP_DEADLINE_CHECK: &str = "DeadlineCheck";

// --- Gates --------------------------------------------------------

/// Receive-side backoff gate upstream of high-volume Recvs.
pub const OP_BACKOFF_GATE_RX: &str = "BackoffGateRx";

/// Send-side backoff gate.
pub const OP_BACKOFF_GATE_TX: &str = "BackoffGateTx";

/// Drop duplicate-arrival Recv envelopes.
pub const OP_DEDUP_GATE_RX: &str = "DedupGateRx";

/// Fast-fail inbound envelopes from unhealthy peers (φ-accrual).
pub const OP_PEER_HEALTH_GATE_RX: &str = "PeerHealthGateRx";

/// Fast-fail outbound envelopes to unhealthy peers (φ-accrual).
pub const OP_PEER_HEALTH_GATE_TX: &str = "PeerHealthGateTx";

/// Default per-hop budget (100 ms) for sizing async wire-Send
/// deadlines. `derive_wire_deadlines` multiplies static
/// `chain_depth` by this; `NodeConfig` can override per-Node.
pub const DEFAULT_PER_HOP_BUDGET_NS: u64 = 100_000_000;
