//! `ai.bytesandbrains.wire v1` opset - engine-native.
//!
//! Two ops, exactly one of which the user authors:
//!
//! - `Send` (user-facing): takes a payload + a typed `peer` input
//!   carrying a [`PeerId`](bb_runtime::ids::PeerId); resolves the peer to
//!   a multiaddr through `ctx.peers.addresses`, builds one `SlotFill`
//!   per non-peer input from the compiler-stamped `dest_suffix`
//!   metadata, ships the envelope, and returns `(data, handle)` -
//!   a structural-placeholder `Trigger` plus a freshly minted
//!   `wire_req_id`.
//!
//! - `Recv` (framework-synthesized): emitted on receiver partitions
//!   by [bb_compiler::synthesize_wire_recvs()]. Pure structural
//!   placeholder so the receiver's `FunctionProto` stays a closed
//!   DAG and the consumer's input value name resolves to a
//!   `NodeSiteId` the inbound `deliver_fill` writes into. The
//!   dispatch handler returns no outputs - Recv never lands on the
//!   frontier in normal flow.
//!
//! Both ops register through the same path as every other syscall
//! (`Engine::register_syscall`). There is no separate `WireRuntime`
//! binding - wire is engine infrastructure, not a user role.

use std::collections::HashMap;

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::envelope::{SlotFill, WireCorrelation, WireEnvelope};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{BytesValue, PeerIdVecValue, WireReqIdValue};

/// Wire opset domain.
pub const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";

/// Wire opset version.
pub const WIRE_VERSION: i64 = 1;

/// Marker struct for `register_syscall::<SendOp>`.
pub struct SendOp;

/// Marker struct for `register_syscall::<RecvOp>`.
pub struct RecvOp;

/// `Send` dispatch entry point matching `StatelessInvokeFn`.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    invoke_send(inputs, ctx)
}

/// Encode a single slot value to its wire bytes. Surfaces encode
/// failures as `OpError::ExecutionFailed` carrying the slot name
/// and the underlying error, so a malformed payload drops the op
/// instead of crashing the node.
fn encode_or_error(name: &str, value: &dyn SlotValue) -> Result<Vec<u8>, OpError> {
    value.to_wire_bytes().map_err(|e| OpError {
        kind: bb_runtime::bus::OpErrorKind::ExecutionFailed,
        reason: "wire_encode_failed",
        detail: format!("wire encode of slot `{name}` failed: {e}"),
    })
}

/// `Recv` dispatch entry point matching `StatelessInvokeFn`. Pure
/// structural placeholder; downstream consumers are pushed onto the
/// frontier when inbound data-plane delivery seeds the slot, not by
/// this dispatch.
pub fn invoke_recv(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    Ok(DispatchResult::Immediate(vec![]))
}

/// Construct an envelope with the resolved address list, one or
/// more fills and the given correlation. `dest_peer_addresses` is
/// the resolved snapshot of `AddressBook::lookup(peer)` at dispatch
/// time per `docs/ADDRESSING.md`. The host's transport adapter
/// picks one of these entries based on its networking capabilities.
/// Each fill carries its own per-slot multiaddr `dest_suffix` for
/// *intra-node* slot routing only.
fn build_envelope(
    dest_peer_addresses: Vec<Vec<u8>>,
    fills: Vec<SlotFill>,
    correlation: WireCorrelation,
) -> WireEnvelope {
    WireEnvelope {
        dest_peer_addresses,
        fills,
        correlation: Some(correlation),
        remaining_deadline_ns: 0,
        edge_rtt_reports: Vec::new(),
        ..Default::default()
    }
}

pub(crate) fn invoke_send(
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let dest_peers = extract_dest_peers(inputs, ctx);
    let fills = collect_fills(inputs, ctx)?;

    // Decide ORIGINATOR vs FORWARDER role:
    // - Forwarder: this Send fires inside an inbound-delivery cascade
    //   AND the inbound envelope carried a non-zero correlation
    //   wire_req_id. Reuse that token; skip register_in_flight.
    // - Originator: read `chain_id` from the Send NodeProto's
    //   metadata (stamped by the compiler's `analyze_round_trips`
    //   pass) and use it as the wire_req_id; if absent, mint fresh.
    //   Register the in-flight entry so the eventual response back
    //   to this site is observable.
    let (req_id_u64, is_forwarder) = match ctx.current.inbound.wire_req_id {
        Some(inbound) if inbound != 0 => (inbound, true),
        _ => {
            // Originator path.
            let chain_id =
                read_metadata_u64(ctx.current.node_metadata, "ai.bytesandbrains.wire.chain_id");
            let token = chain_id.unwrap_or_else(|| ctx.net.requests.mint_token().as_u64());
            (token, false)
        }
    };

    // Derive Dapper-style outbound `remaining_deadline_ns`. When
    // forwarding inside a cascade, subtract elapsed local time from
    // the inbound's remaining budget. Otherwise read the static
    // `deadline_ns` attribute stamped by `derive_wire_deadlines`.
    let chain_ctx = ctx.read_chain_context();
    let mut outbound_deadline_ns: u64 = 0;
    if let (Some(inbound_remaining), Some(arrival_ns)) = (
        ctx.current.inbound.remaining_deadline_ns,
        ctx.current.inbound.arrival_ns,
    ) {
        let now_ns = ctx.time.scheduler.now_ns();
        let elapsed = now_ns.saturating_sub(arrival_ns);
        outbound_deadline_ns = inbound_remaining.saturating_sub(elapsed);
    } else if let Some(static_deadline_ns) =
        read_attribute_u64(ctx.current.node_attributes, "deadline_ns")
    {
        outbound_deadline_ns = static_deadline_ns;
    } else if let Some(first_peer) = dest_peers.first().copied() {
        // Last resort: adaptive RTT-tracker estimate, indexed by the
        // first peer's site (representative for fan-out).
        outbound_deadline_ns = ctx.estimate_wire_budget_ns(
            peer_to_site(first_peer),
            chain_ctx,
            bb_ir::syscall_ids::DEFAULT_PER_HOP_BUDGET_NS,
        );
    }

    if let Some(first_peer) = dest_peers.first().copied() {
        if !is_forwarder {
            // Originator registers ONE in-flight entry per Send call
            // (not per fan-out peer) — the `wire_req_id` is shared
            // across all envelopes in the fan-out, and the first
            // response satisfies the request. Forwarders skip;
            // the upstream originator's entry tracks the chain.
            //
            // `ttl_ns` is `NonZeroU64`; if the deadline
            // budget collapsed to 0 (a degenerate but possible
            // outcome of the deadline cascade above), substitute the
            // minimum 1 ns so the in-flight entry still expires
            // rather than living forever.
            let target_site = peer_to_site(first_peer);
            let ttl = std::num::NonZeroU64::new(outbound_deadline_ns)
                .unwrap_or(unsafe { std::num::NonZeroU64::new_unchecked(1) });
            ctx.net.requests.register_in_flight(
                req_id_u64,
                ctx.time.scheduler.now_ns(),
                target_site,
                chain_ctx,
                ttl,
                None,
            );
        }
    }

    // Empty peer list → no envelopes (composes naturally with
    // samplers that returned no peers this round).
    let kind = if req_id_u64 != 0 {
        bb_runtime::envelope::CorrelationKind::Request as i32
    } else {
        bb_runtime::envelope::CorrelationKind::None as i32
    };
    // Snapshot the sender's local-address list once per Send call.
    // The receiver merges this into its AddressBook entry for the
    // sender so future replies can dial back on any reachable
    // interface; an empty Vec stamps zero entries and the receiver
    // leaves its existing entry untouched.
    let src_peer_addresses: Vec<Vec<u8>> =
        ctx.local_addresses().iter().map(|a| a.to_bytes()).collect();

    for peer in &dest_peers {
        // Resolve the peer's address list. Per-peer failure surfaces
        // as PeerResolveFailed but does NOT abort the fan-out — other
        // destinations may still receive.
        let resolved: Option<Vec<Vec<u8>>> = ctx
            .peers
            .addresses
            .lookup(*peer)
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().map(|a| a.to_bytes()).collect());
        match resolved {
            Some(dest_peer_addresses) => {
                let mut env = build_envelope(
                    dest_peer_addresses,
                    fills.clone(),
                    WireCorrelation {
                        kind,
                        wire_req_id: req_id_u64,
                    },
                );
                env.remaining_deadline_ns = outbound_deadline_ns;
                env.src_peer_addresses = src_peer_addresses.clone();
                ctx.net.outbound.push(env);
            }
            None => {
                ctx.net
                    .pending_peer_resolve_failures
                    .push((Some(*peer), ctx.current.op_ref));
                ctx.bus.publish(bb_runtime::bus::NodeEvent::Infra(
                    bb_runtime::bus::InfraEvent::PeerResolveFailure {
                        peer: Some(*peer),
                        op_ref: ctx.current.op_ref,
                    },
                ));
            }
        }
    }
    // Two outputs by position: (data, handle).
    // - `data` is a structural placeholder on the producer side; the
    //   downstream Recv produces the actual receiver-side payload.
    //   A TriggerValue keeps the slot occupied so any accidental
    //   local read sees a non-empty signal.
    // - `handle` carries the wire_req_id so downstream local ops can
    //   thread completion / timeout tracking even when the envelope
    //   never shipped (empty peer list, all unresolvable, etc.).
    Ok(DispatchResult::Immediate(vec![
        (
            "data".to_string(),
            Box::new(bb_runtime::syscall::values::TriggerValue) as Box<dyn SlotValue>,
        ),
        (
            "handle".to_string(),
            Box::new(WireReqIdValue(req_id_u64)) as Box<dyn SlotValue>,
        ),
    ]))
}

/// Look up a metadata_props entry by key, parsing the value as u64.
fn read_metadata_u64(
    props: &[bb_ir::proto::onnx::StringStringEntryProto],
    key: &str,
) -> Option<u64> {
    props
        .iter()
        .find(|p| p.key == key)
        .and_then(|p| p.value.parse().ok())
}

/// Look up an attribute by name, returning its i64 field as u64.
fn read_attribute_u64(attrs: &[bb_ir::proto::onnx::AttributeProto], name: &str) -> Option<u64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i.max(0) as u64)
}

/// Map a `PeerId` onto a `NodeSiteId` for RTT tracker indexing.
/// Hashes the full multihash digest with FNV-1a so two peers whose
/// multihashes share a leading 8-byte prefix don't collide into the
/// same RTT slot. Production deployments that distinguish multiple
/// logical sites per physical peer (e.g. fast ping handler vs GPU
/// compute handler on the same address) can swap this for an
/// explicit `NodeSiteId` carried in the envelope.
fn peer_to_site(peer: bb_runtime::ids::PeerId) -> bb_runtime::ids::NodeSiteId {
    bb_runtime::ids::NodeSiteId::from(bb_runtime::slot_value::fnv1a_64(peer.digest()))
}

/// Pull the destination `PeerId` from the `dest`/`peer` input.
/// The wire syscall then resolves the PeerId to its multi-address
/// list via the framework's `AddressBook` and packs the list into
/// `WireEnvelope.dest_peer_addresses`. On lookup miss (peer unknown
/// or its address list is empty), the wire syscall emits
/// `EngineStep::PeerResolveFailed` + `InfraEvent::PeerResolveFailure`
/// per `docs/ADDRESSING.md` instead of shipping an envelope.
///
/// Pull the destination peer list from a Send op's inputs.
///
/// Vec-only: the position-1 input must resolve to a
/// `PeerIdVecValue`. Samplers + broadcast components always emit
/// a vector; a single-peer send wraps `[peer]` itself. Returns
/// an empty `Vec` when no peer input resolves — composes
/// naturally with samplers that returned no peers this round (the
/// wire fires no envelopes; the local DAG continues).
pub(crate) fn extract_dest_peers(
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &RuntimeResourceRef<'_>,
) -> Vec<bb_runtime::ids::PeerId> {
    // Closure: decode one input value into a peer vector. The
    // `BytesValue` fallback handles `IngressEvent::Invoke` /
    // `AppEvent` hosts that ship pre-encoded peer-vec bytes — the
    // engine wraps those into a `BytesValue` carrier without per-
    // slot `type_hash` plumbing on those ingress paths.
    let try_decode = |h: &dyn SlotValue| -> Option<Vec<bb_runtime::ids::PeerId>> {
        if let Some(p) = h.as_any().downcast_ref::<PeerIdVecValue>() {
            return Some(p.0.clone());
        }
        if let Some(b) = h.as_any().downcast_ref::<BytesValue>() {
            if let Ok(p) = bincode::deserialize::<PeerIdVecValue>(&b.0) {
                return Some(p.0);
            }
        }
        None
    };

    // 1. Named match — back-compat for hand-authored Sends.
    for (name, h) in inputs {
        if matches!(*name, "dest" | "dest_peer" | "peer" | "peers" | "peer_id") {
            if let Some(v) = try_decode(*h) {
                return v;
            }
        }
    }
    // 2. Position-based fallback — the DSL `g.net_out(name, peers, data)`
    //    emits `Send(input[0]=data, input[1]=peers)` regardless of
    //    the peer source's value name (e.g. `trainer_peers`,
    //    `agg_peers`, or a sampler's auto-minted output name).
    if let Some((_, h)) = inputs.get(1) {
        if let Some(v) = try_decode(*h) {
            return v;
        }
    }
    Vec::new()
}

/// Attribute prefix the compiler's `analyze_wire_edges` pass stamps
/// on each producer Send NodeProto. The value is the canonical
/// `Address` byte encoding of the destination multiaddr suffix
/// (e.g. `/site/<NodeSiteId>` or `/component/<cref>/op/<name>`).
const DEST_SUFFIX_ATTR_PREFIX: &str = "ai.bytesandbrains.dest_suffix.";

/// Attribute prefix the compiler's `analyze_wire_edges` pass stamps
/// when it has classified a per-input edge as trigger-only (the
/// downstream Recv consumer reads only the firing signal, not the
/// payload bytes). Takes priority over the `payload.is_empty()`
/// heuristic — a compiler-attested classification is authoritative
/// even when the value happens to encode to zero bytes.
const TRIGGER_ONLY_ATTR_PREFIX: &str = "ai.bytesandbrains.trigger_only.";

/// Collect every value input (anything not `dest*` or `req_id`) into
/// a `SlotFill`. Each fill's `dest_suffix` is resolved in priority
/// order:
///   1. Per-input attribute on the Send NodeProto (stamped by the
///      compiler's `analyze_wire_edges` pass).
///   2. Companion `<name>_suffix` input (escape hatch for hosts that
///      construct envelopes outside the compilation pipeline - e.g.
///      sim harnesses).
///   3. Empty suffix (envelope receiver drops the fill silently).
///
/// `SlotFill.type_hash` is stamped from the producer-side
/// `SlotValue::type_hash()` so the receiver dispatches by type
/// instead of seeing the always-0 default.
/// `SlotFill.trigger_only` resolves first to the compiler-stamped
/// per-input attribute (`ai.bytesandbrains.trigger_only.<name>`),
/// then falls back to the `payload.is_empty()` heuristic.
fn collect_fills(
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &RuntimeResourceRef<'_>,
) -> Result<Vec<SlotFill>, OpError> {
    // Index attribute-stamped suffixes by base input name.
    let mut attr_suffixes: HashMap<String, Vec<u8>> = HashMap::new();
    // Index compiler-stamped trigger-only classifications by base
    // input name. Attribute value is a non-empty byte string for
    // "trigger_only" / empty (or absent) for "data" / payload.
    let mut attr_trigger_only: HashMap<String, bool> = HashMap::new();
    for attr in ctx.current.node_attributes {
        if let Some(base) = attr.name.strip_prefix(DEST_SUFFIX_ATTR_PREFIX) {
            attr_suffixes.insert(base.to_string(), attr.s.clone());
        }
        if let Some(base) = attr.name.strip_prefix(TRIGGER_ONLY_ATTR_PREFIX) {
            attr_trigger_only.insert(base.to_string(), !attr.s.is_empty() || attr.i != 0);
        }
    }

    // Index companion-input suffixes for the fallback path. The
    // companion input carries multiaddr bytes (typically an
    // `AddressValue` or `BytesValue`); `to_wire_bytes` returns the
    // raw encoded suffix. Encode failures surface as
    // `OpError::ExecutionFailed` with the slot name in `detail` so
    // the engine can drop the op gracefully.
    let mut input_suffixes: HashMap<&str, Vec<u8>> = HashMap::new();
    for (name, h) in inputs {
        if let Some(base) = name.strip_suffix("_suffix") {
            input_suffixes.insert(base, encode_or_error(name, *h)?);
        }
    }

    let mut fills: Vec<SlotFill> = Vec::new();
    for (name, h) in inputs {
        if name.ends_with("_suffix")
            || matches!(
                *name,
                "dest" | "dest_peer" | "peer" | "peers" | "peer_id" | "req_id"
            )
        {
            continue;
        }
        let payload = encode_or_error(name, *h)?;
        let dest_suffix = attr_suffixes
            .get(*name)
            .cloned()
            .or_else(|| input_suffixes.get(*name).cloned())
            .unwrap_or_default();
        let trigger_only = attr_trigger_only
            .get(*name)
            .copied()
            .unwrap_or(payload.is_empty());
        let type_hash = h.type_hash();
        fills.push(SlotFill {
            dest_suffix,
            payload,
            trigger_only,
            type_hash,
        });
    }
    Ok(fills)
}


use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: WIRE_DOMAIN,
        op_type: "Send",
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: WIRE_DOMAIN,
        op_type: "Recv",
        invoke: invoke_recv,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
