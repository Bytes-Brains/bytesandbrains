//! `synthesize_wire_recvs` - synthesize a `wire.Recv` NodeProto on
//! each receiver partition for every `wire.Send`'s cross-partition
//! data consumer.
//!
//! The user-facing wire surface is one op: `wire.send(payload, peer)`
//! returning `(data, handle)`. Semantically the `data` output is on
//! the **receiver** side: downstream ops in the recorded function
//! that consume it end up on the receiver's partition after
//! [`partition_by_wire_ops`](super::partition_by_wire_ops::partition_by_wire_ops) slices.
//! For the engine to actually surface the value on the receiver, the
//! receiver's partition needs a NodeProto whose output name resolves
//! to a `NodeSiteId` that inbound `deliver_fill` writes into.
//!
//! This pass walks the recorded function before partitioning and, for
//! each `Send` whose `data` output is consumed by a NodeProto from a
//! different `module_instance` scope chain, inserts a synthesized
//! `Recv` NodeProto on the consumer's side and rewrites the consumer
//! inputs to point at the new value name.
//!
//! Pre-condition: every NodeProto carries
//! `ai.bytesandbrains.module_instance` metadata stamped by
//! [`Graph::with_module`](bb_dsl::graph::Graph::with_module). Without
//! a scope, all nodes share the default chain and no Recv is
//! synthesized - single-partition modules pass through unchanged.

use std::collections::BTreeMap;

use crate::error::CompileError;
use crate::partition_by_wire_ops::WIRE_DOMAIN;
use bb_ir::peer_class::{home_class_of_node, HOME_CLASS_KEY, SELF_CLASS};
use bb_ir::proto::onnx::{
    GraphProto, NodeProto, StringStringEntryProto, TypeProto, ValueInfoProto,
};

/// `op_type` of the user-facing send. Mirrors
/// [`SEND_OP_TYPES`](super::partition_by_wire_ops::SEND_OP_TYPES).
const SEND_OP: &str = "Send";

/// `op_type` of the framework-synthesized recv emitted by this pass.
const RECV_OP: &str = "Recv";

/// Metadata key the synthesized Recv carries pointing back at the
/// originating Send's index in the recorded function. Used by
/// [`super::analyze_wire_edges::analyze_wire_edges`] to stamp the matching `dest_suffix`
/// on the producer Send.
pub const SYNTHESIZED_FROM_KEY: &str = "ai.bytesandbrains.synthesized_from_send";

/// Walk `function.node` in place. For each `wire.Send`, group its
/// `data`-output consumers by their `HOME_CLASS_KEY` stamp and
/// synthesize one Recv per consumer-side class. Returns the number
/// of synthesized Recv NodeProtos.
///
/// Self-send semantics: when the consumer's class matches the send's
/// class (e.g. gossip's `gossip_peer` → `gossip_peer`), the
/// synthesized Recv lands in the SAME partition as the Send. At
/// runtime, the Send dispatches envelopes to other physical instances
/// of the class; the Recv receives envelopes from other instances.
/// Both paths share the one installed graph.
///
/// Pure per ANALYSIS.md §3.2 - no IO, no global state.
pub fn synthesize_wire_recvs(graph: &mut GraphProto) -> Result<usize, CompileError> {
    // two-phase refactor.
    //
    // Phase 1 (collect): walk every `wire.Send`, identify its
    // cross-partition consumers, and collect ALL planned rewrites
    // into a SINGLE accumulating map keyed by `(consumer_idx,
    // matched_input_name)`. Each map entry stores the
    // `Vec<RewriteOp>` of every Send that wants to rewrite that
    // particular consumer-input slot — N-fanin (one consumer fed
    // by N Sends) keeps ALL N rewrites instead of letting later
    // Sends collide with earlier ones via per-Send rewrite maps.
    // Closes `chief:B8`.
    //
    // Phase 2 (materialize): replay every rewrite against a fresh
    // clone of the original `graph.node` so the final node list
    // carries every consumer's full rewritten input list.

    let snapshot: Vec<NodeProto> = graph.node.clone();
    let mut sentinels: BTreeMap<usize, String> = BTreeMap::new();
    // Phase 1 accumulates Recvs directly keyed by send_idx so Phase 2
    // can interleave them without re-parsing the SYNTHESIZED_FROM_KEY
    // metadata value (which now stores a value NAME, not a node index).
    let mut recvs_by_send: BTreeMap<usize, Vec<NodeProto>> = BTreeMap::new();
    let mut rewrites: BTreeMap<(usize, String), String> = BTreeMap::new();
    let mut next_seqno: usize = 0;
    let mut synthesized = 0usize;
    let mut new_value_info: Vec<ValueInfoProto> = Vec::new();
    let lookup_denotation = |name: &str| -> Option<String> {
        graph
            .value_info
            .iter()
            .find(|v| v.name == name)
            .and_then(|v| v.r#type.as_ref())
            .map(|t| t.denotation.clone())
            .filter(|d| !d.is_empty())
    };

    for (send_idx, original) in snapshot.iter().enumerate() {
        if original.domain != WIRE_DOMAIN || original.op_type != SEND_OP {
            continue;
        }
        let data_name = match original.output.first() {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };

        // Group cross-partition consumers by home class.
        let mut by_class: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (other_idx, other) in snapshot.iter().enumerate() {
            if other_idx == send_idx {
                continue;
            }
            if other.input.iter().any(|n| n == &data_name) {
                let class = home_class_of_node(other)
                    .map(str::to_string)
                    .unwrap_or_else(|| SELF_CLASS.to_string());
                by_class.entry(class).or_default().push(other_idx);
            }
        }

        if by_class.is_empty() {
            continue;
        }

        // Rename the Send's data output to a sentinel so the same
        // name can be reused on the synthesized Recv side.
        let sentinel_name = format!("{data_name}__send_sentinel_{send_idx}");
        sentinels.insert(send_idx, sentinel_name.clone());

        // Inherit the original data_name's denotation on the
        // sentinel + every minted Recv payload so the strict
        // type-solver finds a declared type at the rewritten sites.
        let payload_denotation = lookup_denotation(&data_name);
        if let Some(denot) = payload_denotation.as_deref() {
            new_value_info.push(stamped_value_info(&sentinel_name, denot));
        }

        for (class, consumer_indices) in by_class {
            let minted = format!("{data_name}__recv_{}", next_seqno);
            // synthesized Recv emits the canonical
            // `[payload, sender]` output shape so
            // `installed_graph::recv_sender_sites` pairs the two
            // sites for inbound envelope delivery (the engine
            // already expects this layout — closes the spec-vs-
            // code gap).
            let minted_sender = format!("{data_name}__recv_{}__sender", next_seqno);
            next_seqno += 1;
            synthesized += 1;

            if let Some(denot) = payload_denotation.as_deref() {
                new_value_info.push(stamped_value_info(&minted, denot));
            }
            new_value_info.push(stamped_value_info(&minted_sender, "bb.peer_id"));

            recvs_by_send.entry(send_idx).or_default().push(NodeProto {
                op_type: RECV_OP.into(),
                domain: WIRE_DOMAIN.into(),
                input: vec![],
                output: vec![minted.clone(), minted_sender],
                metadata_props: vec![
                    StringStringEntryProto {
                        key: HOME_CLASS_KEY.into(),
                        value: class,
                    },
                    StringStringEntryProto {
                        key: SYNTHESIZED_FROM_KEY.into(),
                        // Store the Send's original first-output VALUE NAME
                        // (not the node index) so that
                        // `check_wire_edge_types` can look it up via
                        // `TypeSolution::type_of`, which is keyed by value
                        // name. The TypeSolution is built before
                        // `synthesize_wire_recvs` runs, so it records the
                        // pre-sentinel `data_name`. Phase-2 ordering uses
                        // `recvs_by_send` (built during Phase 1 by index)
                        // and no longer parses this field as a usize.
                        value: data_name.clone(),
                    },
                ],
                ..Default::default()
            });

            // Stage a per-input rewrite: consumer N's `data_name`
            // input becomes `minted`. If a different Send already
            // staged a rewrite for the same (consumer_idx,
            // data_name) pair we accept the first rewrite — the
            // distinct map key (input_name) means N-fanin consumers
            // reading distinct value names from N Sends preserve
            // every rewrite.
            for consumer_idx in consumer_indices {
                rewrites
                    .entry((consumer_idx, data_name.clone()))
                    .or_insert_with(|| minted.clone());
            }
        }
    }

    // emit the rewritten node list in topological order:
    // each original node (rewritten if it's a Send or a consumer)
    // is emitted, and synthesized Recvs are interleaved immediately
    // after the Send that produced them so the partitioner's
    // BFS sees `Send → Recv → consumer` order.
    // `recvs_by_send` was built during Phase 1 (keyed by send_idx),
    // so no re-parsing of SYNTHESIZED_FROM_KEY is needed here.

    let mut emitted: Vec<NodeProto> = Vec::with_capacity(snapshot.len() + synthesized);
    for (idx, n) in snapshot.iter().enumerate() {
        let mut clone = n.clone();
        if let Some(sentinel) = sentinels.get(&idx) {
            clone.output[0] = sentinel.clone();
        }
        for input in clone.input.iter_mut() {
            if let Some(replacement) = rewrites.get(&(idx, input.clone())) {
                *input = replacement.clone();
            }
        }
        emitted.push(clone);
        if let Some(recvs) = recvs_by_send.remove(&idx) {
            emitted.extend(recvs);
        }
    }
    graph.node = emitted;
    graph.value_info.extend(new_value_info);
    Ok(synthesized)
}

fn stamped_value_info(name: &str, denotation: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            denotation: denotation.to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

