//! Op invocation lifecycle
//!
//! implements the `Atomic { component_ref }` dispatch path
//! routing through `ProtocolRuntime::dispatch_atomic` (the
//! universal-pair-only role trait - see design call #4 in
//! `docs/internal/IMPLEMENTATION_PLAN.md` ). The `Stateless`
//! syscall path lands.

use crate::atomic::DispatchResult;
use crate::bus::{InfraEvent, NodeEvent, OpError};
use crate::engine::call_context::CallContext;
use crate::engine::core::{graph_name_for, Engine};
use crate::engine::dispatch_entry::{FunctionKey, OpDispatch, StatelessInvokeFn};
use crate::engine::pending_async::PendingAsync;
use crate::engine::step::EngineStep;
use crate::ids::{ComponentRef, ExecId, NodeSiteId, OpRef};
use crate::roles::ProtocolRuntime;
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;
use bb_ir::proto::onnx::NodeProto;

impl Engine {
    /// Dispatch a single `(op_ref, exec_id)` per ENGINE.md §8.1.
    /// Returns one `EngineStep` describing the outcome
    /// (`OpCompleted` / `AsyncSuspended` / `OpFailed`).
    ///
    /// Consult `dispatch_for` for the `OpDispatch` variant stamped at
    /// install time by `Engine::resolve_dispatch` and route to the
    /// dedicated helper. An op without a stamped entry - or stamped
    /// `Unresolved` - is a build invariant violation that surfaced
    /// past the `Node` pre-flight; fail the op so the caller
    /// sees a clean `OpFailure` rather than a panic.
    pub(crate) fn invoke_one(&mut self, op_ref: OpRef, exec_id: ExecId) -> EngineStep {
        let node = match self.node_for(op_ref) {
            Some(n) => n.clone(),
            None => {
                return self.fail_op(
                    op_ref,
                    exec_id,
                    crate::bus::OpErrorKind::ExecutionFailed,
                    "unknown_op_ref",
                    "unknown op_ref".to_string(),
                )
            }
        };

        // Structured op fields on the per-invocation span let
        // operators filter and group traces by op kind, domain, or
        // a specific (exec_id, op_ref) instance — equivalent to the
        // op-keyed metadata ONNX Runtime emits via EventRecord.args.
        let _invoke_span = tracing::debug_span!(
            "engine.invoke_one",
            op.name = %node.name,
            op.kind = %node.op_type,
            op.domain = %node.domain,
            exec_id = %exec_id,
            op_ref = %op_ref,
        )
        .entered();

        match self.dispatch_for(op_ref) {
            Some(OpDispatch::Stateless(invoke_fn)) => {
                self.invoke_stateless(op_ref, exec_id, &node, invoke_fn)
            }
            Some(OpDispatch::Atomic {
                component_ref,
                dispatch_fn,
            }) => self.invoke_atomic(op_ref, exec_id, &node, component_ref, dispatch_fn),
            Some(OpDispatch::FunctionCall {
                target,
                input_rename,
                output_rename,
            }) => {
                self.invoke_function_call(op_ref, exec_id, &target, &input_rename, &output_rename)
            }
            Some(OpDispatch::Unresolved) | None => self.fail_op(
                op_ref,
                exec_id,
                crate::bus::OpErrorKind::NotRegistered,
                "unresolved_dispatch",
                format!("unresolved dispatch for {}::{}", node.domain, node.op_type),
            ),
        }
    }

    /// Look up the install-time-stamped `OpDispatch` for an `OpRef`.
    /// Positional resolution per C5: `OpRef::pack(graph_idx, node_idx)`
    /// → two direct array accesses, no HashMap probes.
    fn dispatch_for(&self, op_ref: OpRef) -> Option<OpDispatch> {
        let (gi, ni) = op_ref.split();
        self.graphs
            .get(gi as usize)?
            .op_dispatch
            .get(ni as usize)
            .cloned()
    }

    /// Dispatch through `ProtocolRuntime::dispatch_atomic` for an op
    /// resolved to a bound component. Factored out of the legacy
    /// `invoke_one` cascade so the new dispatch path can call it
    /// directly without re-doing the syscall / atomic_dispatch lookup.
    fn invoke_atomic(
        &mut self,
        op_ref: OpRef,
        exec_id: ExecId,
        node: &NodeProto,
        component_ref: ComponentRef,
        dispatch_fn: ProtocolDispatchFn,
    ) -> EngineStep {
        // Defensive cap check BEFORE dispatch — any atomic op that
        // returns Async would need to insert into pending_async, so
        // when we're already at cap we reject early. Components that
        // side-effect (e.g. allocate wire requests, push outbound
        // envelopes) inside dispatch don't see the cap rejection
        // until AFTER their effects run, leaking that state into the
        // engine. Pre-dispatch rejection covers Immediate ops too but
        // engine-at-cap is the right backpressure signal regardless.
        if let Some(cap) = self.max_pending_async {
            if self.exec.pending_async.len() >= cap {
                return self.fail_op(
                    op_ref,
                    exec_id,
                    crate::bus::OpErrorKind::Cooldown,
                    "pending_async_limit",
                    "pending-async limit exceeded".to_string(),
                );
            }
        }

        // Alias-aware: when `exec_id` is a function-body's derived id,
        // formal inputs route through `pending_calls[exec_id]` to read
        // the caller's slot at `parent_exec_id` (zero-copy).
        let input_pairs = self.resolve_input_pairs(node, exec_id);

        // D2 take-and-restore: lift the dispatching component out
        // of the Vec so a live ComponentsView can borrow the rest of
        // engine.components while the closure runs. Restore the slot
        // unconditionally on the way out so a panic doesn't leak the
        // Box (Rust catches via Drop, but the slot would stay None
        // and confuse subsequent dispatches).
        let Some(mut taken) = self.take_component(component_ref) else {
            return self.fail_op(
                op_ref,
                exec_id,
                crate::bus::OpErrorKind::MissingSlot,
                "component_missing",
                "component missing".to_string(),
            );
        };

        let result: Result<DispatchResult, String> = {
            let mut input_refs: Vec<(String, &dyn SlotValue)> =
                Vec::with_capacity(input_pairs.len());
            for (site, name, read_exec_id) in &input_pairs {
                if let Some(Some(boxed)) = self.exec.slot_table.get(&(*site, *read_exec_id)) {
                    input_refs.push((name.clone(), boxed.as_ref()));
                }
            }
            let inputs_for_dispatch: Vec<(&str, &dyn SlotValue)> =
                input_refs.iter().map(|(n, h)| (n.as_str(), *h)).collect();

            let (
                envelope_src_peer,
                inbound_correlation_wire_req_id,
                inbound_arrival_ns,
                inbound_remaining_deadline_ns,
            ) = self
                .framework
                .inbound_contexts
                .get(&exec_id)
                .map(|c| {
                    (
                        c.src_peer,
                        c.wire_req_id,
                        c.arrival_ns,
                        c.remaining_deadline_ns,
                    )
                })
                .unwrap_or((None, None, None, None));
            let mut ctx = RuntimeResourceRef {
                peers: crate::runtime::PeerCtx {
                    gate: &mut self.framework.peer_state.gate,
                    backoff: &mut self.framework.peer_state.backoff,
                    governor: &mut self.framework.peer_state.governor,
                    addresses: &mut self.framework.address_book,
                    backpressure: &mut self.framework.peer_state.backpressure,
                },
                net: crate::runtime::NetCtx {
                    outbound: &mut self.framework.outbound_queue,
                    rtt: &mut self.framework.rtt_tracker,
                    requests: &mut self.framework.request_tracker,
                    dedup: &mut self.framework.inbound_dedup,
                    pending_peer_resolve_failures: &mut self
                        .framework
                        .pending_peer_resolve_failures,
                },
                time: crate::runtime::TimeCtx {
                    scheduler: &mut self.framework.scheduler,
                },
                syscall: crate::runtime::SyscallCtx {
                    serialize_queue: &mut self.framework.serialize_queue,
                    hold_table: &mut self.framework.hold_table,
                    record_buffer: &mut self.framework.record_buffer,
                    event_source: &mut self.framework.event_source,
                    counters: &mut self.framework.counters,
                    any_fired_groups: &mut self.framework.any_fired_groups,
                    deadline_match_fired: &mut self.framework.deadline_match_fired,
                    rng: &mut *self.framework.rng,
                    pending_app_events: &mut self.framework.pending_app_events,
                },
                bus: &mut self.bus,
                ingress: std::sync::Arc::clone(&self.ingress),
                components: crate::runtime::ComponentsView {
                    instances: Some(&self.components),
                    slots: Some(&self.slots),
                },
                current: crate::runtime::CurrentCallCtx {
                    op_ref,
                    exec_id,
                    self_peer: self.self_peer,
                    node_attributes: &node.attribute,
                    node_metadata: &node.metadata_props,
                    inbound: crate::runtime::InboundCtx {
                        src_peer: envelope_src_peer,
                        wire_req_id: inbound_correlation_wire_req_id,
                        arrival_ns: inbound_arrival_ns,
                        remaining_deadline_ns: inbound_remaining_deadline_ns,
                    },
                    pending_completions: Vec::new(),
                    next_command_id: &mut self.exec.ids.next_command_id,
                },
            };

            // The install-time-stamped dispatch_fn is the one
            // canonical hot path — runtime calls the downcast
            // closure directly, no per-op TypeId HashMap probe.
            let any: &mut dyn std::any::Any = taken.as_mut();
            let dispatch_result = dispatch_fn(any, &node.op_type, &inputs_for_dispatch, &mut ctx);

            let captured = std::mem::take(&mut ctx.current.pending_completions);
            drop(ctx);
            self.exec.pending_completions.extend(captured);

            dispatch_result
        };

        // Always restore - keeps the slot table consistent across
        // dispatch outcomes (Immediate, Async, or error).
        self.restore_component(component_ref, taken);

        match result {
            Ok(DispatchResult::Immediate(outputs)) => {
                let sites = self.write_outputs(op_ref, exec_id, outputs);
                EngineStep::OpCompleted {
                    op_ref,
                    exec_id,
                    sites_written: sites,
                }
            }
            Ok(DispatchResult::Async(cmd_id)) => {
                let output_sites = self.op_output_sites(op_ref);
                self.exec.pending_async.insert(
                    cmd_id,
                    PendingAsync {
                        op_ref,
                        exec_id,
                        output_sites,
                        deadline_ns: None,
                    },
                );
                EngineStep::AsyncSuspended {
                    op_ref,
                    exec_id,
                    cmd_id,
                }
            }
            Err(detail) => self.fail_op(
                op_ref,
                exec_id,
                crate::bus::OpErrorKind::ExecutionFailed,
                "stateless_invoke",
                detail,
            ),
        }
    }

    /// Splice a function-call invocation per ENGINE.md §8.4.
    ///
    /// Flat-frontier model: the body's `OpRef`s are shared (one
    /// allocation in `engine.graphs[graph_name_for(target)]`); each call
    /// allocates a fresh body `ExecId` and the body's nodes execute at
    /// that derived id. Input slots are NOT copied - the body's formal
    /// parameter reads route through `input_aliases` to the caller's
    /// slot at `parent_exec_id` (zero-copy). Output forwarding is
    /// applied in `write_outputs` (Phase 2b §2b.8).
    ///
    /// Returns `OpCompleted` for the call site itself with no
    /// `sites_written` - the body's writes surface as separate events
    /// as each body node fires.
    pub(crate) fn invoke_function_call(
        &mut self,
        op_ref: OpRef,
        parent_exec_id: ExecId,
        target: &FunctionKey,
        input_rename: &[(String, String)],
        output_rename: &[(String, String)],
    ) -> EngineStep {
        let graph_name = graph_name_for(target);
        if !self.has_graph(&graph_name) {
            return self.fail_op(
                op_ref,
                parent_exec_id,
                crate::bus::OpErrorKind::NotRegistered,
                "function_target_missing",
                format!("function-call target {graph_name} not installed"),
            );
        }

        let body_exec_id = self.allocate_exec_id();
        let body = self.graph(&graph_name).expect("checked above");

        // Snapshot body OpRefs + formal-name → body-site map before
        // releasing the borrow on self.graphs. With positional
        // OpRefs the body's op_refs are `OpRef::pack(body_idx,
        // node_idx)` for `node_idx in 0..body.function.node.len()`.
        let body_idx = self.graph_idx(&graph_name).expect("graph just resolved");
        let body_op_refs: Vec<OpRef> = (0..body.function.node.len() as u32)
            .map(|ni| OpRef::pack(body_idx, ni))
            .collect();
        let body_site_for: std::collections::HashMap<String, NodeSiteId> = body.site_names.clone();

        // input_rename pairs: (caller_value_name, formal_parameter_name)
        // Caller-side names are scoped to the call op's owning graph -
        // body-side names may coincidentally spell the same string.
        let mut input_aliases: std::collections::HashMap<String, NodeSiteId> =
            std::collections::HashMap::with_capacity(input_rename.len());
        for (caller_name, formal_name) in input_rename {
            let Some(caller_site) = self.resolve_site_in_op_graph(op_ref, caller_name) else {
                return self.fail_op(
                    op_ref,
                    parent_exec_id,
                    crate::bus::OpErrorKind::MissingSlot,
                    "function_input_unbound",
                    format!("function-call input {caller_name} not bound"),
                );
            };
            input_aliases.insert(formal_name.clone(), caller_site);
        }

        // output_rename pairs: (formal_output_name, caller_value_name)
        let mut output_forwarding: std::collections::HashMap<NodeSiteId, NodeSiteId> =
            std::collections::HashMap::with_capacity(output_rename.len());
        for (formal_out, caller_out) in output_rename {
            let Some(&body_site) = body_site_for.get(formal_out) else {
                return self.fail_op(
                    op_ref,
                    parent_exec_id,
                    crate::bus::OpErrorKind::NotRegistered,
                    "function_output_missing",
                    format!("function-call output {formal_out} missing from body"),
                );
            };
            let Some(caller_site) = self.resolve_site_in_op_graph(op_ref, caller_out) else {
                return self.fail_op(
                    op_ref,
                    parent_exec_id,
                    crate::bus::OpErrorKind::MissingSlot,
                    "function_output_unbound",
                    format!("function-call output {caller_out} not bound"),
                );
            };
            output_forwarding.insert(body_site, caller_site);
        }

        let outputs_remaining = output_forwarding.len();
        self.exec.pending_calls.insert(
            body_exec_id,
            CallContext {
                parent_exec_id,
                target: target.clone(),
                input_aliases,
                output_forwarding,
                outputs_remaining,
            },
        );

        // Push every body OpRef onto the frontier at body_exec_id.
        // Alias-aware `all_inputs_ready` (Phase 2b §2b.7) gates each
        // node until its formal inputs are populated at the caller's
        // ExecId. Plan §2b.6 step 6: simple v1 - push all, gate.
        for body_op in body_op_refs {
            self.exec.frontier.push_back((body_op, body_exec_id));
        }

        // Zero-outputs corner case: with no forwarding to wait on, the
        // call is conceptually complete the moment the body finishes.
        // `write_outputs`' forwarding hook drops the entry once
        // `outputs_remaining` hits zero - for a no-output call we drop
        // immediately so we don't leak an entry.
        if self
            .exec
            .pending_calls
            .get(&body_exec_id)
            .map(|c| c.outputs_remaining == 0)
            .unwrap_or(false)
        {
            self.exec.pending_calls.remove(&body_exec_id);
        }

        EngineStep::OpCompleted {
            op_ref,
            exec_id: parent_exec_id,
            sites_written: Vec::new(),
        }
    }

    /// Dispatch a stateless syscall op. Shares the input-resolution,
    /// split-borrow, and result-handling shape with `invoke_one`'s
    /// atomic path; differs in calling the fn pointer directly rather
    /// than going through `components[component_ref]`.
    pub(crate) fn invoke_stateless(
        &mut self,
        op_ref: OpRef,
        exec_id: ExecId,
        node: &NodeProto,
        invoke_fn: StatelessInvokeFn,
    ) -> EngineStep {
        // Same pre-dispatch pending_async cap check as invoke_atomic
        // — syscalls (wire.Send, Sleep, ...) routinely side-effect
        // before returning Async, so a post-dispatch rejection would
        // leak state.
        if let Some(cap) = self.max_pending_async {
            if self.exec.pending_async.len() >= cap {
                return self.fail_op(
                    op_ref,
                    exec_id,
                    crate::bus::OpErrorKind::Cooldown,
                    "pending_async_limit",
                    "pending-async limit exceeded".to_string(),
                );
            }
        }
        // Alias-aware input resolution - see `invoke_atomic` for the
        // function-body splice semantics.
        let input_pairs = self.resolve_input_pairs(node, exec_id);

        let result: Result<DispatchResult, OpError> = {
            let mut input_refs: Vec<(String, &dyn SlotValue)> =
                Vec::with_capacity(input_pairs.len());
            for (site, name, read_exec_id) in &input_pairs {
                if let Some(Some(boxed)) = self.exec.slot_table.get(&(*site, *read_exec_id)) {
                    input_refs.push((name.clone(), boxed.as_ref()));
                }
            }
            let inputs_for_dispatch: Vec<(&str, &dyn SlotValue)> =
                input_refs.iter().map(|(n, h)| (n.as_str(), *h)).collect();

            let (
                envelope_src_peer,
                inbound_correlation_wire_req_id,
                inbound_arrival_ns,
                inbound_remaining_deadline_ns,
            ) = self
                .framework
                .inbound_contexts
                .get(&exec_id)
                .map(|c| {
                    (
                        c.src_peer,
                        c.wire_req_id,
                        c.arrival_ns,
                        c.remaining_deadline_ns,
                    )
                })
                .unwrap_or((None, None, None, None));
            let mut ctx = RuntimeResourceRef {
                peers: crate::runtime::PeerCtx {
                    gate: &mut self.framework.peer_state.gate,
                    backoff: &mut self.framework.peer_state.backoff,
                    governor: &mut self.framework.peer_state.governor,
                    addresses: &mut self.framework.address_book,
                    backpressure: &mut self.framework.peer_state.backpressure,
                },
                net: crate::runtime::NetCtx {
                    outbound: &mut self.framework.outbound_queue,
                    rtt: &mut self.framework.rtt_tracker,
                    requests: &mut self.framework.request_tracker,
                    dedup: &mut self.framework.inbound_dedup,
                    pending_peer_resolve_failures: &mut self
                        .framework
                        .pending_peer_resolve_failures,
                },
                time: crate::runtime::TimeCtx {
                    scheduler: &mut self.framework.scheduler,
                },
                syscall: crate::runtime::SyscallCtx {
                    serialize_queue: &mut self.framework.serialize_queue,
                    hold_table: &mut self.framework.hold_table,
                    record_buffer: &mut self.framework.record_buffer,
                    event_source: &mut self.framework.event_source,
                    counters: &mut self.framework.counters,
                    any_fired_groups: &mut self.framework.any_fired_groups,
                    deadline_match_fired: &mut self.framework.deadline_match_fired,
                    rng: &mut *self.framework.rng,
                    pending_app_events: &mut self.framework.pending_app_events,
                },
                bus: &mut self.bus,
                ingress: std::sync::Arc::clone(&self.ingress),
                components: crate::runtime::ComponentsView::default(),
                current: crate::runtime::CurrentCallCtx {
                    op_ref,
                    exec_id,
                    self_peer: self.self_peer,
                    node_attributes: &node.attribute,
                    node_metadata: &node.metadata_props,
                    inbound: crate::runtime::InboundCtx {
                        src_peer: envelope_src_peer,
                        wire_req_id: inbound_correlation_wire_req_id,
                        arrival_ns: inbound_arrival_ns,
                        remaining_deadline_ns: inbound_remaining_deadline_ns,
                    },
                    pending_completions: Vec::new(),
                    next_command_id: &mut self.exec.ids.next_command_id,
                },
            };

            let dispatch_result = invoke_fn(node, &inputs_for_dispatch, &mut ctx);

            let captured = std::mem::take(&mut ctx.current.pending_completions);
            drop(ctx);
            self.exec.pending_completions.extend(captured);

            dispatch_result
        };

        match result {
            Ok(DispatchResult::Immediate(outputs)) => {
                let sites = self.write_outputs(op_ref, exec_id, outputs);
                EngineStep::OpCompleted {
                    op_ref,
                    exec_id,
                    sites_written: sites,
                }
            }
            Ok(DispatchResult::Async(cmd_id)) => {
                let output_sites = self.op_output_sites(op_ref);
                self.exec.pending_async.insert(
                    cmd_id,
                    PendingAsync {
                        op_ref,
                        exec_id,
                        output_sites,
                        deadline_ns: None,
                    },
                );
                EngineStep::AsyncSuspended {
                    op_ref,
                    exec_id,
                    cmd_id,
                }
            }
            Err(err) => {
                self.bus.publish(NodeEvent::Infra(InfraEvent::OpFailure {
                    op_ref,
                    error: err.clone(),
                }));
                EngineStep::OpFailed {
                    op_ref,
                    exec_id,
                    error: err,
                }
            }
        }
    }

    /// Look up a NodeProto by op_ref. Positional resolution via
    /// `OpRef::pack(graph_idx, node_idx)` → two direct array indexes.
    pub(crate) fn node_for(&self, op_ref: OpRef) -> Option<&NodeProto> {
        let (gi, ni) = op_ref.split();
        self.graphs.get(gi as usize)?.function.node.get(ni as usize)
    }

    /// Output sites the given Op declared. Positional resolution
    /// looks up the NodeProto directly; each declared output name
    /// resolves through the graph's `site_names` map (with a
    /// deterministic synth as a fallback for test setups).
    pub(crate) fn op_output_sites(&self, op_ref: OpRef) -> Vec<NodeSiteId> {
        let (gi, ni) = op_ref.split();
        let Some(g) = self.graphs.get(gi as usize) else {
            return Vec::new();
        };
        let Some(node) = g.function.node.get(ni as usize) else {
            return Vec::new();
        };
        node.output
            .iter()
            .enumerate()
            .map(|(i, name)| {
                g.site_names
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| synthesize_site_id(op_ref, i))
            })
            .collect()
    }

    /// Resolve a value-name to its `NodeSiteId` across every
    /// GraphSlot. Returns `None` if no graph has a binding.
    pub(crate) fn resolve_site_name(&self, name: &str) -> Option<NodeSiteId> {
        for g in self.graphs_iter() {
            if let Some(&site) = g.site_names.get(name) {
                return Some(site);
            }
        }
        None
    }

    /// Resolve a value-name to its `NodeSiteId` within the GraphSlot
    /// that owns `op_ref`. Used by `invoke_function_call` to disambiguate
    /// caller-side value names from formal-parameter names that happen
    /// to spell the same string in the callee body (hoisted bodies
    /// canonicalize to `__hoist_*` so this clash is rare in production,
    /// but graph-scoped lookup is the principled fix). Positional
    /// `OpRef::pack` makes the owning graph a direct index.
    pub(crate) fn resolve_site_in_op_graph(&self, op_ref: OpRef, name: &str) -> Option<NodeSiteId> {
        let (gi, _) = op_ref.split();
        self.graphs.get(gi as usize)?.site_names.get(name).copied()
    }

    /// Resolve a NodeProto's inputs into `(NodeSiteId, value_name,
    /// read_exec_id)` triples per ENGINE.md §8.4.
    ///
    /// Function-call splice: when `exec_id` is a body's derived
    /// `ExecId` (i.e. present in `pending_calls`), formal inputs that
    /// appear in `input_aliases` re-route reads to the caller's
    /// `parent_exec_id` and the caller-side `NodeSiteId` - zero-copy
    /// aliasing, the body never copies the caller's slot. Inputs not
    /// in the alias map fall back to the standard
    /// `resolve_site_name(name) → exec_id` path so body-internal
    /// values keep flowing at `body_exec_id`.
    ///
    /// Empty input names (ONNX optional-arg convention) and inputs
    /// that fail to resolve are skipped - the caller checks readiness
    /// separately via `all_inputs_ready`.
    pub(crate) fn resolve_input_pairs(
        &self,
        node: &NodeProto,
        exec_id: ExecId,
    ) -> Vec<(NodeSiteId, String, ExecId)> {
        let cc = self.exec.pending_calls.get(&exec_id);
        let mut out = Vec::new();
        for name in &node.input {
            if name.is_empty() {
                continue;
            }
            if let Some(cc) = cc {
                if let Some(&alias_site) = cc.input_aliases.get(name) {
                    out.push((alias_site, name.clone(), cc.parent_exec_id));
                    continue;
                }
            }
            if let Some(site) = self.resolve_site_name(name) {
                out.push((site, name.clone(), exec_id));
            }
        }
        out
    }

    /// Write Op outputs to the slot_table + push ready downstream
    /// consumers onto the frontier. Returns the list of sites
    /// written (for `EngineStep::OpCompleted`).
    pub(crate) fn write_outputs(
        &mut self,
        op_ref: OpRef,
        exec_id: ExecId,
        outputs: Vec<(String, Box<dyn SlotValue>)>,
    ) -> Vec<NodeSiteId> {
        let output_sites = self.op_output_sites(op_ref);
        for ((site, _name), value) in output_sites
            .iter()
            .zip(outputs.iter().map(|(n, _)| n))
            .zip(outputs.iter().map(|(_, v)| v.as_ref()))
        {
            // Place each value into its slot. We can't move out of
            // `outputs` while iterating borrows, so we collect in
            // two passes. (See below.)
            let _ = (site, _name, value);
        }
        // Two-pass placement: zip site_index → outputs by position.
        let mut sites_written: Vec<NodeSiteId> = Vec::new();
        for (i, (_name, value)) in outputs.into_iter().enumerate() {
            if let Some(site) = output_sites.get(i).copied() {
                self.exec.slot_table.insert((site, exec_id), Some(value));
                sites_written.push(site);
            }
        }

        // Bump execution_state.
        self.exec
            .execution_state
            .entry(exec_id)
            .or_default()
            .outputs_written += sites_written.len() as u32;

        // Push consumers whose inputs are now all ready.
        self.push_ready_consumers(&sites_written, exec_id);

        // Function-call splice: when `exec_id` is a body's derived id,
        // forward any matching outputs back to the caller's slots at
        // `parent_exec_id` and push the caller's downstream consumers.
        // No-op when there's no pending call context.
        self.forward_outputs_to_caller(&sites_written, exec_id);

        self.surface_top_level_outputs(&sites_written, exec_id);
        sites_written
    }

    /// Phase 2b §2b.8 - output forwarding for the function-call splice.
    ///
    /// If `exec_id` keys a `CallContext` in `pending_calls`, every
    /// `sites_written` entry that appears in `output_forwarding` gets
    /// its `SlotValue` MOVED from the body's slot at `exec_id` to
    /// the caller's slot at `parent_exec_id`. `SlotValue` is not
    /// `Clone`; a body output is one-shot, so moving the value is
    /// semantically correct.
    ///
    /// After moving values, pushes the caller-side consumers onto the
    /// frontier at `parent_exec_id` and surfaces top-level outputs.
    /// Drops the `pending_calls` entry once `outputs_remaining`
    /// reaches zero.
    pub(crate) fn forward_outputs_to_caller(
        &mut self,
        sites_written: &[NodeSiteId],
        exec_id: ExecId,
    ) {
        let Some(cc) = self.exec.pending_calls.get(&exec_id) else {
            return;
        };
        // Collect the body_site → caller_site pairs we'll forward,
        // dropping the immutable borrow before mutating slot_table.
        let mut pairs: Vec<(NodeSiteId, NodeSiteId)> = Vec::new();
        for &body_site in sites_written {
            if let Some(&caller_site) = cc.output_forwarding.get(&body_site) {
                pairs.push((body_site, caller_site));
            }
        }
        let parent_exec_id = cc.parent_exec_id;
        if pairs.is_empty() {
            return;
        }
        tracing::trace!(
            target: "engine.function_call.forward",
            call_target = ?cc.target,
            body_exec_id = exec_id.as_u64(),
            parent_exec_id = parent_exec_id.as_u64(),
            pair_count = pairs.len(),
            "forwarding body outputs to caller slots",
        );

        let mut caller_sites: Vec<NodeSiteId> = Vec::with_capacity(pairs.len());
        for (body_site, caller_site) in &pairs {
            // MOVE the value: body's slot loses the value, caller's
            // slot owns it. Skip if the body slot is empty (e.g. the
            // op produced fewer outputs than declared).
            let value = self
                .exec
                .slot_table
                .get_mut(&(*body_site, exec_id))
                .and_then(|opt| opt.take());
            if let Some(value) = value {
                self.exec
                    .slot_table
                    .insert((*caller_site, parent_exec_id), Some(value));
                caller_sites.push(*caller_site);
            }
        }

        // Decrement `outputs_remaining` and drop the entry on
        // completion in a single mut borrow.
        if let Some(cc) = self.exec.pending_calls.get_mut(&exec_id) {
            cc.outputs_remaining = cc.outputs_remaining.saturating_sub(pairs.len());
            if cc.outputs_remaining == 0 {
                self.exec.pending_calls.remove(&exec_id);
            }
        }

        // Push caller-side consumers + surface top-level outputs
        // against the caller's `parent_exec_id`.
        self.push_ready_consumers(&caller_sites, parent_exec_id);
        self.surface_top_level_outputs(&caller_sites, parent_exec_id);
    }

    /// Inspect each freshly-written site: when it corresponds to a
    /// declared `function.output` AND no downstream Op consumes it
    /// inside the function, push an `AppEvent::Emit` onto
    /// `framework.pending_app_events` carrying the slot's serialized
    /// bytes. Phase 8 drains the queue into `EngineStep::AppEvent`
    /// for the host.
    ///
    /// This is the "function signature is the engine I/O contract"
    /// path. Coexists with the explicit `AppEmit` / `AppNotify`
    /// syscall ops from `src/syscall/telemetry/`; both surface as the
    /// same `EngineStep::AppEvent` variant.
    ///
    /// Encode the slot's value for host consumption: a `BytesValue`
    /// surfaces its inner `Vec<u8>` directly (the byte-level
    /// contract), any other carrier surfaces its bincode-encoded
    /// form. Push a `Emit { name, value_bytes }` for every top-
    /// level output site with no in-graph consumer.
    pub(crate) fn surface_top_level_outputs(&mut self, sites: &[NodeSiteId], exec_id: ExecId) {
        for site in sites {
            let consumer_count = self
                .graphs_iter()
                .map(|g| g.consumers.get(site).map(|v| v.len()).unwrap_or(0))
                .sum::<usize>();
            if consumer_count > 0 {
                continue;
            }
            let name_opt = self
                .graphs_iter()
                .find_map(|g| g.top_level_outputs.get(site).cloned());
            let Some(name) = name_opt else { continue };
            let value_bytes = self
                .exec
                .slot_table
                .get(&(*site, exec_id))
                .and_then(|slot| slot.as_ref())
                .map(|boxed| encode_for_host(boxed.as_ref()))
                .unwrap_or_default();
            self.framework
                .pending_app_events
                .push(crate::bus::AppEvent::Emit { name, value_bytes });
        }
    }

    /// Push consumer Ops onto the frontier when all their inputs are
    /// satisfied. minimum-viable: iterates each graph's
    /// `consumers` map for each newly-written site.
    pub(crate) fn push_ready_consumers(&mut self, sites: &[NodeSiteId], exec_id: ExecId) {
        // Collect candidate (consumer_op) refs across graphs first,
        // then filter by readiness, then push - avoids borrow
        // conflicts.
        let mut candidates: Vec<OpRef> = Vec::new();
        for site in sites {
            for g in self.graphs_iter() {
                if let Some(consumers) = g.consumers.get(site) {
                    candidates.extend(consumers.iter().copied());
                }
            }
        }
        for op_ref in candidates {
            if self.all_inputs_ready(op_ref, exec_id) {
                self.exec.frontier.push_back((op_ref, exec_id));
            }
        }
    }

    /// Per ENGINE.md §8.3 - all `Required` inputs must be filled.
    /// treats every input as `Required` (no `AnyOf` yet -
    /// 's syscall opset introduces optional-input ops).
    pub(crate) fn all_inputs_ready(&self, op_ref: OpRef, exec_id: ExecId) -> bool {
        let Some(node) = self.node_for(op_ref) else {
            return false;
        };
        // Mirror `resolve_input_pairs`' alias-aware lookup: when
        // `exec_id` is a body's derived id, formal inputs read from
        // the caller's slot at `parent_exec_id` (per ENGINE.md §8.4).
        let cc = self.exec.pending_calls.get(&exec_id);
        for name in &node.input {
            if name.is_empty() {
                continue; // ONNX optional-arg convention.
            }
            let (site, read_exec_id) = if let Some(cc) = cc {
                if let Some(&alias_site) = cc.input_aliases.get(name) {
                    (alias_site, cc.parent_exec_id)
                } else {
                    let Some(site) = self.resolve_site_name(name) else {
                        return false;
                    };
                    (site, exec_id)
                }
            } else {
                let Some(site) = self.resolve_site_name(name) else {
                    return false;
                };
                (site, exec_id)
            };
            let has_value = self
                .exec
                .slot_table
                .get(&(site, read_exec_id))
                .map(|s| s.is_some())
                .unwrap_or(false);
            if !has_value {
                return false;
            }
        }
        true
    }

    /// Surface an Op failure: emit `InfraEvent::OpFailure` onto the
    /// bus AND return `EngineStep::OpFailed`. Every internal call
    /// site classifies the failure so operators can match on the
    /// `OpErrorKind` taxonomy for retry/report/drop policy without
    /// parsing freeform detail strings.
    pub(crate) fn fail_op(
        &mut self,
        op_ref: OpRef,
        exec_id: ExecId,
        kind: crate::bus::OpErrorKind,
        reason: &'static str,
        detail: String,
    ) -> EngineStep {
        let error = OpError {
            kind,
            reason,
            detail,
        };
        self.bus.publish(NodeEvent::Infra(InfraEvent::OpFailure {
            op_ref,
            error: error.clone(),
        }));
        EngineStep::OpFailed {
            op_ref,
            exec_id,
            error,
        }
    }
}

/// Encode a slot value for host-visible delivery. `BytesValue`
/// surfaces its inner bytes raw - the byte-level contract callers
/// expect when their function output is already wire-formatted.
/// Any other carrier surfaces its bincode-encoded form via
/// `SlotValue::to_wire_bytes`.
///
/// Encode a slot value's wire bytes for host delivery. `BytesValue`
/// short-circuits to its stored bytes. Other values go through
/// `to_wire_bytes`; encode failures here are non-fatal — the host
/// gets empty bytes and a `tracing::warn` records the failure so an
/// observer can attribute the drop to the value rather than to a
/// missing emit.
fn encode_for_host(value: &dyn crate::slot_value::SlotValue) -> Vec<u8> {
    if let Some(b) = value
        .as_any()
        .downcast_ref::<crate::syscall::values::BytesValue>()
    {
        return b.0.clone();
    }
    match value.to_wire_bytes() {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!(error = %e, "encode_for_host: dropping host emit on encode failure");
            Vec::new()
        }
    }
}

/// deterministic NodeSiteId synthesizer for test graphs
/// that don't populate `site_names`. 's Node
/// populates the real map.
fn synthesize_site_id(op_ref: OpRef, output_index: usize) -> NodeSiteId {
    NodeSiteId::from((op_ref.as_u64() << 8) | (output_index as u64 & 0xff))
}

/// Walk a `ProtocolRuntime` dispatcher slice, find the entry that
/// matches the bound component's concrete type, and route the call
/// through its `dispatch_atomic` impl.
///
/// The dispatcher registry is per-Engine (stored on
/// `Engine.role_dispatchers`); the caller passes the slice in so
/// borrow-splitting at the call site can keep `&mut self.components`
/// and `&mut self.framework` independent.
pub(crate) fn call_protocol_dispatch_atomic(
    component: &mut dyn crate::component::ErasedComponent,
    op_type: &str,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
    dispatchers: &std::collections::HashMap<std::any::TypeId, RoleDispatcher>,
) -> Result<DispatchResult, String> {
    let any: &mut dyn std::any::Any = component;
    let tid = (*any).type_id();
    if let Some(dispatcher) = dispatchers.get(&tid) {
        (dispatcher.dispatch)(any, op_type, inputs, ctx)
    } else {
        Err("no ProtocolRuntime dispatcher registered for component".to_string())
    }
}

/// Type alias for the ProtocolRuntime downcast-dispatch fn pointer
/// stored in the dispatcher registry.
pub type ProtocolDispatchFn = fn(
    &mut dyn std::any::Any,
    &str,
    &[(&str, &dyn SlotValue)],
    &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, String>;

/// Type alias for the `BackendRuntime::materialize_from_wire`
/// downcast-dispatch fn pointer. Mirrors [`ProtocolDispatchFn`]'s
/// erased-`Any` shape so the per-T closure stays callable from the
/// engine's wire-decode hot path without re-doing the downcast lookup
/// on every fill.
pub type BackendMaterializeFn =
    fn(
        &mut dyn std::any::Any,
        u64,
        Vec<u8>,
    ) -> Result<Box<dyn SlotValue>, crate::slot_value::BackendMaterializeError>;

/// Type alias for the `Bootstrap::bootstrap` downcast-dispatch fn
/// pointer the engine stores per concrete Bootstrap impl. Mirrors
/// [`ProtocolDispatchFn`]'s erased-`Any` shape so the F3 Component
/// bootstrap fire path can invoke the impl without a per-TypeId
/// downcast on every call. The closure downcasts `any` to `T`,
/// runs `T::bootstrap(&mut BootstrapCtx)`, and reports the
/// `DispatchResult` (Immediate or Async) for the synthetic single-
/// op dispatch.
pub type BootstrapDispatchFn = fn(
    &mut dyn std::any::Any,
    &mut crate::contracts::bootstrap::BootstrapCtx,
) -> Result<DispatchResult, String>;

/// Build a [`BootstrapDispatchFn`] for a concrete `T: Bootstrap`.
/// Called from `Engine::register_bootstrap_dispatcher` so the
/// engine's `fire_component_bootstrap` lookup keys on `TypeId::of::<T>()`
/// and the synthetic op invokes the user's `T::bootstrap` directly.
///
/// The Contract method's return is `Result<(), T::Error>`; this
/// helper converts the `Ok(())` to `DispatchResult::Immediate(Vec::new())`
/// because Component bootstrap declares no output slots — the
/// F3 spec lists `Async` as a future option a Component's
/// override surfaces via a wrapping `CommandId`, but Commit 3
/// lands the synchronous-Immediate path that every default-no-op
/// Bootstrap takes.
pub fn make_bootstrap_dispatcher<T: crate::contracts::bootstrap::Bootstrap + 'static>(
) -> BootstrapDispatchFn
where
    T::Error: std::fmt::Display,
{
    |any, ctx| {
        let concrete = any
            .downcast_mut::<T>()
            .expect("type-erased lookup matched T");
        concrete
            .bootstrap(ctx)
            .map(|_| DispatchResult::Immediate(Vec::new()))
            .map_err(|e| e.to_string())
    }
}

/// Backend-only no-op materialize entry used by non-Backend roles.
/// Roles other than `Backend` never reach the materialize entry
/// (decode_typed_fill only consults it after confirming the slot
/// binds to `ComponentRole::Backend`), so the unused fn pointer
/// returns a descriptive error rather than panicking.
pub fn no_materialize(
    _any: &mut dyn std::any::Any,
    _type_hash: u64,
    _bytes: Vec<u8>,
) -> Result<Box<dyn SlotValue>, crate::slot_value::BackendMaterializeError> {
    Err(crate::slot_value::BackendMaterializeError {
        summary: "component is not a Backend; materialize_from_wire not supported".to_string(),
    })
}

/// One registered `ProtocolRuntime` / `<Role>Runtime` dispatcher.
/// `dispatch` downcasts to the concrete type `T` (guaranteed by the
/// `TypeId`-keyed registry) and delegates to `T::dispatch_atomic`.
/// `materialize` is the parallel entry for
/// [`crate::roles::BackendRuntime::materialize_from_wire`]; non-Backend
/// roles register [`no_materialize`] so the engine's wire-decode hot
/// path always finds an entry without a per-role conditional.
pub struct RoleDispatcher {
    pub(crate) dispatch: ProtocolDispatchFn,
    pub(crate) materialize: BackendMaterializeFn,
}

/// Build a `RoleDispatcher` for a concrete `ProtocolRuntime` impl.
/// Called from `Engine::register_protocol_dispatcher` and from any
/// test/production setup that needs to register dispatcher entries on
/// a fresh Engine.
pub fn make_protocol_dispatcher<T: ProtocolRuntime + 'static>() -> RoleDispatcher
where
    T::Error: std::fmt::Display,
{
    RoleDispatcher {
        dispatch: |any: &mut dyn std::any::Any,
                   op_type: &str,
                   inputs: &[(&str, &dyn SlotValue)],
                   ctx: &mut RuntimeResourceRef<'_>| {
            let concrete = any.downcast_mut::<T>().expect("is_match guaranteed");
            concrete
                .dispatch_atomic(op_type, inputs, ctx)
                .map_err(|e| e.to_string())
        },
        materialize: no_materialize,
    }
}

/// Macro: emit `make_<role>_dispatcher` factories for every
/// non-Protocol role trait. Each factory captures `T`'s concrete
/// `dispatch_atomic` (a role-specific trait method) inside a closure
/// the universal-pair-only dispatch table can call. Authors deriving
/// a single role (e.g. `#[derive(bb::Index)]`) get atomic dispatch
/// wired without needing a manual `ProtocolRuntime` shell.
macro_rules! emit_role_dispatcher_factory {
    ($factory_name:ident, $runtime_trait:path) => {
        #[doc = concat!("Build a `RoleDispatcher` for a concrete impl of ", stringify!($runtime_trait), ". Used by `Engine::register_*_dispatcher` chain (`Node::with_<role>(&value)`) so single-role components dispatch through the same `TypeId`-keyed registry as multi-role / Protocol-bearing components.")]
        pub fn $factory_name<T: $runtime_trait + 'static>() -> RoleDispatcher
        where
            <T as $runtime_trait>::Error: std::fmt::Display,
        {
            RoleDispatcher {
                dispatch: |any: &mut dyn std::any::Any,
                           op_type: &str,
                           inputs: &[(&str, &dyn SlotValue)],
                           ctx: &mut RuntimeResourceRef<'_>| {
                    let concrete = any.downcast_mut::<T>().expect("is_match guaranteed");
                    <T as $runtime_trait>::dispatch_atomic(concrete, op_type, inputs, ctx)
                        .map_err(|e| e.to_string())
                },
                materialize: no_materialize,
            }
        }
    };
}

emit_role_dispatcher_factory!(make_index_dispatcher, crate::roles::IndexRuntime);
emit_role_dispatcher_factory!(make_aggregator_dispatcher, crate::roles::AggregatorRuntime);
emit_role_dispatcher_factory!(make_model_dispatcher, crate::roles::ModelRuntime);
emit_role_dispatcher_factory!(make_codec_dispatcher, crate::roles::CodecRuntime);
emit_role_dispatcher_factory!(make_data_source_dispatcher, crate::roles::DataSourceRuntime);
emit_role_dispatcher_factory!(
    make_peer_selector_dispatcher,
    crate::roles::PeerSelectorRuntime
);

/// Build a `RoleDispatcher` for a concrete `BackendRuntime` impl —
/// the per-`T` `dispatch_atomic` closure plus the `materialize_from_wire`
/// bridge the derive emits. Backend dispatchers are the only ones
/// that wire a real `materialize` entry; every other role registers
/// [`no_materialize`].
pub fn make_backend_dispatcher<T: crate::roles::BackendRuntime + 'static>() -> RoleDispatcher
where
    <T as crate::roles::BackendRuntime>::Error: std::fmt::Display,
{
    RoleDispatcher {
        dispatch: |any: &mut dyn std::any::Any,
                   op_type: &str,
                   inputs: &[(&str, &dyn SlotValue)],
                   ctx: &mut RuntimeResourceRef<'_>| {
            let concrete = any.downcast_mut::<T>().expect("is_match guaranteed");
            <T as crate::roles::BackendRuntime>::dispatch_atomic(concrete, op_type, inputs, ctx)
                .map_err(|e| e.to_string())
        },
        materialize: |any: &mut dyn std::any::Any, type_hash: u64, bytes: Vec<u8>| {
            let concrete = any.downcast_ref::<T>().expect("is_match guaranteed");
            <T as crate::roles::BackendRuntime>::materialize_from_wire(concrete, type_hash, bytes)
        },
    }
}

#[cfg(test)]
#[path = "invoke_function_call_tests.rs"]
mod function_call_tests;
