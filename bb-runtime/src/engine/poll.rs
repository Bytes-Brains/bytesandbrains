//! 8-phase poll cycle + `handle_completion` per `docs/ENGINE.md`
//! §7 + §9.
//!
//! /10: wires `invoke_one` into the
//! canonical 8-phase cycle. Phases 1, 2, 5, 6, 8 are active at
//! ; Phases 3, 4, 7 are no-op pass-throughs (filled in by
use crate::engine::core::Engine;
use crate::engine::step::EngineStep;
use crate::framework::scheduler::TimerKind;
use crate::ids::{CommandId, NodeSiteId};
use crate::ingress::IngressEvent;
use crate::slot_value::SlotValue;
use crate::syscall::values::{BytesValue, WireReqIdValue};

impl Engine {
    /// Handle a CommandId completion per ENGINE.md §9.2.
    /// Writes the values into the suspended Op's output sites +
    /// pushes ready downstream consumers onto the frontier.
    pub fn handle_completion(
        &mut self,
        cmd_id: CommandId,
        values: Vec<(String, Box<dyn SlotValue>)>,
    ) -> Vec<EngineStep> {
        let Some(pending) = self.exec.pending_async.remove(&cmd_id) else {
            // No matching suspension - completion arrived for a
            // CommandId the engine doesn't know. Silently drop;
            return Vec::new();
        };

        let mut steps = Vec::new();

        // Move output_sites out of pending (pending is owned via
        // pending_async.remove above). Helpers take &[NodeSiteId]
        // borrows; the final step consumes by value.
        let sites: Vec<NodeSiteId> = pending.output_sites;
        for (i, (_name, value)) in values.into_iter().enumerate() {
            if let Some(site) = sites.get(i).copied() {
                self.exec
                    .slot_table
                    .insert((site, pending.exec_id), Some(value));
            }
        }

        // Push ready downstream consumers.
        self.push_ready_consumers(&sites, pending.exec_id);

        // Function-call splice: async completion arriving inside a
        // body's derived ExecId forwards to the caller's slots per
        // ENGINE.md §8.4. No-op when there's no pending call context.
        self.forward_outputs_to_caller(&sites, pending.exec_id);

        // Surface top-level function outputs (no in-function consumer)
        // as AppEvents - same path as `Engine::write_outputs`, so
        // async-completion writes participate in the canonical
        // function-signature → engine I/O contract.
        self.surface_top_level_outputs(&sites, pending.exec_id);

        steps.push(EngineStep::OpCompleted {
            op_ref: pending.op_ref,
            exec_id: pending.exec_id,
            sites_written: sites,
        });
        steps
    }

    /// Handle a transport-reported failure for a suspended
    /// `CommandId`. The Op that was waiting on `cmd_id` fails
    /// through the existing `OpFailed` path (bus
    /// `InfraEvent::OpFailure` + `EngineStep::OpFailed`). Use this
    /// when the host's transport adapter learns that the remote
    /// side failed to produce a result - the framework no longer
    /// silently swallows the outcome.
    pub fn handle_completion_failed(
        &mut self,
        cmd_id: CommandId,
        error: crate::bus::OpError,
    ) -> Vec<EngineStep> {
        let Some(pending) = self.exec.pending_async.remove(&cmd_id) else {
            // No matching suspension - failure arrived for a
            // CommandId the engine doesn't know. Silently drop;
            // the host's transport reconciliation should have
            // caught this earlier.
            return Vec::new();
        };
        vec![self.fail_op(
            pending.op_ref,
            pending.exec_id,
            crate::bus::OpErrorKind::RemoteFailed,
            "completion_failed",
            error.detail,
        )]
    }

    /// Expire any pending async suspensions whose `deadline_ns` is
    /// past `scheduler.now_ns()`. Each expired suspension fails via
    /// the existing `OpFailed` surface with
    /// `OpError("deadline exceeded")`. Returns the resulting steps.
    /// Called from Phase 5 of the poll cycle before draining
    /// `pending_completions`, so deadline failures land in the
    /// same poll where they expire.
    fn expire_deadlines(&mut self) -> Vec<EngineStep> {
        let now_ns = self.framework.scheduler.now_ns();
        let expired: Vec<CommandId> = self
            .exec
            .pending_async
            .iter()
            .filter_map(|(cmd, p)| match p.deadline_ns {
                Some(d) if d <= now_ns => Some(*cmd),
                _ => None,
            })
            .collect();
        let mut steps = Vec::new();
        for cmd in expired {
            if let Some(p) = self.exec.pending_async.remove(&cmd) {
                steps.push(self.fail_op(
                    p.op_ref,
                    p.exec_id,
                    crate::bus::OpErrorKind::Timeout,
                    "deadline_exceeded",
                    "deadline exceeded".to_string(),
                ));
            }
        }

        // Drain stale in-flight wire requests. Each evicted entry
        // surfaces as `EngineStep::WireTimeout` for observability;
        // if it carried a `parked_op`, fail the originator's local
        // continuation with "chain timeout" so it doesn't sit
        // parked forever.
        let drained = self.framework.request_tracker.drain_stale(now_ns);
        for (wire_req_id, entry) in drained {
            steps.push(EngineStep::WireTimeout {
                wire_req_id,
                target_site: entry.target_site,
                started_at_ns: entry.started_at_ns,
                parked_op: entry.parked_op,
            });
            if let Some(cmd) = entry.parked_op {
                if let Some(p) = self.exec.pending_async.remove(&cmd) {
                    steps.push(self.fail_op(
                        p.op_ref,
                        p.exec_id,
                        crate::bus::OpErrorKind::Timeout,
                        "chain_timeout",
                        "chain timeout".to_string(),
                    ));
                }
            }
        }
        steps
    }

    /// 8-phase poll cycle per ENGINE.md §7 +
    /// `docs/internal/IMPLEMENTATION_PLAN.md` .
    pub fn poll(&mut self) -> Vec<EngineStep> {
        let _poll_span = tracing::debug_span!("engine.poll").entered();
        // GC executions that finished in the previous cycle. The
        // one-cycle delay lets the host read completion state via
        // `slot_at` between polls; production consumers read via the
        // `EngineStep` stream so the delay is invisible to them.
        self.gc_completed_executions();
        let mut steps = Vec::new();
        // Per-poll counter used by `cycle_op_budget` enforcement.
        // Increments once per `invoke_one` call across Phases 2, 6,
        // and 7. When the budget is hit, the current drain loop
        // breaks and `CycleBudgetExceeded` is appended once.
        let mut ops_invoked: usize = 0;
        let mut budget_exceeded = false;

        // Drive any host-supplied bootstrap requests staged between
        // polls. The host kicks the install-order queue via
        // `Node::run_bootstrap` (which seeds before this poll runs)
        // and stages targets-with-inputs via `enqueue_bootstrap_request`;
        // both paths land here so disjoint targets fire alongside the
        // already-seeded phase and overlapping ones park on
        // `bootstrap.waiting` until `maybe_complete_bootstrap`
        // promotes them. Install no longer auto-seeds — the host
        // owns when bootstrap starts.
        self.drive_pending_bootstrap_requests();
        let bootstrap_was_pending = self.bootstrap.pending;
        // Per-phase BootstrapComplete steps accumulate here as each
        // queued key drains. The final `WaitingOnBootstrap` /
        // terminal `BootstrapComplete` decision below uses
        // `bootstrap_was_pending` + post-drain `bootstrap_pending` to
        // detect a partial drain (queue not yet empty, async pending).
        let mut bootstrap_phases_completed: usize = 0;

        // --- Phase 1 - drain ingress -----------------------------
        // While bootstrap is pending the ingress drain consumes only
        // events that can advance the bootstrap call (its async
        // completions, transport failures). Body-side events
        // (AppEvent, EnvelopeFrom, Invoke) requeue so the host
        // observes the same pre-bootstrap delivery order on the
        // cycle after BootstrapComplete fires. The loop re-drains
        // when bootstrap completes mid-pass so body events queued
        // before bootstrap finished still process in this cycle.
        //
        // Backpressure detection per
        // `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
        // §6(a): the pre-drain depth is the receiver's signal for
        // "we are over the high-water mark"; snapshotting it here
        // lets the per-envelope handler attribute overload to each
        // contributing sender without re-reading the queue (which
        // would already be drained to zero).
        self.phase1_pre_drain_depth = self.ingress.len();
        {
            let _phase1 = tracing::debug_span!("engine.phase1_ingress").entered();
            loop {
                let was_pending = self.bootstrap.pending;
                let ingress_events = self.ingress.drain_all();
                if ingress_events.is_empty() {
                    break;
                }
                for event in ingress_events {
                    if self.bootstrap.pending && self.is_body_phase_ingress(&event) {
                        let _ = self.ingress.push(event);
                        continue;
                    }
                    steps.extend(self.process_ingress_event(event));
                    if self.maybe_complete_bootstrap() {
                        bootstrap_phases_completed += 1;
                        self.seed_bootstrap_call();
                    }
                }
                if !was_pending || self.bootstrap.pending {
                    break;
                }
            }
        }

        // --- Phase 2 - drain frontier (initial pass) -------------
        {
            let _phase2 = tracing::debug_span!("engine.phase2_frontier_drain").entered();
            loop {
                while let Some((op_ref, exec_id)) = self.pop_frontier_fireable() {
                    let step = self.invoke_one(op_ref, exec_id);
                    steps.push(step);
                    ops_invoked += 1;
                    if budget_hit(self.cycle_op_budget, ops_invoked) {
                        budget_exceeded = true;
                        break;
                    }
                }
                // If a queued bootstrap phase just drained, re-seed
                // the next one and cascade in-cycle so the host sees
                // every BootstrapComplete + the body's first ops in a
                // single poll when budget permits.
                if self.maybe_complete_bootstrap() {
                    bootstrap_phases_completed += 1;
                    if budget_exceeded {
                        break;
                    }
                    if !self.seed_bootstrap_call() {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // --- Phase 3 - route bus events to subscribed sites ------
        // Before draining, surface any FIFO drops accumulated since
        // the last poll as an `InfraEvent::BusOverflow`. Publishing
        // before drain keeps the event in this cycle's routing pass.
        let bus_dropped = self.bus.take_dropped_count();
        if bus_dropped > 0 {
            self.bus.publish(crate::bus::NodeEvent::Infra(
                crate::bus::InfraEvent::BusOverflow { count: bus_dropped },
            ));
        }
        // For every NodeEvent on the bus, derive its `kind` string
        // (via NodeEvent::kind()) and look up the subscribed
        // `NodeSiteId`s. For each site, write a `TriggerValue` at a
        // fresh `ExecId` and push the site's downstream consumers
        // onto the frontier. This matches the wire delivery
        // semantics per `docs/ADDRESSING.md`.
        let events = self.bus.drain();
        if !events.is_empty() {
            let mut to_seed: Vec<crate::ids::NodeSiteId> = Vec::new();
            for event in events {
                let kind = event.kind();
                let Some(sites) = self.event_subscriptions.get(kind) else {
                    continue;
                };
                to_seed.extend(sites.iter().copied());
            }
            for site in to_seed {
                let exec_id = self.allocate_exec_id();
                let value: Box<dyn crate::slot_value::SlotValue> =
                    Box::new(crate::syscall::values::TriggerValue);
                self.exec.slot_table.insert((site, exec_id), Some(value));
                let consumers: Vec<crate::ids::OpRef> = self
                    .graphs_iter()
                    .filter_map(|g| g.consumers.get(&site).cloned())
                    .flatten()
                    .collect();
                for op_ref in consumers {
                    self.exec.frontier.push_back((op_ref, exec_id));
                }
            }
        }

        // --- Phase 4 - poll matured timers -----------------------
        let now_ns = self.framework.scheduler.now_ns();
        let matured = self.framework.scheduler.poll_matured(now_ns);
        for kind in matured {
            self.handle_matured_timer(kind);
        }

        // --- Phase 5 - expire deadlines + drain pending_completions -
        // Engine-side deadline scan runs first so an expired
        // suspension fails this cycle even if a (now-stale)
        // completion is also queued.
        {
            let _phase5 = tracing::debug_span!("engine.phase5_completions").entered();
            steps.extend(self.expire_deadlines());
            let completions = std::mem::take(&mut self.exec.pending_completions);
            for c in completions {
                steps.extend(self.handle_completion(c.cmd_id, c.results));
            }
            if self.maybe_complete_bootstrap() {
                bootstrap_phases_completed += 1;
                self.seed_bootstrap_call();
            }
        }

        // --- Phase 5b - φ-accrual liveness scan ------------------
        // tracker entry and publish bus events on state changes.
        {
            let now_ns = self.framework.scheduler.now_ns();
            let transitions = self.framework.rtt_tracker.scan_phi(now_ns);
            for transition in transitions {
                let event = match transition {
                    crate::framework::rtt_tracker::PhiTransition::Suspect { site, phi } => {
                        crate::bus::InfraEvent::PeerSuspect { site, phi }
                    }
                    crate::framework::rtt_tracker::PhiTransition::Down { site, phi } => {
                        crate::bus::InfraEvent::PeerDown { site, phi }
                    }
                    crate::framework::rtt_tracker::PhiTransition::Live { site } => {
                        crate::bus::InfraEvent::PeerLive { site }
                    }
                };
                self.bus.publish(crate::bus::NodeEvent::Infra(event));
            }
        }

        // --- Phase 6 - final frontier drain (cascades) -----------
        if !budget_exceeded {
            loop {
                while let Some((op_ref, exec_id)) = self.pop_frontier_fireable() {
                    let step = self.invoke_one(op_ref, exec_id);
                    steps.push(step);
                    ops_invoked += 1;
                    if budget_hit(self.cycle_op_budget, ops_invoked) {
                        budget_exceeded = true;
                        break;
                    }
                }
                if self.maybe_complete_bootstrap() {
                    bootstrap_phases_completed += 1;
                    if budget_exceeded {
                        break;
                    }
                    if !self.seed_bootstrap_call() {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // --- Phase 7 - fire_lifecycle ----------------------------
        // For each phase queued by the host via
        // `Engine::fire_lifecycle(phase)`, push every enrolled
        // `LifecyclePhase` op onto the frontier with a fresh ExecId
        // and emit a `LifecycleFired` step. Cascade-drain so newly
        // pushed ops invoke in this same poll cycle.
        let fired: Vec<String> = std::mem::take(&mut self.fired_phases);
        for phase in &fired {
            let op_refs: Vec<crate::ids::OpRef> =
                self.lifecycle_table.get(phase).cloned().unwrap_or_default();
            let pairs: Vec<(crate::ids::OpRef, crate::ids::ExecId)> = op_refs
                .into_iter()
                .map(|op_ref| (op_ref, self.allocate_exec_id()))
                .collect();
            for (op_ref, exec_id) in pairs {
                self.exec.frontier.push_back((op_ref, exec_id));
            }
            steps.push(EngineStep::LifecycleFired {
                phase: phase.clone(),
            });
        }
        if !budget_exceeded {
            loop {
                while let Some((op_ref, exec_id)) = self.pop_frontier_fireable() {
                    let step = self.invoke_one(op_ref, exec_id);
                    steps.push(step);
                    ops_invoked += 1;
                    if budget_hit(self.cycle_op_budget, ops_invoked) {
                        budget_exceeded = true;
                        break;
                    }
                }
                if self.maybe_complete_bootstrap() {
                    bootstrap_phases_completed += 1;
                    if budget_exceeded {
                        break;
                    }
                    if !self.seed_bootstrap_call() {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // --- Phase 8 - drain outbound queue + pending app events -
        let _phase8 = tracing::debug_span!("engine.phase8_outbound").entered();
        for env in self.framework.outbound_queue.drain_all() {
            steps.push(EngineStep::SendEnvelope(env));
        }
        // Surface peer-resolution failures captured by the wire
        // syscall during this poll. Each entry becomes a dedicated
        // EngineStep::PeerResolveFailed; the matching bus event was
        // already published by the syscall, so subscribers got the
        // routable mirror in real time.
        for (peer, op_ref) in self.framework.pending_peer_resolve_failures.drain(..) {
            steps.push(EngineStep::PeerResolveFailed {
                peer,
                op_ref,
                exec_id: crate::ids::ExecId::from(0u64),
            });
        }
        // Emit a single `OutboundDropped` step capturing FIFO drops
        // that accumulated since the previous poll (e.g., when push
        // exceeded `max_outbound_queue`).
        let dropped = self.framework.outbound_queue.take_dropped_count();
        if dropped > 0 {
            steps.push(EngineStep::OutboundDropped { count: dropped });
        }
        // Surface a single `CycleBudgetExceeded` at the end so the
        // host knows to re-poll. The step is appended after all
        // observable per-op steps so the budget signal trails the
        // in-cycle work it bounded.
        if budget_exceeded {
            steps.push(EngineStep::CycleBudgetExceeded { ops_invoked });
        }
        // Bootstrap state observable. Each queued bootstrap key that
        // drained during this cycle emits its own `BootstrapComplete`
        // — multi-target installs surface one signal per target in
        // install order. `WaitingOnBootstrap` lands when the
        // *currently* in-flight bootstrap op suspended on async
        // completion and the host must drive the resumption before
        // re-polling. Body-phase ops are gated from firing while
        // `bootstrap_pending` is set.
        if bootstrap_was_pending {
            for _ in 0..bootstrap_phases_completed {
                steps.push(EngineStep::BootstrapComplete);
            }
            if self.bootstrap.pending {
                steps.push(EngineStep::WaitingOnBootstrap);
            }
        }
        // Drain Sub-C syscall outputs (AppEmit / AppNotify) into
        // EngineStep::AppEvent observable by the host. `Emit` carries
        // serialized payload bytes; `Notify` is a marker-only event
        // (empty `value_bytes`).
        for ev in std::mem::take(&mut self.framework.pending_app_events) {
            let (module_name, topic, value_bytes) = match ev {
                crate::bus::AppEvent::Emit { name, value_bytes } => {
                    (String::new(), name, value_bytes)
                }
                crate::bus::AppEvent::Notify { name } => (String::new(), name, Vec::new()),
            };
            steps.push(EngineStep::AppEvent {
                module_name,
                topic,
                value_bytes,
            });
        }

        steps
    }

    /// Whether an ingress event would seed body-phase work. The
    /// bootstrap gate requeues these while the bootstrap call is
    /// outstanding so app-event delivery + envelope routing
    /// observe the post-bootstrap engine state. Bootstrap-resuming
    /// completions, transport failures, and host-injected timer
    /// matures bypass the gate so the bootstrap call can progress.
    fn is_body_phase_ingress(&self, event: &IngressEvent) -> bool {
        matches!(
            event,
            IngressEvent::AppEvent { .. }
                | IngressEvent::EnvelopeFrom { .. }
                | IngressEvent::Invoke { .. }
        )
    }

    /// Ingress-event router. Dispatches each event variant to its
    /// handler: envelopes route to `deliver_inbound_internal`,
    /// completions to `handle_completion`, app events to the bus,
    /// matured timers to `handle_matured_timer`, invoke events to
    /// the per-module entry point.
    fn process_ingress_event(&mut self, event: IngressEvent) -> Vec<EngineStep> {
        match event {
            IngressEvent::Completion { cmd_id, results } => {
                // The host's transport pre-decodes opaque payloads
                // and hands them as `Vec<Vec<u8>>`. The engine wraps
                // each entry as a `BytesValue` and forwards to
                // `handle_completion`, which writes the slots and
                // pushes downstream consumers.
                let typed_results: Vec<(String, Box<dyn crate::slot_value::SlotValue>)> = results
                    .into_iter()
                    .enumerate()
                    .map(|(i, bytes)| {
                        let value = crate::syscall::values::BytesValue(bytes);
                        (
                            format!("out_{i}"),
                            Box::new(value) as Box<dyn crate::slot_value::SlotValue>,
                        )
                    })
                    .collect();
                self.handle_completion(cmd_id, typed_results)
            }
            IngressEvent::EnvelopeFrom {
                src_peer,
                envelope,
                src_observed_address,
            } => {
                // Backpressure protocol pre-flight per
                // `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
                // §3 + §6. The receiver:
                //   1. Drops the envelope without dispatch when the
                //      sender is in silent-drop mode (the K-then-silent
                //      fallback).
                //   2. Otherwise routes the envelope normally, then
                //      checks the post-pop ingress depth against the
                //      configured high-water mark and emits one
                //      `BackoffNotice` to the sender when the mark is
                //      crossed.
                if self
                    .framework
                    .peer_state
                    .backpressure
                    .is_silent_drop_active(src_peer)
                {
                    return Vec::new();
                }
                // Order: claimed (envelope) first so the entry
                // exists for the observed-address registration step.
                // Observed wins for NAT-translated cases because it
                // appends a fresh address the claimed snapshot
                // cannot know.
                //
                // The address-book hint is best-effort under allocator
                // pressure (spec §2.1 S4 + S5): if the dedup buffer
                // cannot be reserved we drop the hint, emit
                // `WireReceiveError::AllocationFailed`, and continue
                // routing — the envelope's fills do not depend on the
                // address book.
                let mut steps = Vec::new();
                if let Err(alloc) =
                    self.merge_src_peer_addresses(src_peer, &envelope.src_peer_addresses)
                {
                    steps.push(self.emit_wire_receive_error(
                        Some(src_peer),
                        0,
                        0,
                        alloc.byte_count,
                        crate::bus::WireReceiveErrorKind::AllocationFailed {
                            byte_count: alloc.byte_count,
                            reason: alloc.reason,
                        },
                    ));
                }
                if let Some(observed) = src_observed_address {
                    self.merge_src_observed_address(src_peer, observed);
                }
                steps.extend(self.route_envelope(envelope, Some(src_peer)));
                let backoff_steps = self.maybe_emit_backoff_notice(
                    src_peer,
                    crate::framework::BackoffCause::QueueFull,
                    None,
                );
                steps.extend(backoff_steps);
                steps
            }
            IngressEvent::AppEvent {
                module_name,
                input_name,
                value_bytes,
            } => self.deliver_app_event(module_name, input_name, value_bytes),
            IngressEvent::Invoke {
                module_name,
                inputs,
                exec_id,
            } => self.deliver_invoke(module_name, inputs, exec_id),
            IngressEvent::TimerMatured { at_ns } => {
                self.framework.scheduler.set_now(at_ns);
                let matured = self.framework.scheduler.poll_matured(at_ns);
                let mut out = Vec::new();
                for kind in matured {
                    out.extend(self.handle_matured_timer(kind));
                }
                out
            }
            IngressEvent::CompletionFailed { cmd_id, detail } => {
                // async completion FAILURE; routes
                // directly to `handle_completion_failed` (already
                // at this file's lines 67-80) which fails the
                // parked op through the typed `OpFailed` path,
                // NOT through the success-bytes masquerade the
                // legacy `CompletionSink::fail` was forced to use.
                self.handle_completion_failed(
                    cmd_id,
                    crate::bus::OpError {
                        detail,
                        ..Default::default()
                    },
                )
            }
            IngressEvent::SendFailed {
                wire_req_id,
                peer: _peer,
                reason: _reason,
            } => {
                // transport-side delivery failure.
                // Consumes the in-flight registration so the
                // request tracker doesn't leak the entry. The
                // parked originator op's failure routing is the
                // wire-timeout drain's job (`drain_stale`); this
                // variant exists so the request tracker observes
                // an explicit transport failure rather than waiting
                // for the TTL to elapse. Wider parked-op routing
                // (peeking the original `InFlightSend.parked_op`
                // through a typed accessor) is a follow-up
                // alongside the snapshot fidelity work in Phase 11.
                let now_ns = self.framework.scheduler.now_ns();
                let _ = self
                    .framework
                    .request_tracker
                    .observe_response(wire_req_id, now_ns);
                Vec::new()
            }
            IngressEvent::AppIngressError {
                source,
                byte_count,
                kind,
            } => {
                // Cross-thread bridge for `CompletionSink::complete`
                // exceeding the per-completion result cap. Re-publish
                // on the bus so subscribers see the rejection
                // alongside the synchronous emissions from
                // `Node::deliver_event` / `Node::invoke`.
                self.bus.publish(crate::bus::NodeEvent::Infra(
                    crate::bus::InfraEvent::AppIngressError {
                        source,
                        byte_count,
                        kind,
                    },
                ));
                Vec::new()
            }
        }
    }

    /// Deliver an inbound `AppEvent`: look up the addressed graph,
    /// resolve the `input_name` to a `NodeSiteId`, wrap the bytes as
    /// a `BytesValue`, seed the slot at a fresh `ExecId`, and push
    /// ready downstream consumers onto the frontier. Surfaces an
    /// observable `EngineStep::AppEvent` so the host can confirm
    /// delivery even if no consumer exists yet.
    fn deliver_app_event(
        &mut self,
        module_name: String,
        input_name: String,
        value_bytes: Vec<u8>,
    ) -> Vec<EngineStep> {
        let step = EngineStep::AppEvent {
            module_name: module_name.clone(),
            topic: input_name.clone(),
            value_bytes: value_bytes.clone(),
        };
        let Some(graph) = self.graph(&module_name) else {
            return vec![step];
        };
        let Some(&site) = graph.site_names.get(&input_name) else {
            return vec![step];
        };
        let exec_id = self.allocate_exec_id();
        let value = crate::syscall::values::BytesValue(value_bytes);
        self.exec
            .slot_table
            .insert((site, exec_id), Some(Box::new(value)));
        self.push_ready_consumers(&[site], exec_id);
        vec![step]
    }

    /// Deliver an inbound `Invoke`: seed every `(input_name,
    /// value_bytes)` pair into the addressed graph's matching site
    /// at the supplied `exec_id`, then push the ready consumers
    /// onto the frontier. Unknown modules / unknown input names are
    /// silent no-ops (the host can detect via subsequent polls
    /// producing no steps).
    fn deliver_invoke(
        &mut self,
        module_name: String,
        inputs: Vec<(String, Vec<u8>)>,
        exec_id: crate::ids::ExecId,
    ) -> Vec<EngineStep> {
        let Some(graph) = self.graph(&module_name) else {
            return Vec::new();
        };
        let mut seeded_sites: Vec<crate::ids::NodeSiteId> = Vec::new();
        let pairs: Vec<(crate::ids::NodeSiteId, Vec<u8>)> = inputs
            .into_iter()
            .filter_map(|(name, bytes)| graph.site_names.get(&name).map(|&site| (site, bytes)))
            .collect();
        for (site, bytes) in pairs {
            let value = crate::syscall::values::BytesValue(bytes);
            self.exec
                .slot_table
                .insert((site, exec_id), Some(Box::new(value)));
            seeded_sites.push(site);
        }
        if !seeded_sites.is_empty() {
            self.push_ready_consumers(&seeded_sites, exec_id);
        }
        Vec::new()
    }

    /// Route a single inbound envelope. Iterates each `SlotFill` and
    /// dispatches it via its multiaddr `dest_suffix` per
    /// `docs/ADDRESSING.md`. The receiver doesn't consult any
    /// subscription table or routing map - the address suffix is the
    /// routing key. Two suffix shapes are supported:
    ///   - `/site/<NodeSiteId>` → data-plane slot fill.
    ///   - `/component/<cref>/op/<name>` → control-plane component
    ///     dispatch.
    fn route_envelope(
        &mut self,
        env: crate::envelope::WireEnvelope,
        src_peer: Option<crate::ids::PeerId>,
    ) -> Vec<EngineStep> {
        // Sender-side back-pressure ingest per
        // `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
        // §5. Inbound envelopes whose first fill carries the reserved
        // `BackoffNoticePayload` type-hash are framework-internal -
        // intercept them here, decode the payload, advise the
        // sender-side `BackoffTable`, and short-circuit the normal
        // data-plane / control-plane delivery so user Components
        // never observe a notice envelope.
        if env
            .fills
            .first()
            .is_some_and(|f| f.type_hash == crate::framework::backoff_notice_type_hash())
        {
            return self.ingest_backoff_notice(env, src_peer);
        }

        let correlation = env.correlation.as_ref().map(|c| c.wire_req_id).unwrap_or(0);

        // hook. If the inbound envelope's wire_req_id matches an
        // in-flight outbound round trip we registered at dispatch,
        // pop the sample + feed it into the RTT tracker so the
        // hierarchical-fallback EMAs converge on real per-edge,
        // per-site, per-chain, and global RTT distributions.
        let mut response_from_site: Option<crate::ids::NodeSiteId> = None;
        if correlation != 0 {
            let now_ns = self.framework.scheduler.now_ns();
            if let Some(sample) = self
                .framework
                .request_tracker
                .observe_response(correlation, now_ns)
            {
                self.framework.rtt_tracker.observe_round_trip(
                    sample.target_site,
                    sample.chain,
                    sample.elapsed_ns,
                    now_ns,
                );
                response_from_site = Some(sample.target_site);
            }
        }

        // piggyback. The sender attached EdgeRttReport entries
        // describing its observed outgoing edges in the chain. We
        // record each report against the entry for the sending
        // site so multi-hop chain budgets can compose from this
        // direct neighbor's table.
        if let Some(from_site) = response_from_site {
            for report in &env.edge_rtt_reports {
                self.framework.rtt_tracker.ingest_reported_outgoing(
                    from_site,
                    crate::ids::NodeSiteId::from(report.next_hop_site_id),
                    report.chain_id,
                    report.srtt_ns,
                    report.rttvar_ns,
                    report.sample_count,
                );
            }
        }

        // Capture the inbound envelope's deadline budget + arrival
        // timestamp so consumer ops (especially `wire.Send` while
        // forwarding) can propagate them per Dapper.
        let inbound_remaining_deadline_ns = if env.remaining_deadline_ns > 0 {
            Some(env.remaining_deadline_ns)
        } else {
            None
        };
        let arrival_ns = self.framework.scheduler.now_ns();

        let mut steps = Vec::new();
        for (fill_index, fill) in env.fills.into_iter().enumerate() {
            steps.extend(self.deliver_fill(
                fill,
                fill_index as u32,
                correlation,
                src_peer,
                arrival_ns,
                inbound_remaining_deadline_ns,
            ));
        }
        steps
    }

    /// Dispatch one `SlotFill` per `docs/ADDRESSING.md`. Parses
    /// `fill.dest_suffix` as a multiaddr and routes by the trailing
    /// segment shape. `fill_index` is the fill's 0-based position
    /// within the inbound envelope; surfaces on per-fill failure
    /// events so subscribers can identify the failing fill when
    /// the envelope partial-delivers.
    fn deliver_fill(
        &mut self,
        fill: crate::envelope::SlotFill,
        fill_index: u32,
        wire_req_id: u64,
        src_peer: Option<crate::ids::PeerId>,
        arrival_ns: u64,
        inbound_remaining_deadline_ns: Option<u64>,
    ) -> Vec<EngineStep> {
        let addr = match crate::framework::Address::from_bytes(&fill.dest_suffix) {
            Ok(a) => a,
            Err(e) => {
                return vec![self.emit_wire_decode_failure(
                    0,
                    fill.payload.len(),
                    format!("dest_suffix parse: {e}"),
                )];
            }
        };

        // Data-plane suffix: /site/<NodeSiteId>. The Site segment
        // uniquely identifies the slot (NodeSiteIds are globally
        // unique within a Node).
        if let Some(site_id) = addr.site_id() {
            return self.deliver_data_plane_fill(
                site_id,
                fill,
                fill_index,
                src_peer,
                wire_req_id,
                arrival_ns,
                inbound_remaining_deadline_ns,
            );
        }

        // Control-plane suffix: /component/<cref>/op/<name>.
        if let (Some(cref), Some(op_name)) = (addr.component_ref(), addr.op_name()) {
            let op_name = op_name.to_string();
            return self.deliver_control_plane_fill(cref, op_name, fill, wire_req_id);
        }

        vec![self.emit_wire_decode_failure(
            0,
            fill.payload.len(),
            "address shape neither data-plane nor control-plane".to_string(),
        )]
    }

    /// Publish a `WireDecodeFailure` onto the bus and return the
    /// matching `EngineStep`. The bus event lets in-process
    /// telemetry subscribers react; the EngineStep surfaces the
    /// same context to the host poll() caller.
    fn emit_wire_decode_failure(
        &mut self,
        hash: u64,
        payload_size: usize,
        detail: String,
    ) -> EngineStep {
        self.bus.publish(crate::bus::NodeEvent::Infra(
            crate::bus::InfraEvent::WireDecodeFailure {
                hash,
                payload_size,
                detail: detail.clone(),
            },
        ));
        EngineStep::WireDecodeFailed {
            hash,
            payload_size,
            detail,
        }
    }

    /// Data-plane delivery: decode the fill payload into a typed
    /// `SlotValue` via the wire decoder registry, write it into the
    /// addressed slot at a fresh `ExecId`, and push the slot's
    /// downstream consumers onto the frontier. Walks each installed
    /// graph's `consumers` map for the matching `NodeSiteId`.
    ///
    /// Per-fill failures (unknown type-hash, type mismatch against
    /// the slot's declared wire type, decoder error) surface as a
    /// `WireReceiveError` InfraEvent + matching `WireReceiveFailed`
    /// EngineStep. The failing fill drops; sibling fills in the
    /// same envelope still deliver (the caller continues iterating).
    #[allow(clippy::too_many_arguments)]
    fn deliver_data_plane_fill(
        &mut self,
        site_id: crate::ids::NodeSiteId,
        mut fill: crate::envelope::SlotFill,
        fill_index: u32,
        src_peer: Option<crate::ids::PeerId>,
        wire_req_id: u64,
        arrival_ns: u64,
        inbound_remaining_deadline_ns: Option<u64>,
    ) -> Vec<EngineStep> {
        // Resolve the consumer ops from each installed graph; a
        // NodeSiteId belongs to at most one graph because IDs are
        // globally unique, but we tolerate empty lookups.
        let consumers: Vec<crate::ids::OpRef> = self
            .graphs_iter()
            .filter_map(|g| g.consumers.get(&site_id).cloned())
            .flatten()
            .collect();

        // Resolve the typed `SlotValue` BEFORE allocating an
        // ExecId or stamping inbound context: failure modes return
        // a WireReceiveError step and the envelope's other fills
        // continue to deliver without polluting the slot table.
        // Trigger fills bypass the decoder lookup entirely.
        let value: Box<dyn crate::slot_value::SlotValue> = if fill.trigger_only {
            Box::new(crate::syscall::values::TriggerValue)
        } else {
            match self.decode_typed_fill(&mut fill, fill_index, site_id, src_peer) {
                Ok(v) => v,
                Err(step) => return vec![step],
            }
        };

        let exec_id = self.allocate_exec_id();
        // Stamp the inbound envelope context for this ExecId.
        // Components access this through `RuntimeResourceRef` (RX
        // gates filter on `src_peer`; `wire.Send` forwarding inside a
        // chain reuses `wire_req_id` + propagates `remaining_deadline_ns`
        // minus elapsed local time).
        self.framework.inbound_contexts.insert(
            exec_id,
            crate::framework::InboundContext {
                src_peer,
                wire_req_id: if wire_req_id != 0 {
                    Some(wire_req_id)
                } else {
                    None
                },
                arrival_ns: Some(arrival_ns),
                remaining_deadline_ns: inbound_remaining_deadline_ns,
            },
        );
        // `slot_write` routes through the engine's budget-release
        // bookkeeping so a wire-receive overwriting a prior carrier
        // releases the prior `charged_bytes()` against
        // `ingress_bytes_in_flight`. Fresh-slot writes (the common
        // case here, since `exec_id` is freshly allocated) hit the
        // no-prior branch and incur the same cost as the raw
        // `slot_table.insert`.
        self.slot_write(site_id, exec_id, value);

        // If site_id is a wire.Recv's payload site, also populate the
        // paired sender site with PeerIdValue(src_peer) for the same
        // ExecId. Downstream user ops read this as a graph value to
        // identify provenance; reply-to is `g.net_out(name, sender, reply)`.
        let sender_site: Option<crate::ids::NodeSiteId> = self
            .graphs_iter()
            .find_map(|g| g.recv_sender_sites.get(&site_id).copied());
        if let (Some(sender_site), Some(peer)) = (sender_site, src_peer) {
            let sender_value: Box<dyn crate::slot_value::SlotValue> =
                Box::new(crate::syscall::values::PeerIdValue(peer));
            self.exec
                .slot_table
                .insert((sender_site, exec_id), Some(sender_value));
        }

        for op_ref in consumers {
            self.exec.frontier.push_back((op_ref, exec_id));
        }
        Vec::new()
    }

    /// Resolve a non-trigger data-plane fill into its typed
    /// `SlotValue` carrier. The routing tree branches on the
    /// destination slot's binding:
    ///
    /// - **Backend-bound slot** — the engine takes ownership of
    ///   `fill.payload` via `std::mem::take` (zero-copy ownership
    ///   transfer; `fill.payload` is already framework-owned from
    ///   envelope decode) and hands it to the bound backend's
    ///   `materialize_from_wire`. The typed tensor lands inside a
    ///   `BackendTensorCarrier` whose `charged_bytes` + `backend_ref`
    ///   are stamped to the engine-side accounting before the carrier
    ///   is returned for slot-table install.
    /// - **Framework-carrier slot** — the wire decoder registry
    ///   resolves the `type_hash` to a bincode decoder; the decoded
    ///   `SlotValue` rides on as-is.
    ///
    /// Failure modes surface as typed `WireReceiveError` InfraEvents
    /// + matching `WireReceiveFailed` EngineSteps:
    ///
    /// - **TypeMismatch** — destination slot declares an expected
    ///   wire-type hash via `GraphSlot::recv_wire_type_hash` and
    ///   the fill's `type_hash` does not match. Checked first so a
    ///   mis-typed payload never reaches the decoder.
    /// - **UnknownTypeHash** — framework-carrier path only; no
    ///   decoder is registered for the stamped hash.
    /// - **DecodeFailed** — registered decoder ran and rejected
    ///   the bytes (framework-carrier path).
    /// - **BudgetExceeded** — admitting the bytes would push the
    ///   engine over `NodeConfig::ingress_byte_budget`.
    /// - **BackendMaterializeFailed** — the bound backend's typed
    ///   `materialize_from_wire` returned `Err` (backend path).
    ///
    /// The `Err` carries the typed `EngineStep` the caller will
    /// surface to `Engine::poll`'s return value. `EngineStep` is
    /// load-bearing for the host's failure visibility surface, so
    /// boxing it here would force every caller through an
    /// indirection layer that adds no clarity for a single
    /// internal call site.
    ///
    /// **No per-fill scratch buffer.** Backend-mediated tensor fills
    /// move `fill.payload` into `materialize_from_wire` via
    /// `std::mem::take`; the empty `Vec<u8>` left behind drops with
    /// the `SlotFill`. The only memcpy on the tensor path is the
    /// one the backend chooses to do — or skips via zero-copy
    /// adoption (`ArrayD::from_shape_vec` when alignment permits).
    #[allow(clippy::result_large_err)]
    fn decode_typed_fill(
        &mut self,
        fill: &mut crate::envelope::SlotFill,
        fill_index: u32,
        site_id: crate::ids::NodeSiteId,
        src_peer: Option<crate::ids::PeerId>,
    ) -> Result<Box<dyn crate::slot_value::SlotValue>, EngineStep> {
        let expected_hash: Option<u64> = self
            .graphs_iter()
            .find_map(|g| g.recv_wire_type_hash.get(&site_id).copied());
        if let Some(expected) = expected_hash {
            if expected != fill.type_hash {
                return Err(self.emit_wire_receive_error(
                    src_peer,
                    fill_index,
                    fill.type_hash,
                    fill.payload.len(),
                    crate::bus::WireReceiveErrorKind::TypeMismatch {
                        expected_hash: expected,
                    },
                ));
            }
        }

        // Resolve the destination slot's binding so the backend-bound
        // branch can fire before the framework-carrier registry
        // lookup. A site without a recv_site → slot_id mapping (no
        // role consumes the Recv payload) takes the framework path.
        let slot_id = self
            .graphs_iter()
            .find_map(|g| g.recv_site_to_slot_id.get(&site_id).copied());
        let role_ref = slot_id.and_then(|id| self.role_ref_for_slot_id(id));

        // Budget guard (uniform across branches): pre-charge the
        // fill's payload length against `NodeConfig::ingress_byte_budget`
        // before invoking either decoder. Successful admission leaves
        // the charge in the counter; failure paths release before
        // returning.
        let byte_count = fill.payload.len();
        if let Err(reason) = self.try_charge(byte_count) {
            return Err(self.emit_wire_receive_error(
                src_peer,
                fill_index,
                fill.type_hash,
                byte_count,
                crate::bus::WireReceiveErrorKind::BudgetExceeded {
                    byte_count: reason.byte_count,
                    budget_remaining: reason.budget_remaining,
                },
            ));
        }

        if let Some((crate::registry::ComponentRole::Backend, backend_ref)) = role_ref {
            return self.materialize_via_backend(
                fill,
                fill_index,
                src_peer,
                byte_count,
                backend_ref,
            );
        }

        let Some(decoder) = crate::slot_value::wire_decoder_registry()
            .get(&fill.type_hash)
            .copied()
        else {
            self.release(byte_count);
            return Err(self.emit_wire_receive_error(
                src_peer,
                fill_index,
                fill.type_hash,
                fill.payload.len(),
                crate::bus::WireReceiveErrorKind::UnknownTypeHash,
            ));
        };
        decoder(&fill.payload).map_err(|e| {
            self.release(byte_count);
            self.emit_wire_receive_error(
                src_peer,
                fill_index,
                fill.type_hash,
                byte_count,
                crate::bus::WireReceiveErrorKind::DecodeFailed {
                    error_summary: e.to_string(),
                },
            )
        })
    }

    /// Backend-bound branch of [`Self::decode_typed_fill`]. Hands the
    /// inbound bytes to the bound backend via the role-dispatch
    /// registry; wraps the typed `Self::Tensor` in a
    /// `BackendTensorCarrier` whose engine-side accounting fields
    /// (`charged_bytes`, `backend_ref`) are stamped before the
    /// carrier is returned. On error releases the byte charge and
    /// emits `BackendMaterializeFailed`.
    #[allow(clippy::result_large_err)]
    fn materialize_via_backend(
        &mut self,
        fill: &mut crate::envelope::SlotFill,
        fill_index: u32,
        src_peer: Option<crate::ids::PeerId>,
        byte_count: usize,
        backend_ref: crate::ids::ComponentRef,
    ) -> Result<Box<dyn crate::slot_value::SlotValue>, EngineStep> {
        // `mem::take` transfers ownership of `fill.payload` to the
        // backend at zero cost: the wire bytes are already
        // framework-owned (prost allocated them during envelope
        // decode), and the empty `Vec<u8>` left in `fill.payload`
        // drops with the `SlotFill`. No scratch buffer, no memcpy on
        // the framework side.
        let bytes = std::mem::take(&mut fill.payload);
        let type_hash = fill.type_hash;

        // Take the backend component out of the Vec so dispatch can
        // borrow it without holding a long lease on `engine.components`.
        // Restore on the way out — even on error paths.
        let Some(mut taken) = self.take_component(backend_ref) else {
            self.release(byte_count);
            return Err(self.emit_wire_receive_error(
                src_peer,
                fill_index,
                type_hash,
                byte_count,
                crate::bus::WireReceiveErrorKind::BackendMaterializeFailed {
                    backend_ref,
                    backend_error_summary: "backend component slot empty".to_string(),
                },
            ));
        };

        // Reach the backend through the per-T dispatcher in
        // `role_dispatchers`. The dispatcher closes over the typed
        // `Self::Tensor` so the boxed `SlotValue` returned here is
        // already a `BackendTensorCarrier` (the derive bridge does
        // the wrap).
        let any: &mut dyn std::any::Any = taken.as_mut();
        let tid = (*any).type_id();
        let dispatcher = self.role_dispatchers.get(&tid).map(|d| d.materialize);

        let result = if let Some(materialize) = dispatcher {
            (materialize)(any, type_hash, bytes)
        } else {
            Err(crate::slot_value::BackendMaterializeError {
                summary: "no BackendRuntime dispatcher registered".to_string(),
            })
        };

        self.restore_component(backend_ref, taken);

        match result {
            Ok(boxed) => {
                // Downcast to `BackendTensorCarrier` and stamp the
                // engine-side accounting fields the derive bridge left
                // as placeholders. The bridge constructs the carrier
                // with the typed clone / encode fn pointers; the
                // engine owns the budget counter and the backend
                // identity, so it fills them in here. The downcast
                // is infallible by the bridge's construction; any
                // hand-rolled `BackendRuntime::materialize_from_wire`
                // that returns a non-carrier `SlotValue` flows
                // through unchanged (no accounting stamp), which is
                // the right behaviour because non-carrier returns
                // never charge against the backend-tensor pool.
                let any_box = boxed.into_any_boxed();
                let final_boxed: Box<dyn crate::slot_value::SlotValue> =
                    match any_box.downcast::<crate::slot_value::BackendTensorCarrier>() {
                        Ok(mut carrier) => {
                            carrier.charged_bytes = byte_count;
                            carrier.backend_ref = backend_ref;
                            carrier
                        }
                        Err(other) => {
                            // The dispatcher returned a non-carrier
                            // `SlotValue`; route it through unchanged.
                            // `Box<dyn Any + Send + Sync>` downcasts
                            // back to `Box<dyn SlotValue>` only via a
                            // typed re-box, which we don't do — log
                            // and release the budget charge instead.
                            let _ = other;
                            self.release(byte_count);
                            return Err(self.emit_wire_receive_error(
                                src_peer,
                                fill_index,
                                type_hash,
                                byte_count,
                                crate::bus::WireReceiveErrorKind::BackendMaterializeFailed {
                                    backend_ref,
                                    backend_error_summary:
                                        "backend bridge returned non-carrier SlotValue".to_string(),
                                },
                            ));
                        }
                    };
                Ok(final_boxed)
            }
            Err(e) => {
                self.release(byte_count);
                Err(self.emit_wire_receive_error(
                    src_peer,
                    fill_index,
                    type_hash,
                    byte_count,
                    crate::bus::WireReceiveErrorKind::BackendMaterializeFailed {
                        backend_ref,
                        backend_error_summary: e.summary,
                    },
                ))
            }
        }
    }

    /// Publish a `WireReceiveError` on the bus and return the
    /// matching `EngineStep`. Mirrors `emit_wire_decode_failure`
    /// for the per-fill typed-decode failure surface.
    fn emit_wire_receive_error(
        &mut self,
        src_peer: Option<crate::ids::PeerId>,
        fill_index: u32,
        actual_hash: u64,
        payload_size: usize,
        kind: crate::bus::WireReceiveErrorKind,
    ) -> EngineStep {
        self.bus.publish(crate::bus::NodeEvent::Infra(
            crate::bus::InfraEvent::WireReceiveError {
                src_peer,
                fill_index,
                actual_hash,
                payload_size,
                kind: kind.clone(),
            },
        ));
        EngineStep::WireReceiveFailed {
            src_peer,
            fill_index,
            actual_hash,
            payload_size,
            kind,
        }
    }

    /// Control-plane delivery: invoke
    /// `component[cref].dispatch_atomic(op_name, [(payload,
    /// correlation, ...)], ctx)`. The component decodes the payload
    /// bytes via its own protocol logic.
    fn deliver_control_plane_fill(
        &mut self,
        component_ref: crate::ids::ComponentRef,
        op_name: String,
        fill: crate::envelope::SlotFill,
        wire_req_id: u64,
    ) -> Vec<EngineStep> {
        let payload = BytesValue(fill.payload);
        let correlation = WireReqIdValue(wire_req_id);

        let inputs_storage: Vec<(String, Box<dyn SlotValue>)> = vec![
            ("payload".to_string(), Box::new(payload)),
            ("correlation".to_string(), Box::new(correlation)),
        ];

        let inputs_for_dispatch: Vec<(&str, &dyn SlotValue)> = inputs_storage
            .iter()
            .map(|(n, h)| (n.as_str(), h.as_ref()))
            .collect();

        // D2 take-and-restore so we can split-borrow the rest of
        // `self.framework` / `self.bus` etc. while the dispatching
        // component is held exclusively.
        let Some(mut taken) = self.take_component(component_ref) else {
            return Vec::new();
        };

        let mut ctx = crate::runtime::RuntimeResourceRef {
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
                pending_peer_resolve_failures: &mut self.framework.pending_peer_resolve_failures,
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
                op_ref: crate::ids::OpRef::from(0u64),
                exec_id: crate::ids::ExecId::from(0u64),
                self_peer: self.self_peer,
                node_attributes: &[],
                node_metadata: &[],
                inbound: crate::runtime::InboundCtx {
                    src_peer: None,
                    wire_req_id: None,
                    arrival_ns: None,
                    remaining_deadline_ns: None,
                },
                pending_completions: Vec::new(),
                next_command_id: &mut self.exec.ids.next_command_id,
            },
        };

        let _ = crate::engine::invoke::call_protocol_dispatch_atomic(
            taken.as_mut(),
            &op_name,
            &inputs_for_dispatch,
            &mut ctx,
            &self.role_dispatchers,
        );
        let captured = std::mem::take(&mut ctx.current.pending_completions);
        drop(ctx);
        self.exec.pending_completions.extend(captured);

        self.restore_component(component_ref, taken);

        Vec::new()
    }

    /// Route a matured timer to its consumer.
    /// - `Sleep`/`Completion` fulfil a pending `CommandId`.
    /// - `Interval` re-pushes its owning Op onto the frontier at a
    ///   fresh `ExecId`; the Op's `invoke` re-schedules the next
    ///   firing and emits the periodic `TriggerValue` downstream.
    /// - `After` fulfils the parked `CommandId` (which the Op
    ///   suspended on) with a single `TriggerValue`.
    fn handle_matured_timer(
        &mut self,
        kind: crate::framework::scheduler::TimerKind,
    ) -> Vec<EngineStep> {
        match kind {
            TimerKind::Sleep(cmd_id) | TimerKind::Completion(cmd_id) => {
                self.handle_completion(cmd_id, Vec::new())
            }
            TimerKind::Interval { key, .. } => {
                let op_ref = crate::ids::OpRef::from(key);
                let exec_id = self.allocate_exec_id();
                self.exec.frontier.push_back((op_ref, exec_id));
                Vec::new()
            }
            TimerKind::After { key } => {
                let cmd_id = CommandId::from(key);
                let value: Box<dyn SlotValue> = Box::new(crate::syscall::values::TriggerValue);
                self.handle_completion(cmd_id, vec![("trigger".to_string(), value)])
            }
        }
    }

    /// Merge a sender-claimed `src_peer_addresses` list into the
    /// receiver's `AddressBook` entry for `src_peer`. Empty list is
    /// a no-op (the sender chose not to advertise). The
    /// skip-on-unchanged guard compares the decoded list to the
    /// existing entry via slice equality and elides the write when
    /// they match — without this the receiver would rewrite the
    /// entry once per envelope, swamping the address book with
    /// idempotent updates under load.
    ///
    /// Returns `Err(SrcAddressMergeAllocError)` when the dedup
    /// buffer (S4) or the address-book peer dedup (S5) cannot
    /// reserve. The caller surfaces this as
    /// `WireReceiveError::AllocationFailed`; the address-book hint
    /// is best-effort under allocator pressure (the envelope's
    /// other fills still route).
    fn merge_src_peer_addresses(
        &mut self,
        src_peer: crate::ids::PeerId,
        claimed_bytes: &[Vec<u8>],
    ) -> Result<(), SrcAddressMergeAllocError> {
        if claimed_bytes.is_empty() {
            return Ok(());
        }
        // S4: dedup buffer for parsed Addresses. Use try_reserve_exact
        // so an exhausted allocator surfaces as AllocationFailed
        // rather than aborting the receiver.
        let mut claimed: Vec<crate::framework::Address> = Vec::new();
        let claim_count = claimed_bytes.len();
        if crate::fallible::try_reserve_exact(&mut claimed, claim_count).is_err() {
            return Err(SrcAddressMergeAllocError {
                byte_count: claim_count
                    .saturating_mul(std::mem::size_of::<crate::framework::Address>()),
                reason: crate::bus::AllocFailReason::HeapExhausted,
            });
        }
        for bytes in claimed_bytes {
            match crate::framework::Address::from_bytes(bytes) {
                Ok(addr) => claimed.push(addr),
                // Parse failure on one segment drops the hint without
                // touching the address book; the envelope's fills still
                // route via their own dest_suffix parsing.
                Err(_) => return Ok(()),
            }
        }
        if let Some(existing) = self.framework.address_book.lookup(src_peer) {
            if existing == claimed.as_slice() {
                return Ok(());
            }
        }
        // S5: `add_peer` runs its own try_reserve_exact for the
        // peer-entry dedup buffer. Allocation failure surfaces here
        // as AddressBookError::AllocationFailed; map to the same
        // WireReceiveErrorKind::AllocationFailed. Other errors
        // (Full, EmptyAddressList) are deployment-level signals
        // already observable elsewhere — swallow as before.
        match self.framework.address_book.add_peer(src_peer, claimed) {
            Ok(()) => Ok(()),
            Err(crate::framework::AddressBookError::AllocationFailed { requested }) => {
                Err(SrcAddressMergeAllocError {
                    byte_count: requested
                        .saturating_mul(std::mem::size_of::<crate::framework::Address>()),
                    reason: crate::bus::AllocFailReason::HeapExhausted,
                })
            }
            Err(_) => Ok(()),
        }
    }

    /// Merge a transport-observed source address into the
    /// receiver's `AddressBook` entry for `src_peer`. Skips the
    /// write when the entry already contains `addr` (slice
    /// containment check) so a steady stream of envelopes from the
    /// same observed endpoint costs at most one `register_address`
    /// call. When no entry exists yet — the sender-claimed merge
    /// upstream may have short-circuited on an empty list — the
    /// observed address bootstraps a fresh one via `add_peer`.
    fn merge_src_observed_address(
        &mut self,
        src_peer: crate::ids::PeerId,
        addr: crate::framework::Address,
    ) {
        if let Some(existing) = self.framework.address_book.lookup(src_peer) {
            if existing.contains(&addr) {
                return;
            }
            let _ = self.framework.address_book.register_address(src_peer, addr);
            return;
        }
        let _ = self.framework.address_book.add_peer(src_peer, vec![addr]);
    }

    /// Sender-side ingest of a `BackoffNotice` envelope per
    /// `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
    /// §5. Called from `route_envelope` when the first fill's
    /// `type_hash` matches `backoff_notice_type_hash`. Decodes the
    /// payload, applies the remote-advised back-off via
    /// `BackoffTable::record_remote_advisory`, records the matching
    /// `PeerGovernor::record_failure` so the existing 5-failure
    /// `LifecycleTransition::WentDown` path stays the single peer-
    /// down decision site, and returns no `EngineStep`s - the notice
    /// never reaches a user Component.
    ///
    /// When the source peer is unknown (the inbound `EnvelopeFrom`
    /// was synthesised by a transport that couldn't attribute the
    /// sender), the engine drops the notice silently. When the
    /// payload bytes fail to decode, it emits a
    /// `WireDecodeFailure` so operators observe the corruption.
    fn ingest_backoff_notice(
        &mut self,
        env: crate::envelope::WireEnvelope,
        src_peer: Option<crate::ids::PeerId>,
    ) -> Vec<EngineStep> {
        let Some(fill) = env.fills.into_iter().next() else {
            return Vec::new();
        };
        let Some(src_peer) = src_peer else {
            // No attributable sender; drop silently.
            return Vec::new();
        };
        let Some(payload) = crate::framework::BackoffNoticePayload::decode(&fill.payload) else {
            return vec![self.emit_wire_decode_failure(
                fill.type_hash,
                fill.payload.len(),
                "BackoffNoticePayload bincode decode failed".to_string(),
            )];
        };

        let now_ns = self.framework.scheduler.now_ns();
        // Advise the sender-side BackoffTable using the receiver's
        // quoted delay (§5.2). The existing BackoffGateTx already
        // reads `should_retry(peer, now_ns)` and respects the new
        // `next_retry_ns`, so no new gate is needed.
        self.framework.peer_state.backoff.record_remote_advisory(
            src_peer,
            now_ns,
            payload.min_backoff_ns,
        );
        // Record a peer-governor failure so the existing 5-failure
        // `LifecycleTransition::WentDown` surfacing remains the
        // single down-decision path (§5.3).
        let transition = self
            .framework
            .peer_state
            .governor
            .record_failure(src_peer, now_ns);
        // Surface the WentDown lifecycle transition if the receipt
        // of this notice pushed the sender's local view of the peer
        // across the threshold. The bus event mirrors the existing
        // PeerSuspect/PeerDown surfacing path.
        if matches!(transition, crate::framework::LifecycleTransition::WentDown,) {
            // The existing PhiTransition path emits PeerDown by site;
            // here the trigger is the per-peer governor decision so
            // no site-level info is available. Telemetry on the
            // tracker entry remains via PeerHealth.
        }
        // Cause is informational on the sender side - it's already
        // logged at the receiver via InfraEvent::BackoffNoticeSent.
        let _ = payload.cause();
        Vec::new()
    }

    /// Receiver-side back-pressure hook per
    /// `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
    /// §6. When the ingress depth crosses the high-water mark (or
    /// the caller forces emission via `force = true` from the
    /// φ-accrual scan), consult the `BackpressureTracker` for the
    /// `src_peer` and - on `Decision::EmitNotice` - mint a
    /// `BackoffNotice` envelope back to the sender, push it onto
    /// `OutboundQueue`, and publish the matching
    /// `InfraEvent::BackoffNoticeSent`. The K-then-silent transition
    /// surfaces `InfraEvent::SilentDropActive` as well so operators
    /// see the fallback engage. Returns the resulting `EngineStep`s;
    /// the caller appends them to the polling step list.
    fn maybe_emit_backoff_notice(
        &mut self,
        src_peer: crate::ids::PeerId,
        cause: crate::framework::BackoffCause,
        hint_ns: Option<u64>,
    ) -> Vec<EngineStep> {
        // Pull config-driven mark; PhiAccrual + ExplicitDrop callers
        // bypass the queue-depth check because they were triggered
        // by an external signal (φ flip / Component reject).
        let force = !matches!(cause, crate::framework::BackoffCause::QueueFull);
        if !force {
            // Compare the pre-drain snapshot captured at Phase 1
            // entry to the configured high-water fraction of
            // ingress capacity. The snapshot stays valid across
            // every `process_ingress_event` call inside the same
            // poll cycle - using the current `ingress.len()` would
            // see post-drain zero and never trip.
            let len = self.phase1_pre_drain_depth;
            let cap = self.ingress.capacity();
            if !self
                .framework
                .peer_state
                .backpressure
                .is_over_high_water(len, cap)
            {
                return Vec::new();
            }
        }

        let now_ns = self.framework.scheduler.now_ns();
        // `hint_ns` for QueueFull is sized by the configured
        // min-notice interval (the BackpressureTracker enforces the
        // floor). PhiAccrual callers may pass a specific mean
        // inter-arrival hint per §6(b); ExplicitDrop callers pass the
        // Component-supplied `BackpressureHint`.
        let min_hint = hint_ns.unwrap_or(0);
        let decision = self
            .framework
            .peer_state
            .backpressure
            .observe_overload(src_peer, cause, min_hint, now_ns);

        let (min_backoff_ns, cause_chosen) = match decision {
            crate::framework::BackpressureDecision::EmitNotice {
                min_backoff_ns,
                cause,
            } => (min_backoff_ns, cause),
            crate::framework::BackpressureDecision::Suppress
            | crate::framework::BackpressureDecision::SilentDrop => return Vec::new(),
        };

        // Build the notice envelope + push it on the outbound queue.
        let payload =
            crate::framework::BackoffNoticePayload::new(min_backoff_ns, cause_chosen, None);
        let envelope =
            crate::framework::build_backoff_notice_envelope(self.self_peer, src_peer, payload);
        self.framework.outbound_queue.push(envelope);

        // Bus event for ops dashboards + Component authors that
        // want to react to local overload signals.
        self.bus.publish(crate::bus::NodeEvent::Infra(
            crate::bus::InfraEvent::BackoffNoticeSent {
                peer: src_peer,
                cause: cause_chosen,
                min_backoff_ns,
            },
        ));

        // If this emission flipped the peer into silent-drop mode,
        // surface `SilentDropActive` once so operators see the
        // K-then-silent transition.
        if self
            .framework
            .peer_state
            .backpressure
            .is_silent_drop_active(src_peer)
        {
            self.bus.publish(crate::bus::NodeEvent::Infra(
                crate::bus::InfraEvent::SilentDropActive { peer: src_peer },
            ));
        }

        Vec::new()
    }
}

/// Returns `true` when a cycle has reached its op-invocation budget.
/// `None` budget always returns `false` (cap disabled).
#[inline]
fn budget_hit(budget: Option<usize>, ops_invoked: usize) -> bool {
    matches!(budget, Some(cap) if ops_invoked >= cap)
}

/// Reservation failure surfaced by `merge_src_peer_addresses` so the
/// caller can mint a single `WireReceiveError::AllocationFailed`
/// `EngineStep` carrying the bytes the boundary tried to claim.
/// Covers both S4 (the claimed-address dedup buffer) and S5
/// (`AddressBook::add_peer`'s peer-entry dedup buffer).
struct SrcAddressMergeAllocError {
    /// Approximate bytes the failing reservation requested
    /// (`address_count * size_of::<Address>()`). Mirrored into the
    /// `WireReceiveError::AllocationFailed::byte_count` field for
    /// telemetry.
    byte_count: usize,
    /// Why the reservation failed. `HeapExhausted` for both sites;
    /// the boundary has no per-item cap (cap-driven rejection lives
    /// on the application-ingress path, not here).
    reason: crate::bus::AllocFailReason,
}

#[cfg(test)]
#[path = "poll_recv_seed_tests.rs"]
mod poll_recv_seed_tests;

#[cfg(test)]
#[path = "poll_bus_routing_tests.rs"]
mod poll_bus_routing_tests;

#[cfg(test)]
#[path = "poll_ingress_handler_tests.rs"]
mod poll_ingress_handler_tests;

#[cfg(test)]
#[path = "poll_budget_tests.rs"]
mod poll_budget_tests;

#[cfg(test)]
#[path = "poll_async_error_tests.rs"]
mod poll_async_error_tests;

#[cfg(test)]
#[path = "poll_wire_timeout_tests.rs"]
mod poll_wire_timeout_tests;

#[cfg(test)]
#[path = "introspection_tests.rs"]
mod introspection_tests;

#[cfg(test)]
#[path = "peer_governor_tests.rs"]
mod peer_governor_tests;

#[cfg(test)]
#[path = "poll_backpressure_tests.rs"]
mod poll_backpressure_tests;

#[cfg(test)]
#[path = "poll_src_peer_addresses_tests.rs"]
mod poll_src_peer_addresses_tests;

#[cfg(test)]
#[path = "poll_observed_address_tests.rs"]
mod poll_observed_address_tests;

#[cfg(test)]
#[path = "poll_typed_receive_tests.rs"]
mod poll_typed_receive_tests;

#[cfg(test)]
#[path = "poll_ingress_alloc_tests.rs"]
mod poll_ingress_alloc_tests;

#[cfg(test)]
#[path = "poll_backend_materialize_tests.rs"]
mod poll_backend_materialize_tests;
