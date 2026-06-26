use super::*;
use bb_runtime::engine::Engine;
use bb_runtime::framework::FrameworkComponents;
use bb_runtime::ids::OpRef;
use bb_runtime::syscall::values::CommandIdValue;
use std::collections::HashMap;

macro_rules! with_ctx {
    ($framework:ident, $bus:ident, $counters:ident, $ctx:ident, $body:block) => {{
        let mut $framework = FrameworkComponents::new();
        let mut $bus = bb_runtime::bus::TypedBus::new();
        let mut $counters: HashMap<String, u64> = HashMap::new();
        let mut $ctx = RuntimeResourceRef {
            peers: bb_runtime::runtime::PeerCtx {
                gate: &mut $framework.peer_state.gate,
                backoff: &mut $framework.peer_state.backoff,
                governor: &mut $framework.peer_state.governor,
                addresses: &mut $framework.address_book,
                backpressure: &mut $framework.peer_state.backpressure,
            },
            net: bb_runtime::runtime::NetCtx {
                outbound: &mut $framework.outbound_queue,
                rtt: &mut $framework.rtt_tracker,
                requests: &mut $framework.request_tracker,
                dedup: &mut $framework.inbound_dedup,
                pending_peer_resolve_failures: &mut $framework.pending_peer_resolve_failures,
            },
            time: bb_runtime::runtime::TimeCtx {
                scheduler: &mut $framework.scheduler,
            },
            syscall: bb_runtime::runtime::SyscallCtx {
                serialize_queue: &mut $framework.serialize_queue,
                hold_table: &mut $framework.hold_table,
                record_buffer: &mut $framework.record_buffer,
                event_source: &mut $framework.event_source,
                counters: &mut $counters,
                any_fired_groups: &mut $framework.any_fired_groups,
                deadline_match_fired: &mut $framework.deadline_match_fired,
                rng: &mut *$framework.rng,
                pending_app_events: &mut $framework.pending_app_events,
            },
            bus: &mut $bus,
            ingress: std::sync::Arc::new(bb_runtime::ingress::IngressQueue::new()),
            components: bb_runtime::runtime::ComponentsView::default(),
            current: bb_runtime::runtime::CurrentCallCtx {
                op_ref: OpRef::new(0),
                exec_id: bb_runtime::ids::ExecId::from(0u64),
                self_peer: bb_runtime::ids::PeerId::from(0u64),
                node_attributes: &[],
                node_metadata: &[],
                inbound: bb_runtime::runtime::InboundCtx {
                    src_peer: None,
                    wire_req_id: None,
                    arrival_ns: None,
                    remaining_deadline_ns: None,
                },
                pending_completions: Vec::new(),
                next_command_id: Box::leak(Box::new(0u64)),
            },
        };
        $body
    }};
}

#[test]
fn bootstrap_output_returns_async_with_cmd_id_from_input() {
    let cmd_handle = CommandIdValue(bb_runtime::ids::CommandId::from(12345));
    let inputs: Vec<(&str, &dyn SlotValue)> = vec![("cmd", &cmd_handle)];

    with_ctx!(framework, bus, counters, ctx, {
        let result = invoke_bootstrap_output(&NodeProto::default(), &inputs, &mut ctx).unwrap();
        match result {
            DispatchResult::Async(cmd) => assert_eq!(cmd.as_u64(), 12345),
            other => panic!("expected Async, got {other:?}"),
        }
    });
}

#[test]
fn bootstrap_output_missing_cmd_input_errors() {
    let inputs: Vec<(&str, &dyn SlotValue)> = vec![];

    with_ctx!(framework, bus, counters, ctx, {
        let err = invoke_bootstrap_output(&NodeProto::default(), &inputs, &mut ctx).unwrap_err();
        assert!(err.detail.contains("missing"));
    });
}

#[test]
fn bootstrap_output_wrong_input_type_errors() {
    let trigger = TriggerValue;
    let inputs: Vec<(&str, &dyn SlotValue)> = vec![("cmd", &trigger)];

    with_ctx!(framework, bus, counters, ctx, {
        let err = invoke_bootstrap_output(&NodeProto::default(), &inputs, &mut ctx).unwrap_err();
        assert!(err.detail.contains("CommandIdValue"));
    });
}

#[test]
fn register_lifecycle_op_is_idempotent() {
    let mut engine = Engine::new();
    let op = OpRef::new(7);
    engine.register_lifecycle_op("Bootstrap", op);
    engine.register_lifecycle_op("Bootstrap", op);
    engine.register_lifecycle_op("Steady", op);

    let bootstrap_ops = engine.lifecycle_table.get("Bootstrap").expect("enrolled");
    assert_eq!(bootstrap_ops.len(), 1);
    assert_eq!(bootstrap_ops[0], op);
    assert_eq!(
        engine.lifecycle_table.get("Steady").map(|v| v.len()),
        Some(1)
    );
}
