//! Sub-F shared tests - exercise each invoke fn through the
//! engine's register_syscall path.

use bb_ir::proto::onnx::{attribute_proto, AttributeProto, NodeProto};
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::TypedBus;
use bb_runtime::framework::FrameworkComponents;
use bb_runtime::ids::OpRef;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::syscall::values::{TimestampValue, U64Value};

fn ctx<'a>(
    framework: &'a mut FrameworkComponents,
    bus: &'a mut TypedBus,
) -> RuntimeResourceRef<'a> {
    RuntimeResourceRef {
        peers: bb_runtime::runtime::PeerCtx {
            gate: &mut framework.peer_state.gate,
            backoff: &mut framework.peer_state.backoff,
            governor: &mut framework.peer_state.governor,
            addresses: &mut framework.address_book,
            backpressure: &mut framework.peer_state.backpressure,
        },
        net: bb_runtime::runtime::NetCtx {
            outbound: &mut framework.outbound_queue,
            rtt: &mut framework.rtt_tracker,
            requests: &mut framework.request_tracker,
            dedup: &mut framework.inbound_dedup,
            pending_peer_resolve_failures: &mut framework.pending_peer_resolve_failures,
        },
        time: bb_runtime::runtime::TimeCtx {
            scheduler: &mut framework.scheduler,
        },
        syscall: bb_runtime::runtime::SyscallCtx {
            serialize_queue: &mut framework.serialize_queue,
            hold_table: &mut framework.hold_table,
            record_buffer: &mut framework.record_buffer,
            event_source: &mut framework.event_source,
            counters: &mut framework.counters,
            any_fired_groups: &mut framework.any_fired_groups,
            deadline_match_fired: &mut framework.deadline_match_fired,
            rng: &mut *framework.rng,
            pending_app_events: &mut framework.pending_app_events,
        },
        bus,
        ingress: std::sync::Arc::new(bb_runtime::ingress::IngressQueue::new()),
        components: bb_runtime::runtime::ComponentsView::default(),
        current: bb_runtime::runtime::CurrentCallCtx {
            op_ref: OpRef::from(0u64),
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
    }
}

#[test]
fn clock_emits_current_now_ns() {
    let mut framework = FrameworkComponents::new();
    framework.scheduler.set_now(12345);
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let result = super::clock::invoke(&NodeProto::default(), &[], &mut c).expect("ok");
    match result {
        DispatchResult::Immediate(outs) => {
            let v = outs[0]
                .1
                .as_any()
                .downcast_ref::<TimestampValue>()
                .expect("TimestampValue");
            assert_eq!(v.0, 12345);
        }
        _ => panic!(),
    }
}

#[test]
fn rng_u64_emits_u64_payload() {
    use bb_runtime::framework::CounterRng;
    let mut framework = FrameworkComponents::new();
    framework.rng = Box::new(CounterRng(99));
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let result = super::rng_u64::invoke(&NodeProto::default(), &[], &mut c).expect("ok");
    match result {
        DispatchResult::Immediate(outs) => {
            let v = outs[0]
                .1
                .as_any()
                .downcast_ref::<U64Value>()
                .expect("U64Value");
            assert_eq!(v.0, 99);
        }
        _ => panic!(),
    }
}

#[test]
fn sleep_schedules_timer_and_returns_async() {
    let mut framework = FrameworkComponents::new();
    framework.scheduler.set_now(100);
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node = NodeProto {
        attribute: vec![AttributeProto {
            name: "duration_ns".into(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: 50,
            ..Default::default()
        }],
        ..Default::default()
    };
    let result = super::sleep::invoke(&node, &[], &mut c).expect("ok");
    assert!(matches!(result, DispatchResult::Async(_)));
    drop(c);
    assert_eq!(framework.scheduler.len(), 1);
    assert!(framework.scheduler.has_matured(150));
}

#[test]
fn deadline_match_rejects_no_inputs() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let err = super::deadline_match::invoke(&NodeProto::default(), &[], &mut c).expect_err("");
    assert!(err.detail.contains("at least one"));
}

#[test]
fn deadline_match_latches_after_first_fire() {
    use bb_runtime::syscall::values::TriggerValue;
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);

    let trigger = TriggerValue;
    let inputs: Vec<(&str, &dyn bb_runtime::slot_value::SlotValue)> = vec![("then", &trigger)];

    let r1 = super::deadline_match::invoke(&NodeProto::default(), &inputs, &mut c).expect("");
    match r1 {
        DispatchResult::Immediate(outs) => assert_eq!(outs.len(), 1, "winner emitted"),
        _ => panic!("expected Immediate"),
    }

    let r2 = super::deadline_match::invoke(&NodeProto::default(), &inputs, &mut c).expect("");
    match r2 {
        DispatchResult::Immediate(outs) => assert!(outs.is_empty(), "absorbed"),
        _ => panic!("expected Immediate"),
    }
}
