use bb_ir::proto::onnx::{attribute_proto, AttributeProto, NodeProto};
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::TypedBus;
use bb_runtime::framework::FrameworkComponents;
use bb_runtime::ids::OpRef;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::syscall::values::TriggerValue;

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
fn pulse_emits_trigger() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let r = super::pulse::invoke(&NodeProto::default(), &[], &mut c).expect("ok");
    match r {
        DispatchResult::Immediate(outs) => {
            assert_eq!(outs.len(), 1);
            assert!(outs[0].1.as_any().is::<TriggerValue>());
        }
        _ => panic!(),
    }
}

#[test]
fn on_trigger_requires_input() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let err = super::on_trigger::invoke(&NodeProto::default(), &[], &mut c).expect_err("");
    assert!(err.detail.contains("one input"));
}

#[test]
fn interval_schedules_next_firing() {
    let mut framework = FrameworkComponents::new();
    framework.scheduler.set_now(100);
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node = NodeProto {
        attribute: vec![AttributeProto {
            name: "period_ns".into(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: 50,
            ..Default::default()
        }],
        ..Default::default()
    };
    let r = super::interval::invoke(&node, &[], &mut c).expect("ok");
    match r {
        DispatchResult::Immediate(outs) => {
            assert_eq!(outs.len(), 1);
            // `tick` is a `TimestampValue` carrying now_ns at firing.
            use bb_runtime::syscall::values::TimestampValue;
            let ts = outs[0]
                .1
                .as_any()
                .downcast_ref::<TimestampValue>()
                .expect("tick is TimestampValue");
            assert_eq!(ts.0, 100);
        }
        _ => panic!("expected Immediate"),
    }
    drop(c);
    assert!(framework.scheduler.has_matured(150));
}

#[test]
fn after_schedules_async_completion() {
    let mut framework = FrameworkComponents::new();
    framework.scheduler.set_now(0);
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node = NodeProto {
        attribute: vec![AttributeProto {
            name: "delay_ns".into(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: 100,
            ..Default::default()
        }],
        ..Default::default()
    };
    let r = super::after::invoke(&node, &[], &mut c).expect("ok");
    assert!(matches!(r, DispatchResult::Async(_)));
}

#[test]
fn event_source_emits_trigger() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let r = super::event_source::invoke(&NodeProto::default(), &[], &mut c).expect("ok");
    assert!(matches!(r, DispatchResult::Immediate(ref o) if o.len() == 1));
}
