use super::*;
use bb_ir::proto::onnx::{attribute_proto, AttributeProto};
use bb_runtime::bus::TypedBus;
use bb_runtime::framework::FrameworkComponents;
use bb_runtime::ids::OpRef;

fn str_attr(n: &str, v: &str) -> AttributeProto {
    AttributeProto {
        name: n.into(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: v.as_bytes().to_vec(),
        ..Default::default()
    }
}
fn int_attr(n: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: n.into(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: v,
        ..Default::default()
    }
}

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
fn app_emit_pushes_app_event() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let v = bb_runtime::syscall::values::BytesValue(vec![1]);
    let node = NodeProto {
        attribute: vec![str_attr("name", "topic")],
        ..Default::default()
    };
    invoke_app_emit(&node, &[("v", &v)], &mut c).expect("");
    drop(c);
    assert_eq!(f.pending_app_events.len(), 1);
    matches!(&f.pending_app_events[0], AppEvent::Emit { name, .. } if name == "topic");
}

#[test]
fn app_notify_pushes_notify_marker() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let node = NodeProto {
        attribute: vec![str_attr("name", "topic")],
        ..Default::default()
    };
    invoke_app_notify(&node, &[], &mut c).expect("");
    drop(c);
    assert!(matches!(f.pending_app_events[0], AppEvent::Notify { .. }));
}

#[test]
fn record_writes_to_record_buffer() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let v = bb_runtime::syscall::values::BytesValue(vec![9]);
    let node = NodeProto {
        attribute: vec![str_attr("name", "obs")],
        ..Default::default()
    };
    invoke_record(&node, &[("v", &v)], &mut c).expect("");
    drop(c);
    let snap = f.record_buffer.snapshot("obs");
    assert_eq!(snap, vec![vec![9]]);
}

#[test]
fn incr_metric_bumps_counter() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let node = NodeProto {
        attribute: vec![str_attr("name", "ops"), int_attr("delta", 5)],
        ..Default::default()
    };
    invoke_incr_metric(&node, &[], &mut c).expect("");
    invoke_incr_metric(&node, &[], &mut c).expect("");
    drop(c);
    assert_eq!(f.counters.get("ops").copied(), Some(10));
}

#[test]
fn app_emit_typed_mismatch_surfaces_error() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let wrong = bb_runtime::syscall::values::U64Value(7);
    let node = NodeProto {
        attribute: vec![str_attr("name", "topic")],
        ..Default::default()
    };
    let err = invoke_app_emit(&node, &[("v", &wrong)], &mut c)
        .expect_err("U64Value into BytesValue slot must error");
    assert_eq!(err.kind, bb_runtime::bus::OpErrorKind::TypeMismatch);
    assert_eq!(err.reason, "expected_bytes");
}

#[test]
fn record_typed_mismatch_surfaces_error() {
    let mut f = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut f, &mut bus);
    let wrong = bb_runtime::syscall::values::U64Value(8);
    let node = NodeProto {
        attribute: vec![str_attr("name", "obs")],
        ..Default::default()
    };
    let err = invoke_record(&node, &[("v", &wrong)], &mut c)
        .expect_err("U64Value into BytesValue slot must error");
    assert_eq!(err.kind, bb_runtime::bus::OpErrorKind::TypeMismatch);
    assert_eq!(err.reason, "expected_bytes");
}
