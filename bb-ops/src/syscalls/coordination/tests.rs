use super::*;
use bb_ir::proto::onnx::{attribute_proto, AttributeProto};
use bb_runtime::bus::TypedBus;
use bb_runtime::framework::FrameworkComponents;
use bb_runtime::ids::OpRef;
use bb_runtime::syscall::values::{BytesValue, U64Value};

fn make_str_attr(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
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
fn limit_acquire_then_release_round_trip() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node = NodeProto {
        attribute: vec![make_str_attr("name", "conn"), make_int_attr("n", 1)],
        ..Default::default()
    };
    // First acquire succeeds.
    let r = invoke_limit_acquire(&node, &[], &mut c).expect("");
    assert!(matches!(r, DispatchResult::Immediate(ref o) if o.len() == 1));
    // Second is gated.
    let r2 = invoke_limit_acquire(&node, &[], &mut c).expect("");
    assert!(matches!(r2, DispatchResult::Immediate(ref o) if o.is_empty()));
    // Release then re-acquire.
    let release_node = NodeProto {
        attribute: vec![make_str_attr("name", "conn")],
        ..Default::default()
    };
    invoke_limit_release(&release_node, &[], &mut c).expect("");
    let r3 = invoke_limit_acquire(&node, &[], &mut c).expect("");
    assert!(matches!(r3, DispatchResult::Immediate(ref o) if o.len() == 1));
}

#[test]
fn serialize_enqueue_dequeue_round_trip() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let v = BytesValue(vec![1, 2, 3]);
    let enq_node = NodeProto {
        attribute: vec![make_str_attr("queue", "Q")],
        ..Default::default()
    };
    invoke_serialize_enqueue(&enq_node, &[("v", &v)], &mut c).expect("");
    let deq_node = NodeProto {
        attribute: vec![make_str_attr("queue", "Q")],
        ..Default::default()
    };
    let r = invoke_serialize_dequeue(&deq_node, &[], &mut c).expect("");
    match r {
        DispatchResult::Immediate(outs) => {
            assert_eq!(outs.len(), 1);
            let bytes = outs[0].1.as_any().downcast_ref::<BytesValue>().expect("");
            assert_eq!(bytes.0, vec![1, 2, 3]);
        }
        _ => panic!(),
    }
}

#[test]
fn correlate_tag_mints_token() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let r = invoke_correlate_tag(&NodeProto::default(), &[], &mut c).expect("");
    match r {
        DispatchResult::Immediate(outs) => {
            assert_eq!(outs.len(), 1);
            assert_eq!(outs[0].0, "token");
            outs[0]
                .1
                .as_any()
                .downcast_ref::<bb_runtime::syscall::values::CorrelationTokenValue>()
                .expect("token is CorrelationTokenValue");
        }
        _ => panic!(),
    }
}

#[test]
fn hold_stash_flush_round_trip() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let v = BytesValue(vec![7, 8]);
    let stash = NodeProto {
        attribute: vec![make_str_attr("slot", "S")],
        ..Default::default()
    };
    invoke_hold_stash(&stash, &[("v", &v)], &mut c).expect("");
    let flush = NodeProto {
        attribute: vec![make_str_attr("slot", "S")],
        ..Default::default()
    };
    let r = invoke_hold_flush(&flush, &[], &mut c).expect("");
    match r {
        DispatchResult::Immediate(outs) => {
            assert_eq!(outs.len(), 1);
            let bytes = outs[0].1.as_any().downcast_ref::<BytesValue>().expect("");
            assert_eq!(bytes.0, vec![7, 8]);
        }
        _ => panic!(),
    }
}

#[test]
fn any_emits_first_input() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let v = U64Value(11);
    let r = invoke_any(&NodeProto::default(), &[("x", &v)], &mut c).expect("");
    match r {
        DispatchResult::Immediate(outs) => {
            let v = outs[0].1.as_any().downcast_ref::<U64Value>().expect("");
            assert_eq!(v.0, 11);
        }
        _ => panic!(),
    }
}

#[test]
fn any_latches_per_group_after_first_arrival() {
    // Same `group` attribute, two arrivals: first emits the value;
    // the second is absorbed (Immediate empty).
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node = NodeProto {
        attribute: vec![make_str_attr("group", "boot")],
        ..Default::default()
    };
    let v1 = U64Value(1);
    let v2 = U64Value(2);

    let r1 = invoke_any(&node, &[("x", &v1)], &mut c).expect("");
    match r1 {
        DispatchResult::Immediate(outs) => assert_eq!(outs.len(), 1),
        _ => panic!("expected Immediate"),
    }

    let r2 = invoke_any(&node, &[("x", &v2)], &mut c).expect("");
    match r2 {
        DispatchResult::Immediate(outs) => assert!(outs.is_empty(), "absorbed"),
        _ => panic!("expected Immediate"),
    }
}

#[test]
fn any_different_groups_fire_independently() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let node_a = NodeProto {
        attribute: vec![make_str_attr("group", "a")],
        ..Default::default()
    };
    let node_b = NodeProto {
        attribute: vec![make_str_attr("group", "b")],
        ..Default::default()
    };
    let v = U64Value(1);

    let r_a = invoke_any(&node_a, &[("x", &v)], &mut c).expect("");
    let r_b = invoke_any(&node_b, &[("x", &v)], &mut c).expect("");

    match (r_a, r_b) {
        (DispatchResult::Immediate(a), DispatchResult::Immediate(b)) => {
            assert_eq!(a.len(), 1);
            assert_eq!(b.len(), 1);
        }
        _ => panic!("both first-firings should emit"),
    }
}

#[test]
fn gate_releases_value_when_invoked() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let v = U64Value(9);
    let r = invoke_gate(&NodeProto::default(), &[("v", &v)], &mut c).expect("");
    match r {
        DispatchResult::Immediate(outs) => {
            let v = outs[0].1.as_any().downcast_ref::<U64Value>().expect("");
            assert_eq!(v.0, 9);
        }
        _ => panic!(),
    }
}

#[test]
fn serialize_enqueue_typed_mismatch_surfaces_error() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let wrong = U64Value(42);
    let node = NodeProto {
        attribute: vec![make_str_attr("queue", "Q")],
        ..Default::default()
    };
    let err = invoke_serialize_enqueue(&node, &[("v", &wrong)], &mut c)
        .expect_err("U64Value into BytesValue slot must error");
    assert_eq!(err.kind, bb_runtime::bus::OpErrorKind::TypeMismatch);
    assert_eq!(err.reason, "expected_bytes");
}

#[test]
fn hold_stash_typed_mismatch_surfaces_error() {
    let mut framework = FrameworkComponents::new();
    let mut bus = TypedBus::new();
    let mut c = ctx(&mut framework, &mut bus);
    let wrong = U64Value(7);
    let node = NodeProto {
        attribute: vec![make_str_attr("slot", "S")],
        ..Default::default()
    };
    let err = invoke_hold_stash(&node, &[("v", &wrong)], &mut c)
        .expect_err("U64Value into BytesValue slot must error");
    assert_eq!(err.kind, bb_runtime::bus::OpErrorKind::TypeMismatch);
    assert_eq!(err.reason, "expected_bytes");
}
