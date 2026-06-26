//! `Tee` syscall component - fans a single input out to N outputs.
//! `fanout` attribute (int) sets N (default 2).

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::registry::OpRegistration;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

pub use bb_ir::syscall_ids::OP_TEE as OP_TYPE;
pub use bb_ir::syscall_ids::SYSCALL_DOMAIN as DOMAIN;

/// Engine dispatch-table marker.
pub struct TeeOp;

/// Invoke fn - duplicates the single input into `fanout` outputs
/// via polymorphic `SlotValue::clone_boxed`. Each output preserves
/// the concrete type.
pub fn invoke(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let Some((_, input)) = inputs.first() else {
        return Err(OpError {
            detail: "Tee requires one input".to_string(),
            ..Default::default()
        });
    };
    let fanout = node
        .attribute
        .iter()
        .find(|a| a.name == "fanout")
        .map(|a| a.i.max(1) as usize)
        .unwrap_or(2);

    let mut outs: Vec<(String, Box<dyn SlotValue>)> = Vec::with_capacity(fanout);
    for i in 0..fanout {
        outs.push((format!("out_{i}"), input.clone_boxed()));
    }
    Ok(DispatchResult::Immediate(outs))
}

inventory::submit! {
    OpRegistration {
        domain: DOMAIN,
        op_type: OP_TYPE,
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

