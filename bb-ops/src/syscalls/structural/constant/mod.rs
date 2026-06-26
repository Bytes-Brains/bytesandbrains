//! `Constant` syscall component - emits a constant value at
//! bootstrap. Reads the `value` AttributeProto's TensorProto at
//! invoke time and emits the encoded `TensorProto` bytes as a
//! `BytesValue`. Downstream tensor-shaped consumers decode via
//! `Tensor::from_proto` using their declared scalar type.

use prost::Message;

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::registry::OpRegistration;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::BytesValue;

pub use bb_ir::syscall_ids::OP_CONSTANT as OP_TYPE;
pub use bb_ir::syscall_ids::SYSCALL_DOMAIN as DOMAIN;

/// Engine dispatch-table marker.
pub struct ConstantOp;

/// Invoke fn - emits the `value` attribute's TensorProto bytes
/// as a `BytesValue`.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let value_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| OpError {
            detail: "Constant requires a `value` attribute".to_string(),
            ..Default::default()
        })?;
    let Some(tensor) = &value_attr.t else {
        return Err(OpError {
            detail: "Constant `value` attribute missing TensorProto".to_string(),
            ..Default::default()
        });
    };
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        Box::new(BytesValue(tensor.encode_to_vec())),
    )]))
}

inventory::submit! {
    OpRegistration {
        domain: DOMAIN,
        op_type: OP_TYPE,
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

