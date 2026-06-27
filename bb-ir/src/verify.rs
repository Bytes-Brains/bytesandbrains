//! IR well-formedness checkers run between compiler passes. Each
//! function is a pure check returning `Result<(), VerifyError>`;
//! the compiler invokes them at the per-pass seams described in
//! `docs/COMPILER.md`.

use std::collections::HashSet;

use crate::proto::onnx::{FunctionProto, ModelProto};

/// IR shape failure with enough context to locate the offending op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// `NodeProto.op_type` empty.
    EmptyOpType {
        /// Function carrying the bad node.
        function_name: String,
        /// Position within the function's `node`.
        node_index: usize,
    },

    /// `wire.Send` without a matching `wire.Recv`. Downstream blocks
    /// forever.
    UnpairedWireSend {
        /// Token stamped on the orphan Send.
        wire_id: u64,
        /// Host function name.
        function_name: String,
    },

    /// `wire.Recv` without a matching `wire.Send`.
    UnpairedWireRecv {
        /// Token stamped on the orphan Recv.
        wire_id: u64,
        /// Host function name.
        function_name: String,
    },

    /// `Call*` node names a function absent from `model.functions`.
    UnresolvedFunctionCall {
        /// Callee name.
        target_name: String,
        /// Caller function name.
        function_name: String,
        /// Position of the bad `Call*` node within the caller.
        node_index: usize,
    },

    /// `ModelProto.functions` empty.
    EmptyFunctionTable,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyOpType {
                function_name,
                node_index,
            } => write!(
                f,
                "empty op_type at function `{}` node #{}",
                function_name, node_index
            ),
            Self::UnpairedWireSend {
                wire_id,
                function_name,
            } => write!(
                f,
                "wire.Send with wire_id={} in function `{}` has no matching Recv",
                wire_id, function_name
            ),
            Self::UnpairedWireRecv {
                wire_id,
                function_name,
            } => write!(
                f,
                "wire.Recv with wire_id={} in function `{}` has no matching Send",
                wire_id, function_name
            ),
            Self::UnresolvedFunctionCall {
                target_name,
                function_name,
                node_index,
            } => write!(
                f,
                "function `{}` node #{} calls undefined function `{}`",
                function_name, node_index, target_name
            ),
            Self::EmptyFunctionTable => f.write_str("ModelProto.functions is empty"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Verify every `NodeProto.op_type` is non-empty and the function
/// table is non-empty.
pub fn types(model: &ModelProto) -> Result<(), VerifyError> {
    if model.functions.is_empty() {
        return Err(VerifyError::EmptyFunctionTable);
    }
    for function in &model.functions {
        for (i, node) in function.node.iter().enumerate() {
            if node.op_type.is_empty() {
                return Err(VerifyError::EmptyOpType {
                    function_name: function.name.clone(),
                    node_index: i,
                });
            }
        }
    }
    Ok(())
}

/// Verify each `wire_id` has both a `Send` and a `Recv`.
pub fn wire_pairs(model: &ModelProto) -> Result<(), VerifyError> {
    for function in &model.functions {
        let mut sends: HashSet<u64> = HashSet::new();
        let mut recvs: HashSet<u64> = HashSet::new();
        for node in &function.node {
            let Some(wire_id) = read_wire_id(node) else {
                continue;
            };
            if node.op_type == "Send" {
                sends.insert(wire_id);
            } else if node.op_type == "Recv" {
                recvs.insert(wire_id);
            }
        }
        if let Some(wire_id) = sends.difference(&recvs).next() {
            return Err(VerifyError::UnpairedWireSend {
                wire_id: *wire_id,
                function_name: function.name.clone(),
            });
        }
        if let Some(wire_id) = recvs.difference(&sends).next() {
            return Err(VerifyError::UnpairedWireRecv {
                wire_id: *wire_id,
                function_name: function.name.clone(),
            });
        }
    }
    Ok(())
}

/// Verify each `Call*` node names a function in `model.functions`.
pub fn function_calls(model: &ModelProto) -> Result<(), VerifyError> {
    let defined: HashSet<&str> = model.functions.iter().map(|f| f.name.as_str()).collect();
    for function in &model.functions {
        for (i, node) in function.node.iter().enumerate() {
            if !node.op_type.starts_with("Call") {
                continue;
            }
            // Recorder stamps the target name on `node.name`.
            let target = node.name.as_str();
            if target.is_empty() {
                continue;
            }
            if !defined.contains(target) {
                return Err(VerifyError::UnresolvedFunctionCall {
                    target_name: target.to_string(),
                    function_name: function.name.clone(),
                    node_index: i,
                });
            }
        }
    }
    Ok(())
}

/// Read [`crate::keys::WIRE_ID_KEY`] as `u64`. `None` when missing
/// or non-numeric.
fn read_wire_id(node: &crate::proto::onnx::NodeProto) -> Option<u64> {
    node.metadata_props
        .iter()
        .find(|p| p.key == crate::keys::WIRE_ID_KEY)
        .and_then(|p| p.value.parse::<u64>().ok())
}

/// Single-function wire-id check (no wrapping `ModelProto` needed).
pub fn wire_pairs_in_function(function: &FunctionProto) -> Result<(), VerifyError> {
    let mut sends: HashSet<u64> = HashSet::new();
    let mut recvs: HashSet<u64> = HashSet::new();
    for node in &function.node {
        let Some(wire_id) = read_wire_id(node) else {
            continue;
        };
        if node.op_type == "Send" {
            sends.insert(wire_id);
        } else if node.op_type == "Recv" {
            recvs.insert(wire_id);
        }
    }
    if let Some(wire_id) = sends.difference(&recvs).next() {
        return Err(VerifyError::UnpairedWireSend {
            wire_id: *wire_id,
            function_name: function.name.clone(),
        });
    }
    if let Some(wire_id) = recvs.difference(&sends).next() {
        return Err(VerifyError::UnpairedWireRecv {
            wire_id: *wire_id,
            function_name: function.name.clone(),
        });
    }
    Ok(())
}

