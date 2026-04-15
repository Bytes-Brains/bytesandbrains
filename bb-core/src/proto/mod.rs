// ONNX proto definitions
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

// bb-core proto definitions
pub mod bb_core_proto {
    include!(concat!(env!("OUT_DIR"), "/bb_core.rs"));
}

// Re-export for convenience
pub use bb_core_proto::*;
pub use onnx::TensorProto;

/// ONNX data type identifier for 32-bit floating point.
pub const DATA_TYPE_FLOAT: i32 = 1;

/// ONNX data type identifier for 8-bit unsigned integer.
pub const DATA_TYPE_UINT8: i32 = 2;


#[derive(Debug, Clone)]
pub enum ProtoConversionError {
    ConversionFailed(String),
    InvalidDataType { expected: i32, actual: i32 },
    InvalidTensorShape { expected: Vec<i64>, actual: Vec<i64> },
}

impl std::fmt::Display for ProtoConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtoConversionError::ConversionFailed(msg) => write!(f, "Proto conversion failed: {}", msg),
            ProtoConversionError::InvalidDataType { expected, actual } => {
                write!(f, "Invalid data type: expected {}, actual {}", expected, actual)
            }
            ProtoConversionError::InvalidTensorShape { expected, actual } => {
                write!(f, "Invalid tensor shape: expected {:?}, actual {:?}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ProtoConversionError {}

mod conversions;
pub use conversions::{addresses_from_proto, addresses_to_proto};
