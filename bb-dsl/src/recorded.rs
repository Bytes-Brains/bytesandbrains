//! `RecordedModule` - the compiler's input type.
//!
//! Bundles the recorded `FunctionProto` body. Per the chosen-path
//! install contract, the IR carries no concrete instance state —
//! install constructs every concrete via the inventory's
//! `construct_fn`, so the recorder doesn't capture
//! `(serialize_fn, restore_fn)` side tables.

use bb_ir::proto::onnx::FunctionProto;

/// Hand-off bundle between the DSL `Graph` (recording surface) and
/// the compiler (consumer).
#[derive(Debug, Default)]
pub struct RecordedModule {
    /// The root recorded FunctionProto body. Top-level
    /// `with_module(self.name(), …)` wraps fold into this entry.
    pub function: FunctionProto,

    /// Sub-functions created by nested `with_module(name, …)`
    /// calls. Each becomes its own FunctionProto in the compiled
    /// `ModelProto.functions[1..]` so the compiler + runtime can
    /// chase them via `ai.bytesandbrains.module` CALL NodeProtos
    /// emitted in `function` (or in other sub-functions).
    pub sub_functions: Vec<FunctionProto>,
}
