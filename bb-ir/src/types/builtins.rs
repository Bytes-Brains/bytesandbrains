//! Built-in `TypeNode` constants the framework + DSL ship out of
//! the box.
//!
//! Every framework primitive's type is listed here and submitted to
//! the inventory so the [`super::Lattice`] picks it up at startup.
//! Custom application or library types follow the same pattern from
//! any downstream crate.

use super::{TypeKind, TypeNode, TypeNodeReg};

// ---- Root --------------------------------------------------------

/// The universal supertype. Every other type is a (transitive)
/// subtype of `Any`. Used as a port bound when no narrower
/// constraint applies.
pub static TYPE_ANY: TypeNode = TypeNode {
    id: "any",
    parent: None,
    kind: TypeKind::Abstract,
    ffi_name: "",
    wire_hash: 0,
    denotation: "",
};

// ---- Tensor branch -----------------------------------------------

/// Abstract `Tensor` - matches any `Tensor<T>` concrete leaf.
pub static TYPE_TENSOR: TypeNode = TypeNode {
    id: "tensor",
    parent: Some("any"),
    kind: TypeKind::Abstract,
    ffi_name: "",
    wire_hash: 0,
    denotation: "",
};

/// Concrete `Tensor<F32>` - dense f32 tensor.
pub static TYPE_TENSOR_F32: TypeNode = TypeNode {
    id: "tensor.f32",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_f32_t",
    wire_hash: 0x0000_0000_0000_0101,
    denotation: "ai.bytesandbrains.tensor.f32",
};

/// Concrete `Tensor<F64>` - dense f64 tensor.
pub static TYPE_TENSOR_F64: TypeNode = TypeNode {
    id: "tensor.f64",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_f64_t",
    wire_hash: 0x0000_0000_0000_0102,
    denotation: "ai.bytesandbrains.tensor.f64",
};

/// Concrete `Tensor<F16>` - dense f16 tensor.
pub static TYPE_TENSOR_F16: TypeNode = TypeNode {
    id: "tensor.f16",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_f16_t",
    wire_hash: 0x0000_0000_0000_0103,
    denotation: "ai.bytesandbrains.tensor.f16",
};

/// Concrete `Tensor<U8>` - dense u8 tensor.
pub static TYPE_TENSOR_U8: TypeNode = TypeNode {
    id: "tensor.u8",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_u8_t",
    wire_hash: 0x0000_0000_0000_0104,
    denotation: "ai.bytesandbrains.tensor.u8",
};

/// Concrete `Tensor<I32>` - dense i32 tensor.
pub static TYPE_TENSOR_I32: TypeNode = TypeNode {
    id: "tensor.i32",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_i32_t",
    wire_hash: 0x0000_0000_0000_0105,
    denotation: "ai.bytesandbrains.tensor.i32",
};

/// Concrete `Tensor<Bool>` - dense bool tensor. Masks, predicates,
/// comparison-op results.
pub static TYPE_TENSOR_BOOL: TypeNode = TypeNode {
    id: "tensor.bool",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_bool_t",
    wire_hash: 0x0000_0000_0000_0106,
    denotation: "ai.bytesandbrains.tensor.bool",
};

/// Concrete `Tensor<BF16>` - dense bfloat16 tensor. Modern ML
/// mixed-precision standard. Stored as `half::bf16` (binary newtype
/// around the 16-bit bfloat bit pattern).
pub static TYPE_TENSOR_BF16: TypeNode = TypeNode {
    id: "tensor.bf16",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_bf16_t",
    wire_hash: 0x0000_0000_0000_0107,
    denotation: "ai.bytesandbrains.tensor.bf16",
};

/// Concrete `Tensor<I8>` - dense i8 tensor. Edge quantization (the
/// positional reason this is shipping as a v1 type, not a future
/// feature: edge ML without int8 isn't really edge ML).
pub static TYPE_TENSOR_I8: TypeNode = TypeNode {
    id: "tensor.i8",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_i8_t",
    wire_hash: 0x0000_0000_0000_0108,
    denotation: "ai.bytesandbrains.tensor.i8",
};

/// Concrete `Tensor<I16>` - dense i16 tensor. Wider-than-byte
/// quantization slots, audio samples.
pub static TYPE_TENSOR_I16: TypeNode = TypeNode {
    id: "tensor.i16",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_i16_t",
    wire_hash: 0x0000_0000_0000_0109,
    denotation: "ai.bytesandbrains.tensor.i16",
};

/// Concrete `Tensor<I64>` - dense i64 tensor. Index / embedding
/// lookups (the canonical 'Int' kind default for most Burn
/// underlying backends).
pub static TYPE_TENSOR_I64: TypeNode = TypeNode {
    id: "tensor.i64",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_i64_t",
    wire_hash: 0x0000_0000_0000_010A,
    denotation: "ai.bytesandbrains.tensor.i64",
};

/// Concrete `Tensor<U16>` - dense u16 tensor. Image bit-depth, count
/// types.
pub static TYPE_TENSOR_U16: TypeNode = TypeNode {
    id: "tensor.u16",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_u16_t",
    wire_hash: 0x0000_0000_0000_010B,
    denotation: "ai.bytesandbrains.tensor.u16",
};

/// Concrete `Tensor<U32>` - dense u32 tensor. Hash buckets, large
/// counts.
pub static TYPE_TENSOR_U32: TypeNode = TypeNode {
    id: "tensor.u32",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_u32_t",
    wire_hash: 0x0000_0000_0000_010C,
    denotation: "ai.bytesandbrains.tensor.u32",
};

/// Concrete `Tensor<U64>` - dense u64 tensor. Wide indices, IDs.
pub static TYPE_TENSOR_U64: TypeNode = TypeNode {
    id: "tensor.u64",
    parent: Some("tensor"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_tensor_u64_t",
    wire_hash: 0x0000_0000_0000_010D,
    denotation: "ai.bytesandbrains.tensor.u64",
};

// ---- Scalar branch -----------------------------------------------

/// Abstract `Scalar` - matches any `Scalar<T>` concrete leaf.
pub static TYPE_SCALAR: TypeNode = TypeNode {
    id: "scalar",
    parent: Some("any"),
    kind: TypeKind::Abstract,
    ffi_name: "",
    wire_hash: 0,
    denotation: "",
};

/// Concrete `Scalar<F32>` - single f32 value.
pub static TYPE_SCALAR_F32: TypeNode = TypeNode {
    id: "scalar.f32",
    parent: Some("scalar"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_f32_t",
    wire_hash: 0x0000_0000_0000_0201,
    denotation: "bb.f32",
};

/// Concrete `Scalar<F64>` - single f64 value.
pub static TYPE_SCALAR_F64: TypeNode = TypeNode {
    id: "scalar.f64",
    parent: Some("scalar"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_f64_t",
    wire_hash: 0x0000_0000_0000_0202,
    denotation: "bb.f64",
};

/// Concrete `Scalar<F16>` - single f16 value.
pub static TYPE_SCALAR_F16: TypeNode = TypeNode {
    id: "scalar.f16",
    parent: Some("scalar"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_f16_t",
    wire_hash: 0x0000_0000_0000_0203,
    denotation: "bb.f16",
};

/// Concrete `Scalar<U8>` - single u8 value.
pub static TYPE_SCALAR_U8: TypeNode = TypeNode {
    id: "scalar.u8",
    parent: Some("scalar"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_u8_t",
    wire_hash: 0x0000_0000_0000_0204,
    denotation: "bb.u8",
};

/// Concrete `Scalar<I32>` - single i32 value.
pub static TYPE_SCALAR_I32: TypeNode = TypeNode {
    id: "scalar.i32",
    parent: Some("scalar"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_i32_t",
    wire_hash: 0x0000_0000_0000_0205,
    denotation: "bb.i32",
};

// ---- Peer + framework primitives ---------------------------------

/// Concrete `PeerId` - canonical peer identifier (multihash).
pub static TYPE_PEER_ID: TypeNode = TypeNode {
    id: "peer_id",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_peer_id_t",
    wire_hash: 0x0000_0000_0000_0010,
    denotation: "bb.peer_id",
};

/// Concrete `Vec<PeerId>` - fan-out destination list for `g.net_out`.
pub static TYPE_PEER_ID_VEC: TypeNode = TypeNode {
    id: "peer_id_vec",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_peer_id_vec_t",
    wire_hash: 0x0000_0000_0000_0012,
    denotation: "bb.peer_id_vec",
};

/// Concrete `Trigger` - zero-payload signal value.
pub static TYPE_TRIGGER: TypeNode = TypeNode {
    id: "trigger",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_trigger_t",
    wire_hash: 0x0000_0000_0000_000A,
    denotation: "bb.trigger",
};

/// Concrete `Bytes` - opaque byte buffer.
pub static TYPE_BYTES: TypeNode = TypeNode {
    id: "bytes",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_bytes_t",
    wire_hash: 0x0000_0000_0000_000B,
    denotation: "ai.bytesandbrains.opaque",
};

/// Concrete `WireReqId` - correlation token returned by `g.net_out`.
pub static TYPE_WIRE_REQ_ID: TypeNode = TypeNode {
    id: "wire_req_id",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_wire_req_id_t",
    wire_hash: 0x0000_0000_0000_0011,
    denotation: "bb.wire_req_id",
};

/// Concrete `Multiaddress` - sequence of typed protocol segments
/// describing a delivery path. The framework's canonical address
/// type; binds to `framework::Address` in `bb-runtime`. Lives
/// directly under `Any` (no abstract address parent — there is no
/// second address-shaped concrete). The 0x03xx wire-hash range is
/// reserved for address-related leaves.
pub static TYPE_MULTIADDRESS: TypeNode = TypeNode {
    id: "multiaddress",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_multiaddress_t",
    wire_hash: 0x0000_0000_0000_0301,
    denotation: "ai.bytesandbrains.multiaddress",
};

/// Concrete `Vec<Address>` - the ordered local-or-peer address list
/// stamped on every `wire.Send` envelope, the carrier for the local
/// address bag installed by `bb::install`, and the payload shape for
/// the multi-address `AddressBook` syscalls. Distinct wire hash from
/// `TYPE_MULTIADDRESS` so receivers can disambiguate single vs many.
pub static TYPE_ADDRESS_VEC: TypeNode = TypeNode {
    id: "address_vec",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_address_vec_t",
    wire_hash: 0x0000_0000_0000_0303,
    denotation: "ai.bytesandbrains.address_vec",
};

/// Concrete `Composite` - envelope holding N typed child payloads,
/// each tagged with its source `type_hash` and bincode-encoded bytes.
/// The DSL `g.bundle` recorder produces this on the output of a
/// `Bundle` op; the matching `Unbundle` op decomposes it back into
/// per-child `BytesValue` outputs whose `ValueInfoProto.denotation`
/// is stamped from the declared child-type list. Parent is `Any`
/// because a composite can wrap any wire-eligible child types.
pub static TYPE_COMPOSITE: TypeNode = TypeNode {
    id: "composite",
    parent: Some("any"),
    kind: TypeKind::Concrete,
    ffi_name: "bb_composite_t",
    wire_hash: 0x0000_0000_0000_0302,
    denotation: "ai.bytesandbrains.composite",
};

// ---- Inventory submissions ---------------------------------------
//
// Every built-in type registers itself so `lookup_by_id` and the
// `Lattice` see them at startup. Custom types follow the same
// pattern from any downstream crate.

inventory::submit! { TypeNodeReg(&TYPE_ANY) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_F32) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_F64) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_F16) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_U8) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_I32) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_BOOL) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_BF16) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_I8) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_I16) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_I64) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_U16) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_U32) }
inventory::submit! { TypeNodeReg(&TYPE_TENSOR_U64) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR_F32) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR_F64) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR_F16) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR_U8) }
inventory::submit! { TypeNodeReg(&TYPE_SCALAR_I32) }
inventory::submit! { TypeNodeReg(&TYPE_PEER_ID) }
inventory::submit! { TypeNodeReg(&TYPE_PEER_ID_VEC) }
inventory::submit! { TypeNodeReg(&TYPE_TRIGGER) }
inventory::submit! { TypeNodeReg(&TYPE_BYTES) }
inventory::submit! { TypeNodeReg(&TYPE_WIRE_REQ_ID) }
inventory::submit! { TypeNodeReg(&TYPE_MULTIADDRESS) }
inventory::submit! { TypeNodeReg(&TYPE_ADDRESS_VEC) }
inventory::submit! { TypeNodeReg(&TYPE_COMPOSITE) }

// ---- Inverse lookup (denotation → TypeNode) ----------------------

/// Map a canonical denotation string to the corresponding
/// [`TypeNode`] static. Returns `None` for denotations the framework
/// doesn't recognize; the solver leaves those values at
/// [`TYPE_ANY`] (custom types can declare their own
/// `type_relations` to seed concrete types).
pub fn lookup_denotation(denotation: &str) -> Option<&'static TypeNode> {
    match denotation {
        // Tensors (bb-ir::wire `TypeNode` canonical strings).
        "ai.bytesandbrains.tensor.f32" => Some(&TYPE_TENSOR_F32),
        "ai.bytesandbrains.tensor.f64" => Some(&TYPE_TENSOR_F64),
        "ai.bytesandbrains.tensor.f16" => Some(&TYPE_TENSOR_F16),
        "ai.bytesandbrains.tensor.u8" => Some(&TYPE_TENSOR_U8),
        "ai.bytesandbrains.tensor.i32" => Some(&TYPE_TENSOR_I32),
        "ai.bytesandbrains.tensor.bool" => Some(&TYPE_TENSOR_BOOL),
        "ai.bytesandbrains.tensor.bf16" => Some(&TYPE_TENSOR_BF16),
        "ai.bytesandbrains.tensor.i8" => Some(&TYPE_TENSOR_I8),
        "ai.bytesandbrains.tensor.i16" => Some(&TYPE_TENSOR_I16),
        "ai.bytesandbrains.tensor.i64" => Some(&TYPE_TENSOR_I64),
        "ai.bytesandbrains.tensor.u16" => Some(&TYPE_TENSOR_U16),
        "ai.bytesandbrains.tensor.u32" => Some(&TYPE_TENSOR_U32),
        "ai.bytesandbrains.tensor.u64" => Some(&TYPE_TENSOR_U64),

        // Scalars.
        "bb.f32" => Some(&TYPE_SCALAR_F32),
        "bb.f64" => Some(&TYPE_SCALAR_F64),
        "bb.f16" => Some(&TYPE_SCALAR_F16),
        "bb.u8" => Some(&TYPE_SCALAR_U8),
        "bb.i32" => Some(&TYPE_SCALAR_I32),

        // Framework primitives.
        "bb.peer_id" => Some(&TYPE_PEER_ID),
        "bb.peer_id_vec" => Some(&TYPE_PEER_ID_VEC),
        "bb.trigger" => Some(&TYPE_TRIGGER),
        "bb.wire_req_id" => Some(&TYPE_WIRE_REQ_ID),
        "ai.bytesandbrains.multiaddress" => Some(&TYPE_MULTIADDRESS),
        "ai.bytesandbrains.address_vec" => Some(&TYPE_ADDRESS_VEC),
        "ai.bytesandbrains.composite" => Some(&TYPE_COMPOSITE),

        // Opaque placeholder — the recording-time signal that the
        // type is a raw byte payload (no further structure beyond
        // bytes-on-the-wire). Maps to the concrete `bytes` leaf so
        // strict-types-by-default accepts it.
        "ai.bytesandbrains.opaque" => Some(&TYPE_BYTES),

        _ => None,
    }
}
