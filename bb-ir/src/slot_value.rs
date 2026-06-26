//! Universal slot value trait. The blanket impl is the only path,
//! so anything `Any + Send + Sync + Clone + Serialize +
//! DeserializeOwned` is a `SlotValue` by construction. Type
//! identity per slot rides on `ValueInfoProto.type_node`; consumers
//! downcast to the graph-guaranteed concrete type.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{de::DeserializeOwned, Serialize};

use crate::types::{TypeNode, TYPE_ANY};

/// Wire-coding failure for a `SlotValue`. The default bincode path
/// is infallible for vanilla serde derives; custom `Serialize` impls
/// and missing receiver-side decoders surface here.
#[derive(Debug)]
pub enum SlotValueError {
    /// Encoder returned an error; boxed inner is the serde
    /// diagnostic.
    EncodeFailed(Box<dyn std::error::Error + Send + Sync>),
    /// Decoder returned an error; boxed inner is the serde
    /// diagnostic.
    DecodeFailed(Box<dyn std::error::Error + Send + Sync>),
    /// Receiver has no registered decoder for the stamped
    /// `type_hash` (older / divergent build).
    UnknownTypeHash(u64),
}

impl std::fmt::Display for SlotValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EncodeFailed(err) => write!(f, "SlotValue::to_wire_bytes failed: {err}"),
            Self::DecodeFailed(err) => write!(f, "SlotValue decode failed: {err}"),
            Self::UnknownTypeHash(hash) => write!(
                f,
                "SlotValue decode: no registered decoder for type_hash {hash:#018x}",
            ),
        }
    }
}

impl std::error::Error for SlotValueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EncodeFailed(err) => Some(err.as_ref()),
            Self::DecodeFailed(err) => Some(err.as_ref()),
            Self::UnknownTypeHash(_) => None,
        }
    }
}

/// Universal slot value. Slot-table values, op outputs, and
/// `dispatch_atomic` inputs are all `Box<dyn SlotValue>` /
/// `&dyn SlotValue`. Local forwarding uses `clone_boxed`; wire +
/// snapshot paths use `to_wire_bytes`.
pub trait SlotValue: Any + Send + Sync {
    /// Downcast surface — recover the concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Repackage `Box<dyn SlotValue>` as `Box<dyn Any>` for
    /// [`Box::downcast`]. Required because the `SlotValue` and
    /// `Any` vtables are distinct even though `SlotValue: Any`.
    fn into_any_boxed(self: Box<Self>) -> Box<dyn Any + Send + Sync>;

    /// Polymorphic clone preserving the concrete type.
    fn clone_boxed(&self) -> Box<dyn SlotValue>;

    /// Wire-boundary encode (bincode + serde). Local forwarding
    /// uses `clone_boxed` instead.
    fn to_wire_bytes(&self) -> Result<Vec<u8>, SlotValueError>;

    /// Stable cross-Node type discriminator. FNV-1a of
    /// `std::any::type_name::<T>()`; receiver decodes only on a
    /// matching hash.
    fn type_hash(&self) -> u64;

    /// Runtime [`TypeNode`] for this value. Returns the leaf
    /// registered via [`register_type_node!`] or [`TYPE_ANY`].
    /// Consulted at wire boundaries + TypeSolver seeding; the
    /// atomic-dispatch hot path uses compile-time-stamped closures.
    fn runtime_type(&self) -> &'static TypeNode {
        let tid = self.as_any().type_id();
        runtime_type_registry()
            .get(&tid)
            .copied()
            .unwrap_or(&TYPE_ANY)
    }

    /// Bytes the carrier owes against
    /// `NodeConfig::ingress_byte_budget`. Slot-table eviction calls
    /// this to release the charge. Default `0` — only
    /// ingress-derived carriers register a non-zero resolver via
    /// [`register_charged_bytes!`].
    fn charged_bytes(&self) -> usize {
        let any = self.as_any();
        let tid = any.type_id();
        match charged_bytes_registry().get(&tid) {
            Some(f) => f(any),
            None => 0,
        }
    }
}

/// Inventory entry mapping `TypeId → &'static TypeNode`. Submitted
/// by [`register_type_node!`].
pub struct RuntimeTypeBinding {
    /// `TypeId::of::<T>` is non-const; use a closure.
    pub type_id_fn: fn() -> TypeId,
    /// Lattice node this concrete type resolves to.
    pub type_node: &'static TypeNode,
}

inventory::collect!(RuntimeTypeBinding);

/// Startup-built `TypeId → &TypeNode` map driving
/// [`SlotValue::runtime_type`].
pub fn runtime_type_registry() -> &'static HashMap<TypeId, &'static TypeNode> {
    static REG: OnceLock<HashMap<TypeId, &'static TypeNode>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut m: HashMap<TypeId, &'static TypeNode> = HashMap::new();
        for binding in inventory::iter::<RuntimeTypeBinding> {
            m.insert((binding.type_id_fn)(), binding.type_node);
        }
        m
    })
}

/// Per-type charged-bytes resolver. Takes the carrier's erased
/// `&dyn Any` and returns the byte count.
pub type ChargedBytesFn = fn(&dyn Any) -> usize;

/// Inventory entry mapping `TypeId → ChargedBytesFn`. Submitted by
/// [`register_charged_bytes!`].
pub struct ChargedBytesBinding {
    /// `TypeId::of::<T>` is non-const; use a closure.
    pub type_id_fn: fn() -> TypeId,
    /// Downcasts and returns the wire-byte count.
    pub resolve_fn: ChargedBytesFn,
}

inventory::collect!(ChargedBytesBinding);

/// Startup-built `TypeId → ChargedBytesFn` map driving
/// [`SlotValue::charged_bytes`].
pub fn charged_bytes_registry() -> &'static HashMap<TypeId, ChargedBytesFn> {
    static REG: OnceLock<HashMap<TypeId, ChargedBytesFn>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut m: HashMap<TypeId, ChargedBytesFn> = HashMap::new();
        for binding in inventory::iter::<ChargedBytesBinding> {
            m.insert((binding.type_id_fn)(), binding.resolve_fn);
        }
        m
    })
}

/// Register a carrier's wire-byte resolver.
///
/// ```ignore
/// register_charged_bytes!(BytesValue, |b: &BytesValue| b.0.len());
/// ```
#[macro_export]
macro_rules! register_charged_bytes {
    ($t:ty, $resolve:expr) => {
        $crate::inventory::submit! {
            $crate::slot_value::ChargedBytesBinding {
                type_id_fn: || ::std::any::TypeId::of::<$t>(),
                resolve_fn: |any| {
                    let resolve: fn(&$t) -> usize = $resolve;
                    match any.downcast_ref::<$t>() {
                        Some(v) => resolve(v),
                        None => 0,
                    }
                },
            }
        }
    };
}

/// Wire-decode fn for a known concrete type.
pub type WireDecodeFn = fn(&[u8]) -> Result<Box<dyn SlotValue>, SlotValueError>;

/// Inventory entry mapping `type_hash → WireDecodeFn`. Emitted by
/// [`register_type_node!`].
pub struct WireDecoderBinding {
    /// Concrete type's stable `type_hash`.
    pub type_hash_fn: fn() -> u64,
    /// Bincode decoder.
    pub decode_fn: WireDecodeFn,
}

inventory::collect!(WireDecoderBinding);

/// Startup-built `type_hash → WireDecodeFn` map used by
/// `CompositeValue`'s wire codec to materialise typed children.
pub fn wire_decoder_registry() -> &'static HashMap<u64, WireDecodeFn> {
    static REG: OnceLock<HashMap<u64, WireDecodeFn>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut m: HashMap<u64, WireDecodeFn> = HashMap::new();
        for binding in inventory::iter::<WireDecoderBinding> {
            m.insert((binding.type_hash_fn)(), binding.decode_fn);
        }
        m
    })
}

/// Register a concrete type's lattice [`TypeNode`] + wire decoder.
/// Emits both a [`RuntimeTypeBinding`] and a [`WireDecoderBinding`].
/// Unregistered types resolve to [`crate::types::TYPE_ANY`] and
/// their wire payloads cannot be decoded.
///
/// ```ignore
/// use bb_ir::slot_value::register_type_node;
/// use bb_ir::types::TYPE_PEER_ID;
/// register_type_node!(PeerIdValue, &TYPE_PEER_ID);
/// ```
#[macro_export]
macro_rules! register_type_node {
    ($t:ty, $node:expr) => {
        $crate::inventory::submit! {
            $crate::slot_value::RuntimeTypeBinding {
                type_id_fn: || ::std::any::TypeId::of::<$t>(),
                type_node: $node,
            }
        }
        $crate::inventory::submit! {
            $crate::slot_value::WireDecoderBinding {
                type_hash_fn: || $crate::slot_value::type_hash_of::<$t>(),
                decode_fn: |bytes| {
                    $crate::bincode::deserialize::<$t>(bytes)
                        .map(|v| Box::new(v) as Box<dyn $crate::slot_value::SlotValue>)
                        .map_err(|e| $crate::slot_value::SlotValueError::DecodeFailed(Box::new(e)))
                },
            }
        }
    };
}

/// The only `SlotValue` impl path. Manual impls are not supported —
/// "derive serde + Clone" gets you a wire-eligible carrier.
impl<T> SlotValue for T
where
    T: Any + Send + Sync + Clone + Serialize + DeserializeOwned,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any_boxed(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }

    fn clone_boxed(&self) -> Box<dyn SlotValue> {
        Box::new(self.clone())
    }

    fn to_wire_bytes(&self) -> Result<Vec<u8>, SlotValueError> {
        bincode::serialize(self).map_err(|e| SlotValueError::EncodeFailed(Box::new(e)))
    }

    fn type_hash(&self) -> u64 {
        type_hash_of::<T>()
    }
}

/// FNV-1a 64-bit hash of `std::any::type_name::<T>()`. Deterministic
/// across runs of the same Rust toolchain; stamped onto
/// `SlotFill.type_hash` and matched on the receive side.
#[inline]
pub fn type_hash_of<T: ?Sized + 'static>() -> u64 {
    fnv1a_64(std::any::type_name::<T>().as_bytes())
}

/// `const`-callable FNV-1a 64-bit. Used by [`type_hash_of`] and
/// compile-time `TY_*` constant builders.
#[inline]
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}
