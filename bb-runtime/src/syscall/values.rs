//! Typed primitive carriers shipped by the framework.
//!
//! Each typed carrier wraps one well-known framework primitive
//! (`PeerId`, `CommandId`, etc.). Slot-table residency comes from
//! the blanket impl on [`crate::slot_value::SlotValue`] - the
//! carriers derive `Serialize + Deserialize + Clone` and that is
//! the entire contract. Downstream consumers downcast to the
//! concrete carrier the graph guarantees at each site.

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use bb_ir::slot_value::{wire_decoder_registry, SlotValue as IrSlotValue};
use bb_ir::types::{
    TYPE_ADDRESS_VEC, TYPE_BYTES, TYPE_COMPOSITE, TYPE_MULTIADDRESS, TYPE_PEER_ID,
    TYPE_PEER_ID_VEC, TYPE_TRIGGER, TYPE_WIRE_REQ_ID,
};
use bb_ir::{register_charged_bytes, register_type_node};

use crate::framework::Address;
use crate::ids::{CommandId, PeerId};

/// Zero-payload signal value. Used for `Trigger` outputs from
/// syscall ops (Pulse, OnTrigger, Threshold, Interval, After, etc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerValue;

/// Typed carrier for a `PeerId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerIdValue(pub PeerId);

/// Typed carrier for a `Vec<PeerId>` — the cardinal peer-list shape
/// flowing into a `wire.Send`. Sampler / DHT / static-constant
/// Components produce this; the Send fans out one envelope per
/// peer. An empty vec is a valid no-op (zero envelopes shipped).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerIdVecValue(pub Vec<PeerId>);

/// Typed carrier for a `CommandId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandIdValue(pub CommandId);

/// Typed carrier for a wall-clock timestamp (nanoseconds since
/// epoch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimestampValue(pub u64);

/// Typed carrier for a `wire_req_id` (request/response correlation
/// id minted by the wire syscall).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireReqIdValue(pub u64);

/// Typed carrier for a `CorrelateTag` correlation token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrelationTokenValue(pub u64);

/// Typed carrier for a multiaddr `Address`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressValue(pub Address);

/// Typed carrier for `Vec<Address>` — the ordered local-address bag
/// installed via `bb::install`, the payload shape of the multi-address
/// `AddressBook` syscalls, and the value flowing into `wire.Send`'s
/// `src_peer_addresses` stamp. Ordering reflects caller preference;
/// the receiver merges into the AddressBook with positional dedup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressVecValue(pub Vec<Address>);

/// Generic byte payload carrier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BytesValue(pub Vec<u8>);

/// Envelope holding N typed child slot values bundled into one
/// wire-eligible slot. Children are owned typed carriers; in-process
/// Bundle/Unbundle round-trip preserves the concrete type without a
/// bincode hop. The wire codec (Commit 2) serialises children as
/// `(type_hash, child.to_wire_bytes())` tuples and the receiver
/// materialises typed children via the decoder registry.
pub struct CompositeValue {
    /// Owned typed children, positionally aligned with the recorded
    /// `Bundle` input slots. Each entry is the original concrete
    /// `SlotValue` cloned via `clone_boxed`.
    pub children: Vec<Box<dyn IrSlotValue>>,
}

impl Clone for CompositeValue {
    fn clone(&self) -> Self {
        Self {
            children: self.children.iter().map(|c| c.clone_boxed()).collect(),
        }
    }
}

impl std::fmt::Debug for CompositeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Trait-object children have no Debug bound; surface count
        // and per-child type_hash so test failures are diagnosable
        // without forcing a Debug bound onto every SlotValue impl.
        let hashes: Vec<u64> = self.children.iter().map(|c| c.type_hash()).collect();
        f.debug_struct("CompositeValue")
            .field("children_len", &self.children.len())
            .field("child_type_hashes", &hashes)
            .finish()
    }
}

// Wire codec: emit each child as a `(type_hash, child_bytes)` pair,
// length-prefixed by the serde sequence header. The receiver looks
// up each type_hash in the `wire_decoder_registry` and materialises
// the typed child. Bincode's sequence + tuple encoding gives a
// compact length-prefixed layout without a hand-rolled framing
// scheme.
impl Serialize for CompositeValue {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut seq = ser.serialize_seq(Some(self.children.len()))?;
        for child in &self.children {
            let bytes = child.to_wire_bytes().map_err(|e| {
                serde::ser::Error::custom(format!("CompositeValue child encode: {e}"))
            })?;
            // Each entry is a 2-tuple `(u64, Vec<u8>)` so the
            // receiver can read the type_hash before consuming the
            // payload.
            seq.serialize_element(&WireChild {
                type_hash: child.type_hash(),
                bytes,
            })?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for CompositeValue {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct ChildrenVisitor;
        impl<'de> Visitor<'de> for ChildrenVisitor {
            type Value = Vec<Box<dyn IrSlotValue>>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a sequence of (type_hash, bytes) child entries")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out: Vec<Box<dyn IrSlotValue>> =
                    Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(entry) = seq.next_element::<WireChild>()? {
                    let decoder = wire_decoder_registry()
                        .get(&entry.type_hash)
                        .copied()
                        .ok_or_else(|| {
                            de::Error::custom(format!(
                                "CompositeValue child decode: no decoder registered for type_hash {:#018x}",
                                entry.type_hash,
                            ))
                        })?;
                    let child = decoder(&entry.bytes).map_err(|e| {
                        de::Error::custom(format!("CompositeValue child decode: {e}"))
                    })?;
                    out.push(child);
                }
                Ok(out)
            }
        }
        let children = de.deserialize_seq(ChildrenVisitor)?;
        Ok(CompositeValue { children })
    }
}

/// On-wire shape of a single child: type discriminator + payload.
/// Serialized as a 2-tuple so the bincode layout is `(u64, Vec<u8>)`
/// with no field-name overhead.
struct WireChild {
    type_hash: u64,
    bytes: Vec<u8>,
}

impl Serialize for WireChild {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut t = ser.serialize_tuple(2)?;
        t.serialize_element(&self.type_hash)?;
        t.serialize_element(&self.bytes)?;
        t.end()
    }
}

impl<'de> Deserialize<'de> for WireChild {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        // Reuse the derived serde for `(u64, Vec<u8>)`; the explicit
        // tuple visitor would duplicate that logic without gain.
        let (type_hash, bytes) = <(u64, Vec<u8>)>::deserialize(de)?;
        Ok(WireChild { type_hash, bytes })
    }
}

/// Generic `u64` carrier - used for tests and `bb.u64`-typed slot
/// values that don't have a more specific carrier above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct U64Value(pub u64);

// ---- Type-node registrations ------------------------------------
//
// Each typed carrier wraps a specific lattice concrete leaf.
// `SlotValue::runtime_type` consults this registry at runtime; the
// compiler's TypeSolver seeds inputs from these mappings.

register_type_node!(TriggerValue, &TYPE_TRIGGER);
register_type_node!(PeerIdValue, &TYPE_PEER_ID);
register_type_node!(PeerIdVecValue, &TYPE_PEER_ID_VEC);
register_type_node!(WireReqIdValue, &TYPE_WIRE_REQ_ID);
register_type_node!(BytesValue, &TYPE_BYTES);
register_type_node!(AddressValue, &TYPE_MULTIADDRESS);
register_type_node!(AddressVecValue, &TYPE_ADDRESS_VEC);
register_type_node!(CompositeValue, &TYPE_COMPOSITE);

// `BytesValue` is the framework-side ingress carrier — every wire
// payload that lands as a framework carrier (not a backend-mediated
// tensor) flows through it. Reporting `self.0.len()` lets the
// slot-table writer release the wire-byte charge on overwrite /
// eviction without the caller having to retain the count
// separately. The default `SlotValue::charged_bytes` body returns
// 0 for every other concrete carrier.
register_charged_bytes!(BytesValue, |b: &BytesValue| b.0.len());

