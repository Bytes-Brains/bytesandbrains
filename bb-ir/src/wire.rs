//! Wire codec hash helper.
//!
//! Type identity for graph values rides on `&'static TypeNode`
//! (`crate::types`). This module owns the `compute_wire_hash` FNV-1a
//! helper used to derive stable wire-envelope hashes from a
//! denotation + version pair. Concrete types stamp the result on
//! their `TypeNode::wire_hash` field.

/// FNV-1a 64-bit hash of `(denotation, "@", version)`. Available
/// as a `const fn` so callers can derive stable identifiers at
/// compile time (e.g. for graph-derived edge identity).
pub const fn compute_wire_hash(denotation: &str, version: i64) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    let bytes = denotation.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash ^= b'@' as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    let v = version.to_le_bytes();
    let mut j = 0;
    while j < v.len() {
        hash ^= v[j] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        j += 1;
    }
    hash
}

