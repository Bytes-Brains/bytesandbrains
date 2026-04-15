use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// Base58 alphabet (Bitcoin/IPFS variant)
const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// A unique identifier for peers in the network.
///
/// Fixed 64-byte identifier that can be:
/// - Hash of any data (embedding, peer info, etc.)
/// - Raw bytes from external systems (libp2p PeerId, etc.)
/// - Any unique 64-byte value
///
/// Being Copy-able (64 bytes) makes it efficient to pass around and use as HashMap keys.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct PeerId([u8; 64]);

impl PeerId {
    /// Create a PeerId from exactly 64 bytes
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        PeerId(bytes)
    }

    /// Create a PeerId from a slice, padding with zeros if needed, or truncating if too long
    pub fn from_slice(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 64];
        let len = bytes.len().min(64);
        arr[..len].copy_from_slice(&bytes[..len]);
        PeerId(arr)
    }

    /// Generate a PeerId from any hashable data using Rust's Hash trait.
    ///
    /// Hashes the data 8 times with different seeds to populate all 64 bytes.
    pub fn from_data<T: Hash>(data: T) -> Self {
        let mut bytes = [0u8; 64];
        for chunk in 0..8 {
            let mut hasher = DefaultHasher::new();
            chunk.hash(&mut hasher);
            data.hash(&mut hasher);
            let hash_value = hasher.finish();
            bytes[chunk * 8..(chunk + 1) * 8].copy_from_slice(&hash_value.to_be_bytes());
        }
        PeerId(bytes)
    }

    /// Get the raw bytes as a slice
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Convert PeerId to owned bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    /// Get a short hex representation (first 4 bytes = 8 hex chars)
    pub fn short_hex(&self) -> String {
        self.0[..4].iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get the full hex representation (all 64 bytes = 128 hex chars)
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get base58 representation (full 64 bytes)
    pub fn base58(&self) -> String {
        Self::bytes_to_base58(&self.0)
    }

    /// Get short base58 representation (first 8 bytes)
    pub fn short_base58(&self) -> String {
        let mut short_bytes = [0u8; 8];
        short_bytes.copy_from_slice(&self.0[..8]);
        Self::bytes_to_base58(&short_bytes)
    }

    /// Convert bytes to base58
    fn bytes_to_base58(bytes: &[u8]) -> String {
        let mut digits = vec![0u8];

        for &byte in bytes {
            let mut carry = byte as usize;
            for digit in &mut digits {
                carry += (*digit as usize) << 8;
                *digit = (carry % 58) as u8;
                carry /= 58;
            }
            while carry > 0 {
                digits.push((carry % 58) as u8);
                carry /= 58;
            }
        }

        // Add leading zeros
        for &byte in bytes {
            if byte == 0 {
                digits.push(0);
            } else {
                break;
            }
        }

        // Convert to base58 characters
        digits.reverse();
        digits
            .iter()
            .map(|&d| BASE58_ALPHABET[d as usize] as char)
            .collect()
    }
}

impl fmt::Debug for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_base58())
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_base58())
    }
}

impl Ord for PeerId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for PeerId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bytes() {
        let bytes = [42u8; 64];
        let id = PeerId::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
    }

    #[test]
    fn test_from_slice_exact() {
        let bytes = [1u8; 64];
        let id = PeerId::from_slice(&bytes);
        assert_eq!(id.as_bytes(), &bytes);
    }

    #[test]
    fn test_from_slice_short() {
        let bytes = [1u8, 2, 3];
        let id = PeerId::from_slice(&bytes);
        assert_eq!(id.as_bytes()[0], 1);
        assert_eq!(id.as_bytes()[1], 2);
        assert_eq!(id.as_bytes()[2], 3);
        assert_eq!(id.as_bytes()[3], 0);
    }

    #[test]
    fn test_from_data_deterministic() {
        let id1 = PeerId::from_data("hello");
        let id2 = PeerId::from_data("hello");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_from_data_different() {
        let id1 = PeerId::from_data("hello");
        let id2 = PeerId::from_data("world");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_from_data_fills_all_bytes() {
        let id = PeerId::from_data("test-full-fill");
        let bytes = id.as_bytes();
        // Bytes beyond position 8 should not all be zero
        assert!(bytes[8..].iter().any(|&b| b != 0));
    }

    #[test]
    fn test_ordering() {
        let id1 = PeerId::from_slice(&[0u8]);
        let id2 = PeerId::from_slice(&[1u8]);
        assert!(id1 < id2);
    }

    #[test]
    fn test_display() {
        let id = PeerId::from_data("test");
        let s = format!("{}", id);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_hex() {
        let mut bytes = [0u8; 64];
        bytes[0] = 0xAB;
        bytes[1] = 0xCD;
        let id = PeerId::from_bytes(bytes);
        assert!(id.hex().starts_with("abcd"));
        assert_eq!(id.short_hex(), "abcd0000");
    }

    #[test]
    fn test_copy() {
        let id1 = PeerId::from_data("test");
        let id2 = id1; // Copy
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_hash_as_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let id = PeerId::from_data("key");
        map.insert(id, "value");
        assert_eq!(map.get(&id), Some(&"value"));
    }
}
