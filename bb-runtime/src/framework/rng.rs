//! `RngU64Source` - pluggable u64 RNG used by the `RngU64`
//! syscall op. Default impl wraps `getrandom`.

/// Source of `u64` random values. Trait so tests can supply a
/// deterministic counter via a different impl.
pub trait RngU64Source: Send + Sync {
    /// Pull the next `u64`.
    fn next_u64(&mut self) -> u64;
}

/// Default `getrandom`-backed RNG.
#[derive(Default)]
pub struct GetrandomU64;

impl GetrandomU64 {
    /// Construct a fresh RNG.
    pub fn new() -> Self {
        Self
    }
}

impl RngU64Source for GetrandomU64 {
    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        // Panic on failure - entropy exhaustion is unrecoverable for
        // the engine.
        getrandom::getrandom(&mut buf).expect("getrandom");
        u64::from_le_bytes(buf)
    }
}

/// Deterministic counter RNG for tests.
pub struct CounterRng(pub u64);

impl RngU64Source for CounterRng {
    fn next_u64(&mut self) -> u64 {
        let v = self.0;
        self.0 = self.0.wrapping_add(1);
        v
    }
}

