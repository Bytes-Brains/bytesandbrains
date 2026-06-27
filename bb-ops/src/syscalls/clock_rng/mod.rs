//! Clock, RNG, sleep, and deadline-match syscalls.

pub mod clock;
pub mod deadline_match;
pub mod rng_u64;
pub mod sleep;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
