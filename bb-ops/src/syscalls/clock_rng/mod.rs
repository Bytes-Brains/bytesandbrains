//! Clock, RNG, sleep, deadline-match ops. //!
//! Spec: Sub-F in `docs/IR_AND_DSL.md` §5a.

pub mod clock;
pub mod deadline_match;
pub mod rng_u64;
pub mod sleep;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
