//! Timer + trigger source ops - After, Interval, Pulse,
//! OnTrigger, EventSource.
//!
//! Spec: Sub-A in `docs/IR_AND_DSL.md` §5a.

pub mod after;
pub mod event_source;
pub mod interval;
pub mod on_trigger;
pub mod pulse;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
