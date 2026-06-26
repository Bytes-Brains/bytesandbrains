//! `composite` opset - `Bundle` + `Unbundle`. Pack N typed slot values
//! into one wire-eligible `CompositeValue` envelope, ship through a
//! single port (so single-port DAG semantics hold), and decompose back
//! into per-child outputs on the receiver. Domain shared with
//! `TYPE_COMPOSITE` (`ai.bytesandbrains.composite`).

pub mod bundle;
pub mod unbundle;
