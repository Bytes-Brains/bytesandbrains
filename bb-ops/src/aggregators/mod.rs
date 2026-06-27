//! Concrete `Aggregator` Contract implementations the framework
//! ships out of the box. Each impl bridges the user-facing
//! `bb_runtime::contracts::Aggregator` trait to the engine's
//! `dispatch_atomic` path through the `bb_derive::Aggregator`
//! macro and self-registers via `inventory::submit!`.

pub mod fedavg;

pub use fedavg::{FedAvg, FedAvgMeta};
