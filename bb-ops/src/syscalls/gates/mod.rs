//! Framework-inserted TX/RX gate ops. Each gate is a compiler-
//! inserted graph op that consults a framework primitive
//! (`PeerGovernor`, `BackoffTable`, `InboundDedup`) and either
//! forwards its envelope/trigger downstream or drops + emits a
//! lifecycle event.
//!
//! TX-side chain (around `wire::Send`):
//!     `wire::Send` → `PeerHealthGateTx` → `BackoffGateTx` → outbound
//!
//! RX-side chain (around `wire::Recv`):
//!     `wire::Recv` → `DedupGateRx` → `PeerHealthGateRx` →
//!     `BackoffGateRx` → user consumer

pub mod backoff_rx;
pub mod backoff_tx;
pub mod dedup_rx;
pub mod peer_health_rx;
pub mod peer_health_tx;
