//! `SnapshotError` — failures `Node::snapshot()` can surface.
//!
//! Snapshot capture refuses to proceed when the in-Node typed bus
//! still holds events that a restore would silently drop or re-fire
//! against stale state. Drain the bus by polling to quiescence
//! before retrying snapshot.

/// Failures `Node::snapshot()` returns instead of panicking.
#[derive(Debug)]
pub enum SnapshotError {
    /// The in-Node typed bus still carries un-drained events at the
    /// moment `snapshot()` is invoked. A restore would either
    /// silently drop them or re-fire stale infra events — neither
    /// preserves Node fidelity. Callers drive `Node::poll` until the
    /// bus is empty before retrying.
    BusNotDrained {
        /// Events still queued at snapshot time.
        queued: usize,
        /// FIFO-dropped events accumulated since the last drain.
        dropped: usize,
    },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BusNotDrained { queued, dropped } => write!(
                f,
                "bus not drained at snapshot time: queued={queued} dropped={dropped}",
            ),
        }
    }
}

impl std::error::Error for SnapshotError {}
