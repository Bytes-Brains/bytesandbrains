use std::fmt;

/// Identifies an in-flight operation.
///
/// `OpId` provides a unique identifier for asynchronous operations across the
/// system. It is used to:
/// - Track index operations (search, add, remove, train)
/// - Correlate distributed search operations across the network
/// - Enable progress tracking and cancellation
///
/// The ID is unique within the lifetime of the component that issued it and
/// can be used to correlate log messages, metrics, or external tracking with
/// a specific operation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct OpId(pub u64);

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpId({})", self.0)
    }
}

/// A handle to an in-flight operation.
///
/// `OpRef` provides a unified API for both local and networked implementations:
///
/// - **Local implementations** can complete the work eagerly inside the method
///   that creates the handle, making [`is_finished`](OpRef::is_finished) return
///   `true` immediately and [`finish`](OpRef::finish) simply unwrap the
///   pre-computed result.
///
/// - **Networked implementations** can return a handle that tracks a remote RPC.
///   The caller polls [`is_finished`](OpRef::is_finished) (or awaits a
///   notification) and calls [`finish`](OpRef::finish) once the remote side has
///   responded.
///
/// This design lets generic code work identically regardless of whether the
/// underlying operation is in-process or across a network boundary.
pub trait OpRef {
    /// Metadata describing the operation (e.g. query parameters, batch size).
    type Info;

    /// Runtime statistics collected during the operation (e.g. distances
    /// computed, nodes visited).
    type Stats;

    /// The successful outcome of the operation.
    type Result;

    /// The error type returned when the operation fails.
    type Error;

    /// Returns the unique identifier assigned to this operation.
    /// Can represent some sort of ID for the entry in an eager operation
    /// context is up to the implementation (for example a entry tag for a local
    /// index)
    fn id(&self) -> &OpId;

    /// Returns metadata describing the operation.
    ///
    /// Returns `None` if the operation no longer exists (e.g., already finished
    /// and removed from tracking).
    fn info(&self) -> Option<Self::Info>;

    /// Returns runtime statistics collected so far.
    ///
    /// Returns `None` if the operation no longer exists.
    fn stats(&self) -> Option<Self::Stats>;

    /// Returns `true` once the operation has completed (successfully or not).
    ///
    /// For local implementations this is typically `true` immediately after
    /// creation. For networked implementations the caller should poll this or
    /// await a notification before calling [`finish`](OpRef::finish).
    fn is_finished(&self) -> bool;

    /// Consumes the pending operation and returns its result.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` if the operation failed.
    fn finish(&mut self) -> Result<Self::Result, Self::Error>;
}

/// An immediate no-op operation reference.
///
/// Useful for protocols that don't support certain operations (e.g., bootstrap)
/// or for stub implementations. Always returns `is_finished() == true` and
/// `finish()` returns `Ok(())`.
pub struct NoopOpRef {
    id: OpId,
}

impl NoopOpRef {
    /// Create a new no-op operation reference with the given ID.
    pub fn new(id: u64) -> Self {
        Self { id: OpId(id) }
    }
}

impl OpRef for NoopOpRef {
    type Info = ();
    type Stats = ();
    type Result = ();
    type Error = std::convert::Infallible;

    fn id(&self) -> &OpId {
        &self.id
    }

    fn info(&self) -> Option<Self::Info> {
        Some(())
    }

    fn stats(&self) -> Option<Self::Stats> {
        Some(())
    }

    fn is_finished(&self) -> bool {
        true
    }

    fn finish(&mut self) -> Result<Self::Result, Self::Error> {
        Ok(())
    }
}
