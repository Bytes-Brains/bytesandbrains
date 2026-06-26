//! `RestoreError`
//!
//! Returned by `Node::restore(snapshot)` when the snapshot can't be
//! reconciled with the current Node's installed modules + wire types.
//! Distinct from [`crate::concrete::RestoreError`], which is the
//! per-component deserialization error type that this enum wraps.

/// Errors surfaced by `Node::restore` (the public surface arrives in
/// the commit that lands `src/node.rs`).
#[derive(Debug)]
pub enum RestoreError {
    /// Snapshot-level invariant violation (incarnation mismatch,
    /// component table mismatch, graph not found, etc.).
    SnapshotMismatch(String),

    /// One of the snapshotted components failed to deserialize via
    /// its registered `RestoreFn`.
    ComponentRestoreFailed {
        /// `ConcreteComponent::TYPE_NAME` of the failing component.
        type_name: String,
        /// The per-component error.
        source: crate::concrete::RestoreError,
    },

    /// Snapshot's `spec_version` doesn't match the live Node's
    /// `CURRENT_SNAPSHOT_SPEC_VERSION`. Bumps to the spec version
    /// happen when the `FrameworkSnapshot` shape changes in a way
    /// older code cannot soundly restore.
    SpecVersionMismatch {
        /// Version stamped on the snapshot.
        got: u32,
        /// Version this build supports.
        expected: u32,
    },
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SnapshotMismatch(reason) => write!(f, "snapshot mismatch: {reason}"),
            Self::ComponentRestoreFailed { type_name, source } => {
                write!(f, "component {type_name} restore failed: {source}",)
            }
            Self::SpecVersionMismatch { got, expected } => write!(
                f,
                "snapshot spec_version mismatch: got={got}, expected={expected}",
            ),
        }
    }
}

impl std::error::Error for RestoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ComponentRestoreFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

