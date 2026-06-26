//! Engine-internal IDs. Wire/IR IDs come from `bb_ir::ids`
//! re-exported here for a single import surface.
//!
//! Each integer ID is a `#[repr(transparent)]` newtype so the
//! borrow checker rejects cross-type substitution.

use std::fmt;

pub use bb_ir::ids::{ComponentTag, OpsetId, PeerId, RequestId};

// --- Macro helpers ----------------------------------------------

macro_rules! u64_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
            serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Construct from an explicit value.
            pub const fn new(inner: u64) -> Self { Self(inner) }

            /// Inner value accessor.
            pub const fn as_u64(self) -> u64 { self.0 }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl From<u64> for $name {
            fn from(inner: u64) -> Self { Self(inner) }
        }
    };
}

macro_rules! u32_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
            serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            /// Construct from an explicit value.
            pub const fn new(inner: u32) -> Self { Self(inner) }

            /// Inner value accessor.
            pub const fn as_u32(self) -> u32 { self.0 }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl From<u32> for $name {
            fn from(inner: u32) -> Self { Self(inner) }
        }
    };
}

// --- Engine-internal integer IDs --------------------------------

u64_id! {
    /// IR-level value site. Names a slot inside a `GraphProto`'s
    /// flow that the engine fills with a slot value.
    NodeSiteId
}

u64_id! {
    /// Op handle within a graph. Positional refs encode
    /// `(graph_idx << 32) | node_idx`; one indexed lookup per
    /// invoke, no HashMap probe.
    OpRef
}

impl OpRef {
    /// Pack a `(graph_idx, node_idx)` into one `OpRef`.
    pub const fn pack(graph_idx: u32, node_idx: u32) -> Self {
        Self::new(((graph_idx as u64) << 32) | (node_idx as u64))
    }

    /// Unpack a positional `OpRef`. Globally-counter-minted refs
    /// have a zero high half.
    pub const fn split(self) -> (u32, u32) {
        let v = self.as_u64();
        ((v >> 32) as u32, v as u32)
    }
}

u64_id! {
    /// Per-execution identifier; survives async completions and
    /// cross-Node wire hops.
    ExecId
}

u64_id! {
    /// Async-dispatch command id finalized via
    /// `ctx.complete_command(cmd, outputs)`.
    CommandId
}

u32_id! {
    /// Dense per-Node component instance handle.
    ComponentRef
}

