//! Foundation `SlotValue` impls - `PeerIdValue`, `WireReqIdValue`,
//! `TriggerValue`, `BytesValue`, `CommandIdValue`. The polymorphism
//! layer every syscall component consumes; kept in bb-runtime so
//! components in bb-ops link against a single canonical source.

pub mod values;
