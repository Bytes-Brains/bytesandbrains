//! `peers` syscall category — DAG-mutable `AddressBook` ops.
//!
//! `ai.bytesandbrains.address_book`-domain custom ops emitted by
//! [`bb_derive::register_op`] — `Insert`, `InsertMany`, `Lookup`.
//! Single-address `Insert` is the common transport-learn case
//! (observed source from one envelope); `InsertMany` carries the
//! full advertised bag from identity-announcement sites; `Lookup`
//! emits the full ordered slice so callers that want a single
//! address pick one explicitly.
//!
//! Spec: `docs/ADDRESSING.md` — "DAG-mutable AddressBook".

pub mod insert;
pub mod insert_many;
pub mod lookup;
