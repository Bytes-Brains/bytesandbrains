//! Network components - `wire.Send` + `wire.Recv` (the transport
//! primitives). Each component self-registers via
//! `inventory::submit!`.

pub mod wire;
