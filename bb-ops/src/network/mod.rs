//! Network components - `wire.Send` + `wire.Recv` (the transport
//! primitives) live here per Each component
//! self-registers via `inventory::submit!`.

pub mod wire;
