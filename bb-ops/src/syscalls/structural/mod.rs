//! Structural syscalls - `PassThrough`, `Tee`, `Constant`. These
//! components self-register via `inventory::submit!`; the module
//! tree just declares each sub-module so the linker pulls them in.

pub mod constant;
pub mod pass_through;
pub mod tee;
