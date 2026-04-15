#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod pq;

pub use pq::{PQCode, PQDistanceTable, PQError, ProductQuantizer, SDCTable, EagerOpRef, FinishableError};
