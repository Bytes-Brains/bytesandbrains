pub mod gossip_proto {
    include!(concat!(env!("OUT_DIR"), "/gossip_proto.rs"));
}

mod conversions;

#[allow(unused_imports)]
pub use gossip_proto::*;
