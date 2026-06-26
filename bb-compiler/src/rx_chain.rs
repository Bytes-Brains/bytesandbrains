//! RX-chain head bookkeeping shared by the RX gate-insertion passes.
//!
//! Each synthesized `wire::Recv` carries a `RX_CHAIN_HEAD_KEY`
//! metadata entry naming the currently-effective output that
//! downstream consumers read from. Initial value is the Recv's own
//! `output[0]`; each RX gate pass replaces it with the gate's
//! output. Successive passes consult this to chain themselves in
//! order: `Recv → DedupGateRx → PeerHealthGateRx → BackoffGateRx`.

use bb_ir::proto::onnx::{NodeProto, StringStringEntryProto};

/// Metadata key on a `wire::Recv` carrying its current RX-chain head.
pub const RX_CHAIN_HEAD_KEY: &str = "ai.bytesandbrains.rx_chain_head";

/// Current RX-chain head for `recv` - the output name downstream
/// consumers read from. Defaults to `recv.output[0]` when no gate
/// has yet been inserted.
pub fn rx_chain_head(recv: &NodeProto) -> String {
    recv.metadata_props
        .iter()
        .find(|p| p.key == RX_CHAIN_HEAD_KEY)
        .map(|p| p.value.clone())
        .unwrap_or_else(|| recv.output.first().cloned().unwrap_or_default())
}

/// Set the RX-chain head metadata on `recv` to `new_head`.
pub fn set_rx_chain_head(recv: &mut NodeProto, new_head: &str) {
    if let Some(existing) = recv
        .metadata_props
        .iter_mut()
        .find(|p| p.key == RX_CHAIN_HEAD_KEY)
    {
        existing.value = new_head.to_string();
        return;
    }
    recv.metadata_props.push(StringStringEntryProto {
        key: RX_CHAIN_HEAD_KEY.to_string(),
        value: new_head.to_string(),
    });
}
