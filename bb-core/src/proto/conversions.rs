use super::{DistanceProto, PeerProto, ProtoConversionError, TensorProto, DATA_TYPE_FLOAT};
use crate::{
    address::{Address, AddressBook},
    embedding::{F32Distance, F32Embedding, Embedding},
    peer::Peer,
    peer_id::PeerId,
};

impl From<F32Distance> for DistanceProto {
    fn from(dist: F32Distance) -> Self {
        DistanceProto {
            data_type: DATA_TYPE_FLOAT,
            float_data: vec![dist.0],
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
        }
    }
}

impl TryFrom<DistanceProto> for F32Distance {
    type Error = ProtoConversionError;

    fn try_from(proto: DistanceProto) -> Result<Self, ProtoConversionError> {
        if proto.data_type != DATA_TYPE_FLOAT {
            return Err(ProtoConversionError::InvalidDataType {
                expected: DATA_TYPE_FLOAT,
                actual: proto.data_type,
            });
        }
        if proto.float_data.len() != 1 {
            return Err(ProtoConversionError::ConversionFailed(
                format!("Expected 1 float in DistanceProto, got {}", proto.float_data.len()),
            ));
        }
        Ok(F32Distance(proto.float_data[0]))
    }
}

impl<const L: usize> From<F32Embedding<L>> for TensorProto {
    fn from(embedding: F32Embedding<L>) -> Self {
        TensorProto {
            dims: vec![L as i64],
            data_type: DATA_TYPE_FLOAT,
            float_data: embedding.0.to_vec(),
            ..Default::default()
        }
    }
}

impl<const L: usize> TryFrom<TensorProto> for F32Embedding<L> {
    type Error = ProtoConversionError;

    fn try_from(proto: TensorProto) -> Result<Self, ProtoConversionError> {
        if proto.data_type != DATA_TYPE_FLOAT {
            return Err(ProtoConversionError::InvalidDataType {
                expected: DATA_TYPE_FLOAT,
                actual: proto.data_type,
            });
        }

        let expected_dims = vec![L as i64];
        if proto.dims != expected_dims {
            return Err(ProtoConversionError::InvalidTensorShape {
                expected: expected_dims,
                actual: proto.dims,
            });
        }

        if proto.float_data.len() != L {
            return Err(ProtoConversionError::ConversionFailed(
                format!("Expected {} floats in TensorProto, got {}", L, proto.float_data.len()),
            ));
        }

        Ok(F32Embedding::from_slice(&proto.float_data))
    }
}

impl From<PeerId> for Vec<u8> {
    fn from(peer_id: PeerId) -> Self {
        peer_id.to_bytes()
    }
}

impl<A: Address> From<&Peer<A>> for PeerProto {
    fn from(peer: &Peer<A>) -> Self {
        PeerProto {
            peer_id: peer.peer_id.to_bytes(),
            addresses: peer.addresses.iter().map(|a| a.to_string()).collect(),
        }
    }
}

impl<A: Address> From<Peer<A>> for PeerProto {
    fn from(peer: Peer<A>) -> Self {
        PeerProto::from(&peer)
    }
}

impl<A: Address> TryFrom<PeerProto> for Peer<A> {
    type Error = ProtoConversionError;

    fn try_from(proto: PeerProto) -> Result<Self, ProtoConversionError> {
        let peer_id = PeerId::from_slice(&proto.peer_id);
        let max_size = proto.addresses.len();
        let addresses = addresses_from_proto(proto.addresses, max_size)?;
        Ok(Peer { peer_id, addresses })
    }
}

pub fn addresses_to_proto<A: Address>(addresses: &AddressBook<A>) -> Vec<String> {
    addresses.iter().map(|a| a.to_string()).collect()
}

pub fn addresses_from_proto<A: Address>(
    addresses: Vec<String>,
    max_size: usize,
) -> Result<AddressBook<A>, ProtoConversionError> {
    if addresses.is_empty() {
        return Err(ProtoConversionError::ConversionFailed(
            "Empty addresses".to_string(),
        ));
    }

    let first: A = addresses[0].parse().map_err(|_| {
        ProtoConversionError::ConversionFailed(
            format!("Failed to parse address: {}", addresses[0]),
        )
    })?;

    let mut book = AddressBook::new(first, max_size);

    for addr_str in addresses.into_iter().skip(1) {
        let addr: A = addr_str.parse().map_err(|_| {
            ProtoConversionError::ConversionFailed(
                format!("Failed to parse address: {}", addr_str),
            )
        })?;
        book.seen(addr);
    }

    Ok(book)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_distance_proto_roundtrip() {
        let dist = F32Distance(42.5);
        let proto: DistanceProto = dist.into();
        let recovered: F32Distance = proto.try_into().unwrap();
        assert_eq!(dist, recovered);
    }

    #[test]
    fn test_f32_embedding_proto_roundtrip() {
        let embedding = F32Embedding::<4>([1.0, 2.0, 3.0, 4.0]);
        let proto: TensorProto = embedding.clone().into();
        let recovered: F32Embedding<4> = proto.try_into().unwrap();
        assert_eq!(embedding, recovered);
    }

    #[test]
    fn test_peer_proto_roundtrip() {
        let peer_id = PeerId::from_data("test-peer");
        let addresses = AddressBook::new("192.168.1.1:8080".to_string(), 5);
        let peer = Peer::new(peer_id, addresses);

        let proto: PeerProto = peer.clone().into();
        let recovered: Peer<String> = proto.try_into().unwrap();

        assert_eq!(peer.peer_id, recovered.peer_id);
        assert_eq!(peer.addresses.first(), recovered.addresses.first());
    }

    #[test]
    fn test_peer_proto_multi_address_roundtrip() {
        let peer_id = PeerId::from_data("test-peer");
        let mut addresses = AddressBook::new("addr1".to_string(), 5);
        addresses.seen("addr2".to_string());
        addresses.seen("addr3".to_string());
        let peer = Peer::new(peer_id, addresses);

        let proto: PeerProto = peer.clone().into();
        assert_eq!(proto.addresses.len(), 3);

        let recovered: Peer<String> = proto.try_into().unwrap();
        assert_eq!(recovered.addresses.len(), 3);
    }

    #[test]
    fn test_peer_proto_empty_addresses_fails() {
        let proto = PeerProto {
            peer_id: vec![0u8; 64],
            addresses: vec![],
        };
        let result: Result<Peer<String>, _> = proto.try_into();
        assert!(result.is_err());
    }
}
