use std::time::Duration;

use bb_core::pending_requests::RequestId;
use bb_core::proto::ProtoConversionError;
use bb_core::{Address, Peer};

use super::gossip_proto::*;
use crate::gossip::config::{GossipConfig, GossipMode};
use crate::gossip::exchange::RandomizedExchange;
use crate::gossip::peer::AgePeer;
use crate::gossip::selector::RandomizedSelectorMode;
use crate::gossip::GossipMessage;

// --- GossipMode ---

impl From<GossipMode> for GossipModeProto {
    fn from(mode: GossipMode) -> Self {
        match mode {
            GossipMode::PushPull => GossipModeProto::PushPull,
            GossipMode::Push => GossipModeProto::Push,
            GossipMode::Pull => GossipModeProto::Pull,
        }
    }
}

impl From<GossipModeProto> for GossipMode {
    fn from(proto: GossipModeProto) -> Self {
        match proto {
            GossipModeProto::PushPull => GossipMode::PushPull,
            GossipModeProto::Push => GossipMode::Push,
            GossipModeProto::Pull => GossipMode::Pull,
        }
    }
}

// --- RandomizedSelectorMode ---

impl From<RandomizedSelectorMode> for RandomizedSelectorModeProto {
    fn from(selector: RandomizedSelectorMode) -> Self {
        match selector {
            RandomizedSelectorMode::Tail => RandomizedSelectorModeProto::Tail,
            RandomizedSelectorMode::UniformRandom => RandomizedSelectorModeProto::UniformRandom,
        }
    }
}

impl From<RandomizedSelectorModeProto> for RandomizedSelectorMode {
    fn from(proto: RandomizedSelectorModeProto) -> Self {
        match proto {
            RandomizedSelectorModeProto::Tail => RandomizedSelectorMode::Tail,
            RandomizedSelectorModeProto::UniformRandom => RandomizedSelectorMode::UniformRandom,
        }
    }
}


impl From<Duration> for DurationProto {
    fn from(d: Duration) -> Self {
        DurationProto {
            millis: d.as_millis() as u64,
        }
    }
}

impl From<DurationProto> for Duration {
    fn from(proto: DurationProto) -> Self {
        Duration::from_millis(proto.millis)
    }
}


impl From<&GossipConfig> for GossipConfigProto {
    fn from(config: &GossipConfig) -> Self {
        GossipConfigProto {
            view_size: config.view_size as u32,
            mode: GossipModeProto::from(config.mode).into(),
            max_concurrent_requests: config.max_concurrent_requests as u32,
            retry_time: Some(config.retry_time.into()),
            request_timeout: Some(config.request_timeout.into()),
            poll_interval: Some(config.poll_interval.into()),
        }
    }
}

impl From<GossipConfigProto> for GossipConfig {
    fn from(proto: GossipConfigProto) -> Self {
        let defaults = GossipConfig::default();
        GossipConfig {
            view_size: proto.view_size as usize,
            mode: GossipModeProto::try_from(proto.mode)
                .unwrap_or(GossipModeProto::PushPull)
                .into(),
            max_concurrent_requests: if proto.max_concurrent_requests == 0 {
                defaults.max_concurrent_requests
            } else {
                proto.max_concurrent_requests as usize
            },
            retry_time: proto
                .retry_time
                .map(Duration::from)
                .unwrap_or(defaults.retry_time),
            request_timeout: proto
                .request_timeout
                .map(Duration::from)
                .unwrap_or(defaults.request_timeout),
            poll_interval: proto
                .poll_interval
                .map(Duration::from)
                .unwrap_or(defaults.poll_interval),
        }
    }
}


impl From<&RandomizedExchange> for RandomizedExchangeConfigProto {
    fn from(exchange: &RandomizedExchange) -> Self {
        RandomizedExchangeConfigProto {
            healing: exchange.healing as u32,
            swap: exchange.swap as u32,
        }
    }
}

impl From<RandomizedExchangeConfigProto> for RandomizedExchange {
    fn from(proto: RandomizedExchangeConfigProto) -> Self {
        RandomizedExchange {
            healing: proto.healing as usize,
            swap: proto.swap as usize,
        }
    }
}

impl<A: Address> From<AgePeer<A>> for GossipPeerProto {
    fn from(gp: AgePeer<A>) -> Self {
        GossipPeerProto {
            peer: Some(gp.peer.into()),
            age: gp.age,
        }
    }
}

impl<A: Address> TryFrom<GossipPeerProto> for AgePeer<A> {
    type Error = ProtoConversionError;

    fn try_from(proto: GossipPeerProto) -> Result<Self, ProtoConversionError> {
        let peer_proto = proto.peer.ok_or_else(|| {
            ProtoConversionError::ConversionFailed("Missing peer in GossipPeerProto".to_string())
        })?;
        let peer: Peer<A> = peer_proto.try_into()?;
        Ok(AgePeer {
            peer,
            age: proto.age,
        })
    }
}

// --- GossipMessage (generic over any P convertible to/from GossipPeerProto) ---

impl<P> From<GossipMessage<P>> for GossipMessageProto
where
    P: Clone + Into<GossipPeerProto>,
{
    fn from(msg: GossipMessage<P>) -> Self {
        match msg {
            GossipMessage::Request { request_id, mode, view } => GossipMessageProto {
                message: Some(gossip_message_proto::Message::Request(
                    GossipRequestProto {
                        request_id: request_id.0,
                        mode: GossipModeProto::from(mode).into(),
                        view: view.into_iter().map(|p| p.into()).collect(),
                    },
                )),
            },
            GossipMessage::Response { request_id, view } => GossipMessageProto {
                message: Some(gossip_message_proto::Message::Response(
                    GossipResponseProto {
                        request_id: request_id.0,
                        view: view.into_iter().map(|p| p.into()).collect(),
                    },
                )),
            },
        }
    }
}

impl<P> TryFrom<GossipMessageProto> for GossipMessage<P>
where
    P: Clone + TryFrom<GossipPeerProto, Error = ProtoConversionError>,
{
    type Error = ProtoConversionError;

    fn try_from(proto: GossipMessageProto) -> Result<Self, ProtoConversionError> {
        let inner = proto.message.ok_or_else(|| {
            ProtoConversionError::ConversionFailed(
                "Missing message in GossipMessageProto".to_string(),
            )
        })?;
        match inner {
            gossip_message_proto::Message::Request(req) => {
                let request_id = RequestId(req.request_id);
                let mode: GossipMode = GossipModeProto::try_from(req.mode)
                    .unwrap_or(GossipModeProto::PushPull)
                    .into();
                let view: Result<Vec<P>, _> =
                    req.view.into_iter().map(|gp| gp.try_into()).collect();
                Ok(GossipMessage::Request {
                    request_id,
                    mode,
                    view: view?,
                })
            }
            gossip_message_proto::Message::Response(resp) => {
                let request_id = RequestId(resp.request_id);
                let view: Result<Vec<P>, _> =
                    resp.view.into_iter().map(|gp| gp.try_into()).collect();
                Ok(GossipMessage::Response { request_id, view: view? })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gossip_peer(addr: &str, age: u32) -> AgePeer<String> {
        AgePeer::new(addr.to_string(), age)
    }

    #[test]
    fn gossip_mode_roundtrip() {
        for mode in [GossipMode::Push, GossipMode::Pull, GossipMode::PushPull] {
            let proto: GossipModeProto = mode.into();
            let recovered: GossipMode = proto.into();
            assert_eq!(mode, recovered);
        }
    }

    #[test]
    fn selector_mode_roundtrip() {
        for selector in [
            RandomizedSelectorMode::Tail,
            RandomizedSelectorMode::UniformRandom,
        ] {
            let proto: RandomizedSelectorModeProto = selector.into();
            let recovered: RandomizedSelectorMode = proto.into();
            assert_eq!(selector, recovered);
        }
    }

    #[test]
    fn gossip_config_roundtrip() {
        let config = GossipConfig::default();
        let proto: GossipConfigProto = (&config).into();
        let recovered: GossipConfig = proto.into();
        assert_eq!(config.view_size, recovered.view_size);
        assert_eq!(config.mode, recovered.mode);
        assert_eq!(config.max_concurrent_requests, recovered.max_concurrent_requests);
        assert_eq!(config.retry_time, recovered.retry_time);
        assert_eq!(config.request_timeout, recovered.request_timeout);
        assert_eq!(config.poll_interval, recovered.poll_interval);
    }

    #[test]
    fn randomized_exchange_config_roundtrip() {
        let exchange = RandomizedExchange::default();
        let proto: RandomizedExchangeConfigProto = (&exchange).into();
        let recovered: RandomizedExchange = proto.into();
        assert_eq!(exchange.healing, recovered.healing);
        assert_eq!(exchange.swap, recovered.swap);
    }

    #[test]
    fn gossip_peer_roundtrip() {
        let gp = make_gossip_peer("10.0.0.1:5000", 3);
        let proto: GossipPeerProto = gp.clone().into();
        let recovered: AgePeer<String> = proto.try_into().unwrap();
        assert_eq!(gp.peer.peer_id, recovered.peer.peer_id);
        assert_eq!(gp.age, recovered.age);
    }

    #[test]
    fn gossip_peer_missing_peer_fails() {
        let proto = GossipPeerProto {
            peer: None,
            age: 0,
        };
        let result: Result<AgePeer<String>, _> = proto.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn gossip_message_request_roundtrip() {
        let msg: GossipMessage<AgePeer<String>> = GossipMessage::Request {
            request_id: RequestId::new(42),
            mode: GossipMode::PushPull,
            view: vec![
                make_gossip_peer("10.0.0.1:5000", 0),
                make_gossip_peer("10.0.0.2:5000", 2),
            ],
        };
        let proto: GossipMessageProto = msg.into();
        let recovered: GossipMessage<AgePeer<String>> = proto.try_into().unwrap();
        match recovered {
            GossipMessage::Request { request_id, mode, view } => {
                assert_eq!(request_id.0, 42);
                assert_eq!(mode, GossipMode::PushPull);
                assert_eq!(view.len(), 2);
            }
            _ => panic!("Expected Request"),
        }
    }

    #[test]
    fn gossip_message_response_roundtrip() {
        let msg: GossipMessage<AgePeer<String>> = GossipMessage::Response {
            request_id: RequestId::new(99),
            view: vec![make_gossip_peer("10.0.0.5:5000", 1)],
        };
        let proto: GossipMessageProto = msg.into();
        let recovered: GossipMessage<AgePeer<String>> = proto.try_into().unwrap();
        match recovered {
            GossipMessage::Response { request_id, view } => {
                assert_eq!(request_id.0, 99);
                assert_eq!(view.len(), 1);
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn gossip_message_empty_message_fails() {
        let proto = GossipMessageProto { message: None };
        let result: Result<GossipMessage<AgePeer<String>>, _> = proto.try_into();
        assert!(result.is_err());
    }
}
