fn main() {
    #[cfg(feature = "proto")]
    {
        let mut config = prost_build::Config::new();
        config.extern_path(".bb_core", "::bb_core::proto");

        let bb_core_proto_include = std::env::var("DEP_BYTESANDBRAINS_CORE_PROTO_INCLUDE").expect(
            "DEP_BYTESANDBRAINS_CORE_PROTO_INCLUDE must be set by bytesandbrains-core build script",
        );

        let protos = vec!["src/gossip/proto/gossip.proto"];
        let includes = vec![bb_core_proto_include, "src/gossip/proto/".to_string()];

        config.compile_protos(&protos, &includes).unwrap();
    }
}
