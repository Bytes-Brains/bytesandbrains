fn main() {
    #[cfg(feature = "proto")]
    {
        let mut config = prost_build::Config::new();
        config.extern_path(".bb_core", "::bb_core::proto");

        let mut protos = Vec::new();
        let mut includes = vec!["../bb-core/src/proto/"];

        protos.push("src/gossip/proto/gossip.proto");
        includes.push("src/gossip/proto/");

        config.compile_protos(&protos, &includes).unwrap();
    }
}
