fn main() {
    #[cfg(feature = "proto")]
    {
        prost_build::compile_protos(
            &["src/proto/onnx-ml.proto", "src/proto/bb_core.proto"],
            &["src/proto/"],
        )
        .unwrap();
    }
}
