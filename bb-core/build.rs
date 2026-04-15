fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_include = format!("{manifest_dir}/src/proto");
    println!("cargo:include={proto_include}");
    println!("cargo:rerun-if-changed=src/proto");

    #[cfg(feature = "proto")]
    {
        prost_build::compile_protos(
            &["src/proto/onnx-ml.proto", "src/proto/bb_core.proto"],
            &["src/proto/"],
        )
        .unwrap();
    }
}
