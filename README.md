<div align="center">
  <img src="assets/logo.png" alt="Bytes & Brains" width="240"/>
  <h1>bytesandbrains</h1>
  <p><strong>Composable building blocks for decentralized + federated machine learning.</strong></p>
  <p>
    <a href="https://crates.io/crates/bytesandbrains"><img src="https://img.shields.io/crates/v/bytesandbrains" alt="crates.io"/></a>
    <a href="https://docs.rs/bytesandbrains"><img src="https://img.shields.io/docsrs/bytesandbrains" alt="docs.rs"/></a>
    <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="license"/>
    <img src="https://img.shields.io/badge/rust-1.86%2B-orange" alt="rust"/>
  </p>
</div>

---

`bytesandbrains` is a sans-IO Rust framework for authoring
decentralized and federated learning systems. You describe what
should run as a **Module** — a struct whose `body()` method records
your computation into an ONNX `ModelProto`. The framework partitions
the graph across BB Nodes, inlines role behavior, dispatches each op
to one of your bound runtime impls, and routes inter-Node values via
a single wire envelope.

**Status: v0.3.0** — the canonical authoring + runtime surface. See
[`docs/`](docs/) for the design specification.

## Quick start

```bash
cargo add bytesandbrains
```

```toml
[dependencies]
bytesandbrains = "0.3"
```

```rust
use std::task::{Context, Poll, Waker};

use bytesandbrains::ops::backends::cpu::CpuBackend;
use bytesandbrains::ops::placeholders::Backend as BackendSlot;
use bytesandbrains::{Compiler, Config, Graph, Module, Output};

/// A Module is your computation. The `body()` method records DSL
/// calls onto a `Graph` recorder; the compiler stamps the recorded
/// `ModelProto` with the bindings the runtime needs.
pub struct EmbeddingPipeline {
    backend: BackendSlot,
}

impl Module for EmbeddingPipeline {
    fn name(&self) -> &str { "EmbeddingPipeline" }
    fn body(&self, g: &mut Graph) {
        let batch      = g.input("batch");
        let normalized = self.backend.l2_normalize(g, batch);
        let output     = self.backend.l2_normalize(g, normalized);
        g.output("embedding", output);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Three-phase construction: author → compile → install.
    let module   = EmbeddingPipeline { backend: BackendSlot };
    let model    = module.build()?;
    let compiled = Compiler::new()
        .bind_backend::<CpuBackend>("backend")
        .compile(model)?;

    let peer_id  = bytesandbrains::PeerId::from(0u64);
    let addr     = bytesandbrains::framework::Address::empty();
    let mut node = bytesandbrains::install::install(
        peer_id,
        addr,
        compiled,
        "EmbeddingPipeline",
        Config::new(),
    )?;

    // Drive the Node's poll loop on your runtime of choice.
    let waker  = Waker::noop();
    let mut cx = Context::from_waker(waker);
    while let Poll::Ready(_steps) = node.poll(&mut cx) {}
    Ok(())
}
```

The framework is **sans-IO**: the Node is a state machine; the
caller drives `poll()` and ships outbound envelopes. There's no
`tokio` in `src/`; transport adapters live outside the core crate.

## Mental model in 60 seconds

- **Module** = your code. Implements `name()` + `body()`. Every DSL
  call (`self.backend.matmul(g, a, b)`) records a NodeProto.
- **Module::build()** returns a recorded `ModelProto`.
- **Compiler::new().bind_\<role\>::<T>("slot").compile(model)** runs
  the 18-pass canonical pipeline and stamps the result with the
  compilation passport + binding-table metadata.
- **bb::install(peer, addr, compiled, target, config)** verifies the
  passport, constructs every bound concrete via the inventory, and
  returns a ready-to-poll `Node`.
- **Engine** dispatches each op by `(domain, op_type, instance)` to
  one bound `dispatch_atomic` per instance. The wire opset is the
  one universal cross-Node transport.
- **ConcreteComponent** trait is the polymorphism contract:
  `TYPE_NAME` + `serialize` + `restore`. Derives (`bb::Concrete`,
  `bb::<Role>`) emit the inventory submission that lets the
  installer find each type by name.

See [`docs/AUTHORING_COMPONENTS.md`](docs/AUTHORING_COMPONENTS.md)
for the library-writer + extension-author walkthrough.

## Workspace

| Crate                | Notes |
|----------------------|-------|
| **`bytesandbrains`** | Facade — re-exports the six workspace crates as one surface. End users depend on this. |
| **`bb-ir`**          | Foundation: prost-generated ONNX + `bb.core` proto bindings, wire envelope, type lattice, ids, atomic-op declarations. |
| **`bb-dsl`**         | Authoring surface: `Module`, `Graph` recorder, `Output`, Contract traits, placeholders. |
| **`bb-compiler`**    | Compilation pipeline (the 18 canonical passes) + `Compiler` driver + `CompileError`. |
| **`bb-runtime`**     | Sans-IO `Engine` + `Node` + ingress + envelope codec + `<Role>Runtime` trait surfaces + inventory registry. |
| **`bb-ops`**         | Concrete components: CPU backend, wire transport, syscall implementations, role placeholders, bundled protocols. |
| **`bb-derive`**      | Proc-macros: `#[derive(bb::Concrete)]`, `#[derive(bb::<Role>)]`, `register_op!{}`, `register_protocol!{}`. |

All seven crates ship at the workspace version in lockstep, managed by `release-plz`.

## Documentation

The canonical design specification lives under [`docs/`](docs/):

- [`API_DESIGN.md`](docs/API_DESIGN.md) — Module trait + Graph DSL + install API.
- [`AUTHORING_COMPONENTS.md`](docs/AUTHORING_COMPONENTS.md) — library-writer + app-extension walkthrough.
- [`ROLES.md`](docs/ROLES.md) — the eight `<Role>Runtime` traits (Backend, Model, Index, Aggregator, Codec, DataSource, PeerSelector, Protocol).
- [`COMPILER.md`](docs/COMPILER.md) — the 18-pass compilation pipeline.
- [`ENGINE.md`](docs/ENGINE.md) — the sans-IO Engine + per-Node atomic dispatch.
- [`IR_AND_DSL.md`](docs/IR_AND_DSL.md) — how Module + Graph map onto ONNX `ModelProto`.
- [`WIRE.md`](docs/WIRE.md) — wire envelope + transport-plane mechanics.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.

## License

Dual-licensed:

- **[AGPL-3.0-or-later](LICENSE)** — for open-source use that
  satisfies the AGPL's network-use copyleft.
- **Commercial license** — for organizations that need an
  alternative to AGPL. Contact <license@bytesandbrains.com>.

The dual-licensing model matches the framework's intended use:
build open infrastructure for decentralized ML, or ship a commercial
product with a clean license.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for build/test/commit
conventions. Issues are open; external pull requests are accepted by
invitation during the v0.x stabilization period. Security
disclosures go through [`SECURITY.md`](SECURITY.md).
