# Heterogeneous deployment

Two patterns are supported. Pick the one that fits your app; they
compose if you need both.

| Pattern | When to use | Cost |
|---|---|---|
| **Form 1 — placeholder authoring** | Per-device binaries can be built independently. Each binary owns its own concrete bindings. | Zero. Vanilla Rust DCE handles the per-binary stripping. |
| **Form 2 — shared compiled model** | The build host owns Component configuration (build-host secrets, pre-validated graphs). Cross-language deployment. | One `Compiler::new().bind_*().compile(model)` on the build host, ship the prost-encoded `ModelProto` bytes, each device `prost::Message::decode` + `bb::install(...)`. |

---

## Form 1 — placeholder authoring

The shared crate holds only role placeholder structs (`Backend`,
`Index`, `Model`, ...). Each per-device bin crate compiles its own
`ModelProto` from the shared `Module`, binds its concretes via
`Compiler::new().bind_<role>::<T>("slot")`, and installs through
`bb::install(...)`. **The compiler never sees Components a binary
doesn't use** — they are not in that binary's dependency closure.

### Layout

```
my_app/
├── Cargo.toml               # workspace
├── shared/                  # Module struct + custom message types only
│   ├── Cargo.toml
│   └── src/lib.rs
├── server/                  # bin crate; depends on shared + server_components
│   └── src/main.rs
└── client/                  # bin crate; depends on shared + client_components
    └── src/main.rs
```

### `shared/src/lib.rs`

```rust
use bytesandbrains::{Backend, Graph, Index, Module};

pub struct FedDemo {
    pub backend: Backend,
    pub index: Index,
}

impl Module for FedDemo {
    fn name(&self) -> &str { "fed_demo" }
    fn body(&self, g: &mut Graph) {
        // Recording-time methods on placeholders. The compiler
        // partitions by wire-op boundaries; each partition only
        // references the placeholders it actually uses.
        let _ = g.input("request");
    }
}
```

### `server/src/main.rs`

```rust
use bytesandbrains::compiler::Compiler;
use bytesandbrains::framework::Address;
use bytesandbrains::ids::PeerId;
use bytesandbrains::{install, Config};
use server_components::{ServerBackend, ServerIndex};
use shared::FedDemo;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One ModelProto from the shared Module. Compiler runs the
    // canonical 18-pass pipeline, emits ONE ModelProto whose
    // `functions[]` carries every partition root.
    let model = FedDemo {
        backend: Backend,
        index: Index,
    }
    .build()?;

    let compiled = Compiler::new()
        .bind_backend::<ServerBackend>("compute")
        .bind_index::<ServerIndex>("primary_index")
        .compile(model)?;

    // Pick the target function this binary runs. The bare
    // function name suffices; install also accepts the
    // content-hash-suffixed form the compiler emits.
    let _node = install(
        PeerId::from(1u64),
        Address::empty(),
        compiled,
        "fed_demo_server",
        Config::new(),
    )?;

    // The client's concrete types are never `use`d in this file.
    // The linker never sees them. The server binary does not ship
    // the client's kernels.
    Ok(())
}
```

### What gets stripped

When you `cargo build -p server`, the linker only sees
`server_components::ServerBackend` etc. The shared crate's
placeholder structs are zero-sized — no concrete code path. There
is nothing to strip, because the unused side's Components were
never compiled in.

You can verify with `cargo bloat` or `nm`:

```bash
nm target/release/server | grep -i ClientBackend
# (no output)
```

### Cross-compile

Each binary is a normal Rust crate. Standard
`cargo build --target=aarch64-unknown-linux-gnu` for ARM, etc.
The framework's `[dependencies]` (`prost`, `serde`, `bincode`,
`concurrent-queue`, `atomic-waker`, `getrandom`, `tracing`) are
pure-Rust and cross-compile cleanly to any `std` target.

### Wire-type compatibility

Wire-eligibility is the `SlotValue` blanket: any type satisfying
`Clone + serde::Serialize + serde::DeserializeOwned + Send + Sync
+ 'static` rides the wire by construction. There is no
registration step. The shared crate declares any custom message
type once + derives serde on it; both binaries pick up the
serde-compatible encoding automatically. Wire encoding is bincode-
derived, so two binaries built from the same shared types always
agree.

### Known limitation: same-type multiple placeholders

Two `Backend` placeholders inside one Module collapse to a single
slot. The recording layer keys slots by `(TypeId, *const ())` of
the placeholder, but placeholder structs are zero-sized — and Rust
gives all ZST instances the same address. Distinct placeholder
*types* (e.g. `Backend` + `Index`) are fine, because the TypeId
discriminator separates them.

Workaround: if you need two role slots of the same trait, split
the Module by role across two structs or wrap one in a per-binary
newtype. The Form 1 workspace pattern above naturally satisfies
this (each binary's Module exposes one slot per role).

A `tests/two_placeholders.rs` regression test pins this behavior
so a future const-generic-discriminated-placeholder change can't
silently regress.

---

## Form 2 — shared compiled model

The build host runs the shared Module through `Module::build()`
followed by `Compiler::new().bind_*().compile(model)` ONCE,
producing ONE `ModelProto` that carries every partition's
function in `model.functions[]` plus the compilation passport
and per-target binding metadata. Each device receives the same
prost-encoded `ModelProto` bytes and selects its partition by
naming the function at install time.

This is the right pattern when Component configuration must
happen on the build host (build-host secrets, pre-validated
graphs, cross-language deployment).

### Build host

```rust
use bytesandbrains::compiler::Compiler;
use prost::Message;
use shared::FedDemo;
use shared_components::{Aggregator, Backend, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = FedDemo {
        backend: bytesandbrains::Backend,
        index: bytesandbrains::Index,
    }
    .build()?;

    // Bind every concrete the deployment needs across ALL targets.
    // The compiler emits one ModelProto whose `functions[]` carries
    // every partition root + the binding metadata for each target.
    let compiled = Compiler::new()
        .bind_backend::<Backend>("compute")
        .bind_index::<Index>("primary_index")
        .bind_aggregator::<Aggregator>("aggregator")
        .compile(model)?;

    // Ship the prost-encoded bytes (single wire artifact).
    std::fs::write("dist/fed_demo.model", compiled.encode_to_vec())?;
    Ok(())
}
```

### Device binary

```rust
use bytesandbrains::framework::Address;
use bytesandbrains::ids::PeerId;
use bytesandbrains::proto::onnx::ModelProto;
use bytesandbrains::{install, Config};
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode the prost-encoded ModelProto the build host shipped.
    let bytes = std::fs::read("dist/fed_demo.model")?;
    let compiled = ModelProto::decode(bytes.as_slice())?;

    // Pick the function this device runs by name. install()
    // verifies the compilation passport, walks the binding
    // metadata for the chosen target, constructs each declared
    // concrete via the inventory `construct_fn`, and brings the
    // Node up. Concretes the binary doesn't link are unreachable
    // for any target they don't appear in.
    let _node = install(
        PeerId::from(1u64),
        Address::empty(),
        compiled,
        "fed_demo_server",
        Config::new(),
    )?;
    Ok(())
}
```

### Missing-provider errors are exhaustive

`bb::install(...)` returns `InstallError::UnregisteredConcrete`
naming the first `TYPE_NAME` the artifact references that no
linked `inventory::submit!` carrier registers. Link the missing
concrete's crate (and its `link_force()` helper, if any) into
the binary to satisfy the binding.

### `TYPE_NAME` is wire identity

The `TYPE_NAME` constant on each `ConcreteComponent` impl is the
artifact's lookup key. Bumping `TYPE_NAME` is a breaking change
for artifact compatibility — same contract as ONNX operator
names. Pin it as a stable identifier and version-tag through
naming conventions (e.g. `myapp::v1::ServerBackend`).

### Artifact format

The compiled artifact is a single prost-encoded `ModelProto`.
Two metadata stamps drive install:

- `metadata_props["ai.bytesandbrains.compiled"] = "v1"` — the
  compilation passport. `install` returns
  `InstallError::NotCompiled` when the stamp is absent and
  `InstallError::IncompatibleCompiledVersion` when the version
  doesn't match the running framework.
- `metadata_props["ai.bytesandbrains.binding.<target>.<slot>"]
  = "<role>|<TYPE_NAME>|<slot_id|-1>"` — one entry per
  `(target_function, slot)` pair. `install` parses the entries
  for the chosen target to recover the slot table.

Encoding is plain `prost`; decoding is `ModelProto::decode(bytes)`.
No envelope wrapper, no codec layer.

### Cross-language deployment

Since the artifact is `prost`-encoded `ModelProto` (widely
supported), a Go or Python runtime could load the same artifact
provided it registers equivalent Component impls under matching
`TYPE_NAME`s. Not a v1 framework goal, but the design admits it.

---

## Combining both

A Module can use both forms — a `Backend` placeholder for the
role you bind per-device + a concrete `KnownTokenizer` you want
configured on the build host. The Form-2 artifact encodes the
concrete bindings the build host chose; the per-device install
still surfaces the same `bb::install(...)` entry point. The two
patterns compose cleanly.

## Verification (rough sizing)

After landing your build:

- `cargo build --release -p server` then `cargo build --release -p client`. Both succeed independently.
- `cargo bloat --crates -p server | head` — confirm the
  client-only crate names don't appear.
- `nm target/release/server | grep <ClientType>` — empty output.

## See also

- `docs/AUTHORING_COMPONENTS.md` — Module + Component authoring
- `docs/IR_AND_DSL.md` — DSL → ONNX `ModelProto` story
- `tests/two_placeholders.rs` — regression test for the
  placeholder slot-distinctness invariant
- `src/install.rs` — `bb::install` entry point source
