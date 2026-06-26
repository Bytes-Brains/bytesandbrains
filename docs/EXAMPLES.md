# Examples

The `examples/` directory carries seven runnable examples that
demonstrate the canonical authoring + execution pipeline.

All seven follow the same shape: build the Module, compile the
bound concretes, install, drive the poll loop.

```rust
use std::task::{Context, Poll, Waker};

use bytesandbrains::compiler::Compiler;
use bytesandbrains::framework::Address;
use bytesandbrains::ids::PeerId;
use bytesandbrains::{install, Config};

let model    = MyModule.build()?;
let compiled = Compiler::new()
    .bind_backend::<MyBackend>("compute")   // bind every declared slot
    .compile(model)?;                        // → one ModelProto
let mut node = install(
    PeerId::from(1u64),
    Address::empty(),
    compiled,
    "MyModule",
    Config::new(),
)?;

let waker  = Waker::noop();
let mut cx = Context::from_waker(waker);
while let Poll::Ready(steps) = node.poll(&mut cx) {
    if steps.is_empty() {
        break;
    }
    // handle the EngineStep stream
}
```

`Compiler::compile()` returns ONE `ModelProto` whose `functions[]`
carries every partition (root + sub-Modules + backend subgraphs).
`install(peer, addr, model, target, config)` picks the function
whose name matches `target`, constructs each declared component
from the binding metadata, and brings the Node up. `Node::poll`
returns `Poll::Ready(steps)` when the engine made progress and
`Poll::Pending` when it drained to quiescence (the ingress
waker registers with `cx`).

## `component_with_dependency`

Demonstrates an `Index` impl that declares a `Backend` dependency via
`#[depends(backend = "compute")]`. Drives the install path where one
component reaches its sibling through `ctx.dependency::<T>("compute")`.

```text
cargo run --example component_with_dependency
```

Highlights: `#[depends]`, sibling component resolution, slot-based
dependency wiring.

## `custom_compiler_pass`

Demonstrates registering a user-supplied compiler stage on the
`Compiler`. The example authors a `StampTracingIds` stage that walks
every `wire.Send` and stamps a `tracing_id` metadata.

```text
cargo run --example custom_compiler_pass
```

Highlights: `Compiler::push_back_stage`, custom compiler-stage
extension, declarative inspection of `ModelProto`.

## `custom_index_hnsw`

Implements `bb::contracts::Index` for an `HnswIndex` wrapping
`instant_distance::HnswMap`. Spins a worker thread that owns the
HNSW data; Contract methods push `WorkItem`s to the worker over an
`mpsc::Sender`. Completions ship back through the
`CompletionHandle` from the worker thread.

```text
cargo run --example custom_index_hnsw
```

Highlights: async Contract dispatch, worker-thread pattern,
`CompletionHandle` cross-thread shipping, k-NN search.

## `federated_learning`

Three-Module federated-learning topology (Client, ServerLogic,
ServerReduce) that demonstrates aggregator + model + peer-selector
composition end-to-end across multiple Nodes in the same process.

```text
cargo run --example federated_learning
```

Highlights: multi-component composition, per-Node install with
different `target` names, end-to-end federated round flow.

## `multi_target_network`

Five-Module federated topology (`DataSource`, `Trainer`,
`Aggregator`, `Coordinator`, `Predictor`) composed into a root
`FedApp` Module. The example highlights every step from Module
authoring through per-Node install. Each Node receives the SAME
ModelProto and selects its role by `target`.

```text
cargo run --example multi_target_network
```

Highlights: composition, build, per-target install, in-process bus
routing.

## `named_ports_module`

Demonstrates the `named ports` Module pattern: ports labeled by
author-chosen names rather than positional. Useful for asymmetric
multi-input / multi-output Modules whose inputs and outputs aren't
naturally ordered.

```text
cargo run --example named_ports_module
```

Highlights: named ports, asymmetric Module wiring.

## `polymorphic_types`

Demonstrates the type system's polymorphism: ops authored against
generic tensor types resolve to concrete element types via the
TypeSolver pass.

```text
cargo run --example polymorphic_types
```

Highlights: TypeNode lattice, TypeSolver, op signature
polymorphism.
