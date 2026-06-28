# Examples

The `examples/` directory carries eight runnable examples that
demonstrate the canonical authoring + execution pipeline.

All eight follow the same shape: build the Module, compile the
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
    vec![Address::empty()],
    compiled,
    &["MyModule"],
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
`install(peer, addrs, model, targets, config)` picks each function
whose name matches a `target`, constructs each declared component
from the binding metadata, and brings the Node up. `Node::poll`
returns `Poll::Ready(steps)` when the engine made progress and
`Poll::Pending` when it drained to quiescence (the ingress
waker registers with `cx`).

## `quickstart`

Smallest end-to-end shape: define two no-op Concretes, record a
one-line `Module::body`, compile, install, drive `Node::poll` to
quiescence. No bootstrap override, no wire boundary, no transport.

```text
cargo run --example quickstart --features test-components
```

Highlights: smallest viable Concretes, single-line `Module::body`,
bare `Compiler::new().bind_data_source().bind_model().compile()`.

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

Highlights: async Contract dispatch (`ContractResponse::Later`),
worker-thread pattern, `CompletionHandle` cross-thread shipping,
k-NN search.

## `federated_learning`

Three-Module federated-learning topology (Client, ServerLogic,
ServerReduce) that demonstrates aggregator + model + peer-selector
composition end-to-end across multiple Nodes in the same process.
Uses the canonical client-side bootstrap-as-function pattern
(`Module::bootstrap` override seeding `address_book_insert_many`
+ `GlobalRegistryClient::announce`).

```text
cargo run --example federated_learning --features test-components
```

Highlights: multi-component composition, per-Node install with
different `target` names, `Module::bootstrap` override, end-to-end
federated round flow.

## `single_node_federated_learning`

Multi-role-single-Node federated learning: one Node hosts BOTH
the client-side and server-side partitions via
`install(&[client_target, server_target])`. The deduplicating install
pass collapses the shared `model` and `aggregator` slots so both
partitions dispatch through the same in-memory `ComponentRef`.

```text
cargo run --example single_node_federated_learning --features test-components
```

Highlights: multi-target install on a single Node, slot-binding
dedup for shared in-memory state, `Node::invoke` to seed a
composition-level input.

## `multi_target_network`

Three-leaf federated topology (`LoaderLeaf`, `TrainerLeaf`,
`SinkLeaf`) composed into a `FedNetwork` Module with a single
`g.net_out` partition boundary. The example demonstrates
multi-target install across two separate Nodes plus the
in-process `Bus` pattern that forwards `EngineStep::SendEnvelope`
between Nodes.

```text
cargo run --example multi_target_network
```

Highlights: composition, build, per-target install across distinct
peers, in-process bus routing.

## `polymorphic_types`

Demonstrates the type system's polymorphism: ops authored against
generic tensor types resolve to concrete element types via the
TypeSolver pass.

```text
cargo run --example polymorphic_types
```

Highlights: TypeNode lattice, TypeSolver, op signature
polymorphism.
