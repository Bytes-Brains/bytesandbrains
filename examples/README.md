# Examples — reading order

Eight runnable examples that build a working mental model of the
`bytesandbrains` framework. Each one is single-file, runnable with
`cargo run --example <name> --features test-components`, and chains
into the next.

Read in this order:

| # | Example | What it adds |
|---|---|---|
| 1 | [`quickstart`](quickstart.rs) | The smallest end-to-end Module: `Compile → install → Node::poll`. Two no-op Concretes + a one-line `Module::body`. |
| 2 | [`component_with_dependency`](component_with_dependency.rs) | A Concrete that declares a dependency on a sibling Concrete via `#[depends(backend = "compute")]` and reaches it through `ctx.dependency::<T>("compute")`. |
| 3 | [`custom_index_hnsw`](custom_index_hnsw.rs) | A real `Index` Contract impl wrapping a third-party HNSW. Demonstrates `CompletionHandle` + `ContractResponse::Later` for async role methods. |
| 4 | [`multi_target_network`](multi_target_network.rs) | The graph is split into multiple install targets. Shows how `g.net_out` declares a network boundary, how `compile` emits one `FunctionProto` per target, and how an in-process `Bus` forwards `SendEnvelope` between Nodes. |
| 5 | [`federated_learning`](federated_learning.rs) | The canonical use case: a Module composing `ModelSlot`, `AggregatorSlot`, `PeerSelectorSlot`, and `DataSourceSlot`. Includes the FedAvg aggregator and the `Module::bootstrap` override pattern. |
| 6 | [`single_node_federated_learning`](single_node_federated_learning.rs) | One Node hosts both client-side and server-side partitions via `install(&[client_target, server_target])`. Slot-binding dedup shares one `Model` and one `FedAvg` aggregator across both roles. |
| 7 | [`custom_compiler_pass`](custom_compiler_pass.rs) | Plugging a user-defined `CompilerStage` into the compilation pipeline. Useful when you want graph rewrites that aren't in the canonical 18-pass set. |
| 8 | [`polymorphic_types`](polymorphic_types.rs) | The `TypeNode` polymorphism tree end-to-end: the `TypeSolver` rejects mixed-precision graphs, and `Codec` bridges two storage positions. |

## Running

Most examples require `test-components` to pull in the lightweight
`KademliaHand` peer-sampling fixture that the framework uses for
end-to-end smoke tests:

```sh
cargo run --example federated_learning --features test-components
```

Each example prints a short final line on the last `EngineStep` it
verifies — when the print fires, the example actually executed the
documented path (no cosmetic prints).

## Common fixtures

`examples/common/mod.rs` carries reusable Concrete fixtures (a stub
data source, a stub model, a stub aggregator) plus the in-process
`Bus` router so examples can focus on the one thing they
demonstrate. Read it once if you want to author fixtures for your
own tests.
