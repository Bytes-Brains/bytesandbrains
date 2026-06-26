# Examples — reading order

Six runnable examples that build a working mental model of the
`bytesandbrains` framework. Each one is single-file, runnable with
`cargo run --example <name> --features test-components`, and chains
into the next.

Read in this order:

| # | Example | What it adds |
|---|---|---|
| 1 | [`component_with_dependency`](component_with_dependency.rs) | The smallest end-to-end Module: `Compile → install → Node::poll`. Shows how a Concrete declares a dependency on a sibling Concrete via `#[depends(backend = "compute")]`. |
| 2 | [`custom_index_hnsw`](custom_index_hnsw.rs) | A real `Index` Contract impl wrapping a third-party HNSW. Demonstrates `CompletionHandle` + `ContractResponse::Later` for async role methods. |
| 3 | [`multi_target_network`](multi_target_network.rs) | The graph is split into multiple install targets. Shows how `g.net_out` declares a network boundary and how `compile_partitions` emits one `ModelProto` per target. |
| 4 | [`federated_learning`](federated_learning.rs) | The canonical use case: a Module composing `ModelSlot`, `AggregatorSlot`, `PeerSelectorSlot`, and `DataSourceSlot`. Includes the FedAvg aggregator. |
| 5 | [`custom_compiler_pass`](custom_compiler_pass.rs) | Plugging a user-defined `CompilerStage` into the compilation pipeline. Useful when you want graph rewrites that aren't in the canonical 18-pass set. |
| 6 | [`polymorphic_types`](polymorphic_types.rs) | The `TypeNode` polymorphism tree end-to-end: the `TypeSolver` rejects mixed-precision graphs, and `Codec` bridges two storage positions. |

## Running

Most examples require `test-components` to pull in the lightweight
`KademliaHand` peer-sampling fixture that the framework uses for
end-to-end smoke tests:

```sh
cargo run --example federated_learning --features test-components
```

Each example prints a short `✓` line on the last `EngineStep` it
verifies — when the print fires, the example actually executed the
documented path (no cosmetic prints).

## Common fixtures

`examples/common/mod.rs` carries reusable Concrete fixtures (a stub
data source, a stub model, a stub aggregator) so examples can focus on
the one thing they demonstrate. Read it once if you want to author
fixtures for your own tests.
