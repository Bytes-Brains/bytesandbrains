# Component Dependencies

The framework uses generic slot-keyed dependency declarations.
Any Component (Index, Aggregator, Model, etc.) declares the
sibling Components it depends on; the compiler verifies each
dependency at compile time; the runtime resolves them through a
single generic slot registry.

The dependency surface is symmetric across roles — there is no
backend-specific special case. The same `#[depends(role = "slot")]`
mechanism that lets an Index reach a Backend lets it reach a
Model, an Aggregator, or any other Component.

## Declaring dependencies

Authors write `#[depends(<role> = "<slot>")]` on the struct
deriving `bb::Concrete`:

```rust
#[derive(bb::Concrete, bb::Index)]
#[depends(backend = "compute")]
pub struct HnswIndex {
    // ... fields ...
}
```

Multiple dependencies stack (multiple attributes OR multiple
entries inside one attribute):

```rust
#[derive(bb::Concrete, bb::Model)]
#[depends(backend = "compute")]
#[depends(index = "knowledge_base")]
pub struct RagModel { … }

// Equivalent:
#[depends(backend = "compute", index = "knowledge_base")]
pub struct RagModel { … }
```

Valid role names: `index | aggregator | model | compressor | data_source | peer_selector | backend | protocol`.
Each maps to the canonical PascalCase role identifier
(`Backend`, `Index`, `Model`, ...) carried on `DependencyDecl.role`.

## The surface

The derive macro emits two pieces of data:

1. A `DEPENDENCIES: &'static [DependencyDecl]` const on the
   `ConcreteComponent` trait:
   ```rust
   pub struct DependencyDecl {
       pub role: &'static str,     // "Backend"
       pub slot: &'static str,     // "compute"
   }
   ```

2. The same slice on the `ConcreteComponentRegistration` inventory
   carrier so the framework recovers the dep list from a
   `TYPE_NAME` lookup alone.

## Compile-time enforcement

The compiler's `resolve_component_dependencies` pass walks the
artifact's `BindingSpec` and verifies every declared dependency:

```rust
let artifact = bb::Compiler::new()
    .bind_index::<HnswIndex>("primary_index")  // declares need for "compute"
    .bind_backend::<CpuBackend>("compute")      // satisfies it
    .compile(module)?;
```

If the author forgets `.bind_backend::<CpuBackend>("compute")`:

```
error: unbound dependency
  HnswIndex bound at slot "primary_index" requires:
    - a Backend at slot "compute"
  but no such slot is bound.
```

(Typed as `CompileError::UnboundDependency`.) If the slot exists
but is bound to the wrong KIND of concrete (an Index instead of
a Backend at "compute"):

```
error: dependency role mismatch
  HnswIndex bound at slot "primary_index" requires a Backend at
  slot "compute", but the slot is bound to an IndexRuntime.
```

(Typed as `CompileError::DependencyRoleMismatch`.)

## IR metadata

The compiler also stamps each declared dependency onto every
NodeProto the Component contributes to the graph:

```text
ai.bytesandbrains.dep.Backend = "compute"
```

Downstream passes + tooling read the metadata directly from the
IR.

## Runtime resolution

Contracts that take a `&mut RuntimeResourceRef<'_>` ctx reach
their bound dependencies via `ctx.dependency::<T>(slot)`. The
`PeerSelector` Contract is the canonical ctx-bearing surface:

```rust
use bb::completion::{CompletionHandle, ContractResponse};
use bb::contracts::peer_selector::{PeerSelector, SelectParams};
use bb::runtime::RuntimeResourceRef;
use bb::ids::PeerId;

impl PeerSelector for MySelector {
    type Error = MyError;

    fn select(
        &mut self,
        params: SelectParams,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<Vec<PeerId>, MyError>,
    ) -> ContractResponse<Vec<PeerId>, MyError> {
        let backend = ctx
            .dependency::<CpuBackend>("compute")
            .expect("compiler verified this is bound");
        let _ = backend; // use the dependency to compute the selection
        let _ = completion; // or retain it and return Later
        ContractResponse::Now(Ok(Vec::new()))
    }

    // ... `sample` + `current_view` as required ...
}
```

The `Index`, `Backend`, `Aggregator`, `Model`, `Codec`, and
`DataSource` Contracts do not take a ctx parameter — each is a
self-contained method-per-op surface (`Index::add`,
`Index::search`, `Index::remove`, …). Components on those
Contracts reach siblings through the engine's slot registry by
embedding their dependencies as fields and stamping the slot via
`#[depends(...)]`; the install path injects the bound
`ComponentRef` into the field at construction time.

The `ctx.dependency::<T>(slot)` accessor returns
`Result<&T, DependencyError>` typed as:

- `NotBound { slot }` — no Component bound at the slot.
- `TypeMismatch { slot, expected }` — bound concrete is a
  different type.

The compile-time verification means both errors are unreachable
in production; the typed surface lets test fixtures + tooling
probe the slot registry without aborting the process.

## The generic slot registry

The engine's runtime carrier is a single generic map:

```rust
pub struct Engine {
    pub slots: HashMap<String, ComponentRef>,
    // …
}
```

Backends, indexes, models, peer selectors, custom Components —
every Component lives in this map keyed by author-chosen slot
name.

Dispatch reads through `ComponentsView::for_slot(name)` or
`for_slot_as::<T>(name)` (the typed downcast) at op invocation
time.

## Why the generalization matters

Two Backends on one Node now bind cleanly:

```rust
let artifact = bb::Compiler::new()
    .bind_backend::<SimdCpu>("vector_ops")     // index's SIMD-heavy work
    .bind_backend::<GpuBackend>("matmul")      // model's matmul work
    .bind_index::<HnswIndex>("index")          // depends(backend = "vector_ops")
    .bind_model::<MyTransformer>("model")      // depends(backend = "matmul")
    .compile(module)?;
```

The old `default_backend_role` first-installed-wins resolution is
gone; every Component states which slot it depends on, and the
compiler verifies the wiring.
