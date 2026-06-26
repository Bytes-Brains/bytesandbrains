# Polymorphic Type System

The framework's type system is a polymorphic hierarchy
resolved at compile time by the bb-compiler's TypeSolver. Authors
write zero type annotations on Module ports; the compiler walks
the ops connected to each port and computes the most permissive
bound that satisfies every connected use site.

## The `Storage` trait

`bb_ir::types::Storage` is the static link between a Rust storage type
and its `TypeNode`. Library makers declare *where in the polymorphism
tree* their concrete sits by picking the `Storage` impl for the
associated type on their Contract.

Source: `bb-ir/src/types/storage.rs`.

```rust
pub trait Storage: Send + Sync + 'static {
    /// Position-in-tree declaration. The TypeNode static this constant
    /// points at decides what other storage types unify with this one
    /// during the type-solver walk.
    const TYPE: &'static TypeNode;
}
```

The associated type on each Contract is `?Sized + Storage` so unsized
slice types like `[f32]` work directly (`&[f32]` is the borrowed form;
`Box<[f32]>` is the owned return form).

**Framework blanket impls** (in `bb-ir/src/types/storage.rs`):

| Rust type | `Storage::TYPE`      |
|-----------|----------------------|
| `[f32]`   | `TYPE_TENSOR_F32`    |
| `[f64]`   | `TYPE_TENSOR_F64`    |
| `[u16]`   | `TYPE_TENSOR_F16`    |
| `[u8]`    | `TYPE_TENSOR_U8`     |
| `[i32]`   | `TYPE_TENSOR_I32`    |
| `f32`     | `TYPE_SCALAR_F32`    |
| `f64`     | `TYPE_SCALAR_F64`    |
| `u16`     | `TYPE_SCALAR_F16`    |
| `u8`      | `TYPE_SCALAR_U8`     |
| `i32`     | `TYPE_SCALAR_I32`    |
| `AnyTensor` | `TYPE_TENSOR`      |

(`[u16]` is the canonical bit-packed f16 representation; see
`bb-ir/src/types/builtins.rs` for the `TYPE_TENSOR_F16` static.)

## `AnyTensor`

`bb_ir::types::AnyTensor` is the concrete-erased tensor. It stores
raw bytes plus a runtime-known `Dtype` and shape. Compute-outsourcing
concretes — an `Index` that delegates distance math to a bound
`Backend` — declare `type Vector = AnyTensor`. Because
`AnyTensor::TYPE = &TYPE_TENSOR` (a non-leaf position in the tree),
any tensor subtype unifies into it at the type-solver level.

```rust
pub struct AnyTensor {
    pub bytes: Vec<u8>,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

pub enum Dtype { F32, F64, F16, U8, I32 }
```

`Dtype::type_node()` maps each variant to the corresponding
framework `TypeNode` static so callers can dispatch on the runtime
dtype without hard-coding string ids.

Source: `bb-ir/src/types/storage.rs`.

## Type tree

Types form an open lattice registered via `inventory`:

```
Any
├── Tensor
│   ├── Tensor<F32>
│   ├── Tensor<F64>
│   ├── Tensor<F16>
│   ├── Tensor<U8>
│   ├── Tensor<I32>
│   └── … (downstream backends extend via inventory::submit!)
├── Scalar
│   ├── F32, F64, F16, U8, I32, …
├── PeerId
├── PeerIdVec
├── Trigger
├── Bytes
└── (custom user types — registered the same way)
```

Each node is a `bb_ir::types::TypeNode`:

```rust
pub struct TypeNode {
    pub id: &'static str,                    // "tensor.f32"
    pub parent: Option<&'static str>,        // "tensor"
    pub kind: TypeKind,                      // Concrete | Abstract
    pub ffi_name: &'static str,              // "bb_tensor_f32_t"
    pub wire_hash: u64,                      // stable cross-language identity
}
```

Subtype queries are cached in a `Lattice` built once at startup.

## Type relations on ops

`AtomicOpDecl` carries `type_relations: &'static [TypeRelation]`
declaring constraint bounds, not concrete types:

```rust
pub enum TypeRelation {
    SameType(&'static [PortRef]),
    SameElementType(&'static [PortRef]),
    BroadcastShape { in0: PortRef, in1: PortRef, out: PortRef },
    Elementwise { input: PortRef, output: PortRef },
    ReduceOver { input: PortRef, output: PortRef },
    Custom(fn(&mut TypeSolver, &[TypeRef]) -> RelationResult),
}
```

`Add`, `Mul`, `Sub`, `Div` declare `[SameElementType([in0, in1, out0]), BroadcastShape{...}]`.
`MatMul` declares `[SameElementType([in0, in1, out0]), MatmulShape{...}]`.
`ReduceSum`/`ReduceMean` declare `[Elementwise]`.

## TypeSolver

`bb_compiler::TypeSolver` runs a bipartite worklist (TVM Relay
shape):

1. Allocate a type variable per port; seed declared bounds.
2. Allocate a relation node per `TypeRelation`; cross-link with
   `rel_set` back-edges.
3. Drain the worklist: pop a relation, run it. `Refined` →
   requeue dependents. `Satisfied` → remove. `Defer` → leave for
   later. `Failed` → typed `BuildError::TypeConstraintFailed`.
4. Fixpoint: worklist empty.
5. **Post-condition**: every type variable resolves to a concrete
   (leaf) `TypeNode`. Any unresolved variable → typed
   `BuildError::UnresolvedType`.

## Coercion is explicit

When the solver finds a producer/consumer type mismatch (e.g. a
`wire.Recv` delivers `Tensor<F32>` to a port declared `Tensor<F64>`),
it does NOT silently coerce. `insert_cast_ops` inserts an
explicit `ai.bytesandbrains.types::Cast` NodeProto with
`from = F32, to = F64` attributes. Coercion is visible in the IR.

## Install-time dispatch (Backend Contract surface)

The Backend Contract trait in `bb-runtime/src/contracts/backend.rs`
exposes the framework's tensor-compute surface as two
complementary halves:

- **30 typed per-op methods** — one entry per primitive in
  `bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS` (`add`,
  `mul`, `matmul`, `reduce_sum`, `reshape`, …). Each method
  takes typed `&Self::Tensor` inputs and returns
  `Result<Self::Tensor, Self::Error>`.
- **One `execute(&GraphProto, inputs, attrs)` method** that
  runs a whole subgraph.

Default impls in `backend_default_walk` bridge the two halves:
a backend overriding the 30 per-op methods (e.g. a CPU
backend wired through `ndarray`) gets `execute` for free via
a walker; a backend overriding `execute` natively (e.g. a
Burn-style backend that compiles the whole `GraphProto` to its
own IR) gets the per-op surface for free via one-node graph
wrapping. Backends MUST override at least one side.

`Compiler::compile()` walks each bound concrete and stamps
resolved dispatch metadata onto every contributing NodeProto.
The inventory registry maps each declared `TYPE_NAME` to the
concrete's role-method dispatch table; the runtime hot path
reads through that stamped metadata at op invocation time.

Per-op extensions (Relu, Sigmoid, MaxPool, BatchNormalization,
Conv, …) are not on the Contract surface. A backend may declare
extension opsets through
`BackendRuntime::extension_opsets` and handle them in its own
`execute` override; a future lowering pass decomposes extension
ops into the 30 primitives so the Contract surface covers any
graph.

## Module port types are inferred

Ports are recorded inside the Module body via `g.input(name)` and
`g.output(name, value)`. The author writes no type annotations:

```rust
impl Module for Example {
    fn name(&self) -> &str { "Example" }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let response = compute(g, query);
        g.output("response", response);
    }
}
```

The TypeSolver walks the body's ops + computes each port's
effective type. The author never writes `TENSOR_F32`, never
writes `Tensor`, never writes `Scalar`. Types are derived from
the ops.

## How to declare ports

**Inputs are declared by name.** There is exactly one way:

```rust
let query = g.input("query");
```

The TypeSolver narrows the type from connected ops at compile
time. The recorder stamps the input with
`OPAQUE_PLACEHOLDER`; the canonical pipeline's `type_solver`
pass walks the op constraints (`type_relations` on each
`AtomicOpDecl`) and resolves the input's TypeNode to the most
specific bound that satisfies every connected use. The
resolved type is stamped back onto the IR via
`apply_solution_to_value_info`.

`Graph::input(name)` is the single canonical form. Authors who
need control-plane peer routing
get it for free: a graph input feeding a `wire.Send`'s peer
slot is detected by `infer_peer_classes` and stamped with
`peer_class = <input_name>` automatically.

## Strict by default; permissive as escape hatch

The TypeSolver runs **strictly by default**. Every value site
must resolve to a concrete TypeNode; `BuildError::UnresolvedType`
surfaces the first abstract slot. The recorder helpers
(placeholders, syscalls, `Graph::net_out`) stamp denotations
on every output they mint, so canonical-shape graphs pass
without further annotation.

Tests + bespoke pipelines that hand-author `NodeProto`s without
declaring `value_info` for every value drop into permissive
mode:

```rust
let artifact = bb::Compiler::new()
    .with_permissive_types()
    .bind_backend::<CpuBackend>("compute")
    .compile(model)?;
```

Permissive mode lets unresolved values stay at `TYPE_ANY` (the
`ai.bytesandbrains.opaque` denotation maps to the concrete
`bytes` leaf, so opaque-payload graphs typecheck even in
strict mode — only genuinely under-declared values surface as
errors).

## Open tree via inventory

Third-party backends extend the type tree the same way they
extend the op registry:

```rust
inventory::submit! {
    TypeNode {
        id: "tensor.bf16",
        parent: Some("tensor"),
        kind: TypeKind::Concrete,
        ffi_name: "bb_tensor_bf16_t",
        wire_hash: 0x1234_5678_9abc_def0,
    }
}
```

The lattice rebuilds at startup; subtype queries cache.
