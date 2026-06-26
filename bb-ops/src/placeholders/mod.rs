//! Role-method dispatch slot placeholders. Each `*Slot` unit
//! struct is a generic slot bound at compile time via
//! `Compiler::new().bind_<role>::<T>("slot")`. DSL methods record
//! NodeProtos stamped with `(required_trait, slot_id)` for
//! binding-resolution routing.
//!
//! ```ignore
//! pub struct MyModule {
//!     backend: BackendSlot,       // bind any BackendRuntime
//!     data:    DataLoaderSlot,    // bind any DataSourceRuntime
//! }
//! ```

use bb_dsl::graph::{attr_float, attr_graph, attr_int, attr_ints, attr_tensor, kv, Graph};
use bb_dsl::output::Output;
use bb_ir::proto::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto};
use bb_ir::types::{TYPE_TENSOR, TYPE_TENSOR_F32, TYPE_TRIGGER};

/// Generic Backend slot. Carries the `ai.onnx v1` DSL catalog
/// (48 methods). Outputs are typed `&TYPE_TENSOR_F32`. The
/// `BackendSubgraph` carrier is compiler-emitted, not a DSL method.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendSlot;

impl BackendSlot {
    // --- Private recording helper --------------------------------

    /// Records an `ai.onnx::<op_type>` NodeProto stamped with
    /// `(required_trait, slot_id)`. Returns `&TYPE_TENSOR_F32` outputs.
    fn record_op(
        &self,
        g: &mut Graph,
        op_type: &str,
        input_names: Vec<String>,
        n_outputs: usize,
        attribute: Vec<AttributeProto>,
    ) -> Vec<Output> {
        let slot_id = g.register_generic(self, "BackendRuntime");
        let output_names: Vec<String> = (0..n_outputs).map(|_| g.next_site_name()).collect();
        g.push_node(NodeProto {
            op_type: op_type.into(),
            domain: "ai.onnx".into(),
            input: input_names,
            output: output_names.clone(),
            attribute,
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "BackendRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
            ],
            ..Default::default()
        });
        // Stamp value_info for each output so the type_solver sees a
        // declared denotation at every recorder site. This is the
        // recorder side of #20 (strict-types-by-default): with
        // denotations stamped here, downstream passes treat the
        // graph as fully typed without needing a `with_strict_types`
        // toggle.
        for name in &output_names {
            g.declare_value_info(name, &TYPE_TENSOR_F32);
        }
        output_names
            .into_iter()
            .map(|n| Output::new(n, &TYPE_TENSOR_F32))
            .collect()
    }

    fn record_one(
        &self,
        g: &mut Graph,
        op_type: &str,
        input_names: Vec<String>,
        attribute: Vec<AttributeProto>,
    ) -> Output {
        self.record_op(g, op_type, input_names, 1, attribute)
            .into_iter()
            .next()
            .expect("record_op with n_outputs=1")
    }

    // --- Creation ------------------------------------------------

    /// `Zeros(dims)` - zero-initialized tensor of given shape.
    pub fn zeros(&self, g: &mut Graph, dims: Vec<i64>) -> Output {
        self.record_one(g, "Zeros", vec![], vec![attr_ints("dims", dims)])
    }

    /// `Ones(dims)` - one-initialized tensor of given shape.
    pub fn ones(&self, g: &mut Graph, dims: Vec<i64>) -> Output {
        self.record_one(g, "Ones", vec![], vec![attr_ints("dims", dims)])
    }

    /// `Constant(value)` - embedded literal tensor.
    pub fn constant(&self, g: &mut Graph, value: TensorProto) -> Output {
        self.record_one(g, "Constant", vec![], vec![attr_tensor("value", value)])
    }

    // --- Element-wise arithmetic ---------------------------------

    /// `Add` - element-wise `a + b`.
    pub fn add(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Add", vec![a.name, b.name], vec![])
    }

    /// `Sub` - element-wise `a - b`.
    pub fn sub(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Sub", vec![a.name, b.name], vec![])
    }

    /// `Mul` - element-wise `a * b`.
    pub fn mul(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Mul", vec![a.name, b.name], vec![])
    }

    /// `Div` - element-wise `a / b`.
    pub fn div(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Div", vec![a.name, b.name], vec![])
    }

    /// `Neg` - element-wise negation.
    pub fn neg(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Neg", vec![t.name], vec![])
    }

    /// `Abs` - element-wise absolute value.
    pub fn abs(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Abs", vec![t.name], vec![])
    }

    /// `Sqrt` - element-wise square root.
    pub fn sqrt(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Sqrt", vec![t.name], vec![])
    }

    /// `Exp` - element-wise natural exponential.
    pub fn exp(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Exp", vec![t.name], vec![])
    }

    /// `Log` - element-wise natural logarithm.
    pub fn log(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Log", vec![t.name], vec![])
    }

    /// `Pow` - element-wise `a ** b`.
    pub fn pow(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Pow", vec![a.name, b.name], vec![])
    }

    // --- Linear algebra ------------------------------------------

    /// `MatMul` - matrix multiplication (canonical example).
    pub fn matmul(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "MatMul", vec![a.name, b.name], vec![])
    }

    /// `Gemm` - `alpha * (a @ b) + beta * c` with optional transpose.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        g: &mut Graph,
        a: Output,
        b: Output,
        c: Option<Output>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) -> Output {
        let mut inputs = vec![a.name, b.name];
        if let Some(c) = c {
            inputs.push(c.name);
        }
        self.record_one(
            g,
            "Gemm",
            inputs,
            vec![
                attr_float("alpha", alpha),
                attr_float("beta", beta),
                attr_int("transA", trans_a as i64),
                attr_int("transB", trans_b as i64),
            ],
        )
    }

    /// `Dot` - dot product (reduces along last axis for higher rank).
    pub fn dot(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Dot", vec![a.name, b.name], vec![])
    }

    // --- Activations ---------------------------------------------

    /// `Relu` - `max(0, x)`.
    pub fn relu(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Relu", vec![t.name], vec![])
    }

    /// `Sigmoid` - `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Sigmoid", vec![t.name], vec![])
    }

    /// `Tanh` - hyperbolic tangent.
    pub fn tanh(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Tanh", vec![t.name], vec![])
    }

    /// `Softmax(axis)` - softmax along the given axis.
    pub fn softmax(&self, g: &mut Graph, t: Output, axis: i64) -> Output {
        self.record_one(g, "Softmax", vec![t.name], vec![attr_int("axis", axis)])
    }

    /// `LeakyRelu(alpha)` - `x if x > 0 else alpha * x`.
    pub fn leaky_relu(&self, g: &mut Graph, t: Output, alpha: f32) -> Output {
        self.record_one(
            g,
            "LeakyRelu",
            vec![t.name],
            vec![attr_float("alpha", alpha)],
        )
    }

    /// `Gelu` - Gaussian Error Linear Unit.
    pub fn gelu(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Gelu", vec![t.name], vec![])
    }

    // --- Shape / structural --------------------------------------

    /// `Reshape(dims)` - reshape to given dims.
    pub fn reshape(&self, g: &mut Graph, t: Output, dims: Vec<i64>) -> Output {
        self.record_one(g, "Reshape", vec![t.name], vec![attr_ints("dims", dims)])
    }

    /// `Transpose(perm)` - `None` reverses all dims.
    pub fn transpose(&self, g: &mut Graph, t: Output, perm: Option<Vec<i64>>) -> Output {
        let attrs = match perm {
            Some(p) => vec![attr_ints("perm", p)],
            None => vec![],
        };
        self.record_one(g, "Transpose", vec![t.name], attrs)
    }

    /// `Concat(axis)` - concatenate `tensors` along axis.
    pub fn concat(&self, g: &mut Graph, tensors: Vec<Output>, axis: i64) -> Output {
        let inputs = tensors.into_iter().map(|t| t.name).collect();
        self.record_one(g, "Concat", inputs, vec![attr_int("axis", axis)])
    }

    /// `Split(axis, sizes)` - split into N parts. Returns one
    /// `Output` per size.
    pub fn split(&self, g: &mut Graph, t: Output, axis: i64, sizes: Vec<i64>) -> Vec<Output> {
        let n = sizes.len();
        self.record_op(
            g,
            "Split",
            vec![t.name],
            n,
            vec![attr_int("axis", axis), attr_ints("split", sizes)],
        )
    }

    /// `Slice(starts, ends, axes?, steps?)` - NumPy-style slice.
    pub fn slice(
        &self,
        g: &mut Graph,
        t: Output,
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
        steps: Option<Vec<i64>>,
    ) -> Output {
        let mut attrs = vec![attr_ints("starts", starts), attr_ints("ends", ends)];
        if let Some(a) = axes {
            attrs.push(attr_ints("axes", a));
        }
        if let Some(s) = steps {
            attrs.push(attr_ints("steps", s));
        }
        self.record_one(g, "Slice", vec![t.name], attrs)
    }

    /// `Squeeze(axes?)` - remove length-1 dimensions.
    pub fn squeeze(&self, g: &mut Graph, t: Output, axes: Option<Vec<i64>>) -> Output {
        let attrs = match axes {
            Some(a) => vec![attr_ints("axes", a)],
            None => vec![],
        };
        self.record_one(g, "Squeeze", vec![t.name], attrs)
    }

    /// `Unsqueeze(axes)` - insert length-1 dimensions.
    pub fn unsqueeze(&self, g: &mut Graph, t: Output, axes: Vec<i64>) -> Output {
        self.record_one(g, "Unsqueeze", vec![t.name], vec![attr_ints("axes", axes)])
    }

    /// `Identity` - clone pass-through.
    pub fn identity(&self, g: &mut Graph, t: Output) -> Output {
        self.record_one(g, "Identity", vec![t.name], vec![])
    }

    /// `Cast(to)` - cast to the given ONNX `DataType` enum value.
    pub fn cast(&self, g: &mut Graph, t: Output, to_elem_type: i32) -> Output {
        self.record_one(
            g,
            "Cast",
            vec![t.name],
            vec![attr_int("to", to_elem_type as i64)],
        )
    }

    // --- Reductions ----------------------------------------------

    fn reduce(
        &self,
        g: &mut Graph,
        op_type: &str,
        t: Output,
        axes: Option<Vec<i64>>,
        keepdims: bool,
    ) -> Output {
        let mut attrs = vec![attr_int("keepdims", keepdims as i64)];
        if let Some(a) = axes {
            attrs.push(attr_ints("axes", a));
        }
        self.record_one(g, op_type, vec![t.name], attrs)
    }

    /// `ReduceSum(axes?, keepdims)`.
    pub fn reduce_sum(
        &self,
        g: &mut Graph,
        t: Output,
        axes: Option<Vec<i64>>,
        keepdims: bool,
    ) -> Output {
        self.reduce(g, "ReduceSum", t, axes, keepdims)
    }

    /// `ReduceMean(axes?, keepdims)`.
    pub fn reduce_mean(
        &self,
        g: &mut Graph,
        t: Output,
        axes: Option<Vec<i64>>,
        keepdims: bool,
    ) -> Output {
        self.reduce(g, "ReduceMean", t, axes, keepdims)
    }

    /// `ReduceMax(axes?, keepdims)`.
    pub fn reduce_max(
        &self,
        g: &mut Graph,
        t: Output,
        axes: Option<Vec<i64>>,
        keepdims: bool,
    ) -> Output {
        self.reduce(g, "ReduceMax", t, axes, keepdims)
    }

    /// `ReduceMin(axes?, keepdims)`.
    pub fn reduce_min(
        &self,
        g: &mut Graph,
        t: Output,
        axes: Option<Vec<i64>>,
        keepdims: bool,
    ) -> Output {
        self.reduce(g, "ReduceMin", t, axes, keepdims)
    }

    // --- Comparison ----------------------------------------------

    /// `Equal` - element-wise `a == b` (bool tensor).
    pub fn equal(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Equal", vec![a.name, b.name], vec![])
    }

    /// `Greater` - element-wise `a > b` (bool tensor).
    pub fn greater(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Greater", vec![a.name, b.name], vec![])
    }

    /// `Less` - element-wise `a < b` (bool tensor).
    pub fn less(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        self.record_one(g, "Less", vec![a.name, b.name], vec![])
    }

    // --- Normalization -------------------------------------------

    /// `BatchNormalization(epsilon, momentum)`.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_normalization(
        &self,
        g: &mut Graph,
        input: Output,
        scale: Output,
        bias: Output,
        mean: Output,
        variance: Output,
        epsilon: f32,
        momentum: f32,
    ) -> Output {
        self.record_one(
            g,
            "BatchNormalization",
            vec![input.name, scale.name, bias.name, mean.name, variance.name],
            vec![
                attr_float("epsilon", epsilon),
                attr_float("momentum", momentum),
            ],
        )
    }

    /// `LayerNormalization(axis, epsilon)`.
    pub fn layer_normalization(
        &self,
        g: &mut Graph,
        input: Output,
        scale: Output,
        bias: Option<Output>,
        axis: i64,
        epsilon: f32,
    ) -> Output {
        let mut inputs = vec![input.name, scale.name];
        if let Some(b) = bias {
            inputs.push(b.name);
        }
        self.record_one(
            g,
            "LayerNormalization",
            inputs,
            vec![attr_int("axis", axis), attr_float("epsilon", epsilon)],
        )
    }

    // --- Conv / Pool ---------------------------------------------

    /// `Conv(kernel_shape, strides, pads, dilations, group)`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        g: &mut Graph,
        input: Output,
        weight: Output,
        bias: Option<Output>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
    ) -> Output {
        let mut inputs = vec![input.name, weight.name];
        if let Some(b) = bias {
            inputs.push(b.name);
        }
        self.record_one(
            g,
            "Conv",
            inputs,
            vec![
                attr_ints("kernel_shape", kernel_shape),
                attr_ints("strides", strides),
                attr_ints("pads", pads),
                attr_ints("dilations", dilations),
                attr_int("group", group),
            ],
        )
    }

    /// `MaxPool(kernel_shape, strides, pads)`.
    pub fn max_pool(
        &self,
        g: &mut Graph,
        input: Output,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
    ) -> Output {
        self.record_one(
            g,
            "MaxPool",
            vec![input.name],
            vec![
                attr_ints("kernel_shape", kernel_shape),
                attr_ints("strides", strides),
                attr_ints("pads", pads),
            ],
        )
    }

    /// `AveragePool(kernel_shape, strides, pads, count_include_pad)`.
    pub fn average_pool(
        &self,
        g: &mut Graph,
        input: Output,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        count_include_pad: bool,
    ) -> Output {
        self.record_one(
            g,
            "AveragePool",
            vec![input.name],
            vec![
                attr_ints("kernel_shape", kernel_shape),
                attr_ints("strides", strides),
                attr_ints("pads", pads),
                attr_int("count_include_pad", count_include_pad as i64),
            ],
        )
    }

    /// `GlobalAveragePool` - collapse spatial dims to length 1.
    pub fn global_average_pool(&self, g: &mut Graph, input: Output) -> Output {
        self.record_one(g, "GlobalAveragePool", vec![input.name], vec![])
    }

    // --- Indexing ------------------------------------------------

    /// `Gather(axis)`.
    pub fn gather(&self, g: &mut Graph, data: Output, indices: Output, axis: i64) -> Output {
        self.record_one(
            g,
            "Gather",
            vec![data.name, indices.name],
            vec![attr_int("axis", axis)],
        )
    }

    /// `Scatter(axis)`.
    pub fn scatter(
        &self,
        g: &mut Graph,
        data: Output,
        indices: Output,
        updates: Output,
        axis: i64,
    ) -> Output {
        self.record_one(
            g,
            "Scatter",
            vec![data.name, indices.name, updates.name],
            vec![attr_int("axis", axis)],
        )
    }

    // --- Control flow --------------------------------------------

    /// `If(then_branch, else_branch)` - both branches are sub-graphs
    /// carried on `AttributeProto.g` per IR_AND_DSL.md Part 2 line 80.
    /// Returns one `Output` per branch output.
    pub fn if_op(
        &self,
        g: &mut Graph,
        cond: Output,
        then_branch: GraphProto,
        else_branch: GraphProto,
        n_outputs: usize,
    ) -> Vec<Output> {
        self.record_op(
            g,
            "If",
            vec![cond.name],
            n_outputs,
            vec![
                attr_graph("then_branch", then_branch),
                attr_graph("else_branch", else_branch),
            ],
        )
    }

    /// `Loop(body)` - execute `body` until `cond` becomes false or
    /// `max_trip_count` is reached.
    pub fn loop_op(
        &self,
        g: &mut Graph,
        max_trip_count: Option<Output>,
        cond: Option<Output>,
        body: GraphProto,
        initial: Vec<Output>,
        n_outputs: usize,
    ) -> Vec<Output> {
        let mut inputs = vec![
            max_trip_count.map(|o| o.name).unwrap_or_default(),
            cond.map(|o| o.name).unwrap_or_default(),
        ];
        inputs.extend(initial.into_iter().map(|o| o.name));
        self.record_op(g, "Loop", inputs, n_outputs, vec![attr_graph("body", body)])
    }
}

// ---------------------------------------------------------------
// Role-method recording helper
// ---------------------------------------------------------------

/// Record a NodeProto for one role-method DSL call against a generic
/// placeholder. Domain follows the `ai.bytesandbrains.role.<role>`
/// convention from `docs/IR_AND_DSL.md` §5c; metadata stamps the
/// `(required_trait, slot_id)` pair so the compiler's
/// `inline_role_methods` pass can swap in the bound impl's body.
#[allow(clippy::too_many_arguments)]
fn record_role_op<P: 'static>(
    g: &mut Graph,
    placeholder: &P,
    required_trait: &'static str,
    role_domain: &'static str,
    op_type: &str,
    input_names: Vec<String>,
    n_outputs: usize,
    attribute: Vec<AttributeProto>,
) -> Vec<Output> {
    let slot_id = g.register_generic(placeholder, required_trait);
    let output_names: Vec<String> = (0..n_outputs).map(|_| g.next_site_name()).collect();
    g.push_node(NodeProto {
        op_type: op_type.into(),
        domain: role_domain.into(),
        input: input_names,
        output: output_names.clone(),
        attribute,
        metadata_props: vec![
            kv("ai.bytesandbrains.required_trait", required_trait),
            kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
        ],
        ..Default::default()
    });
    for name in &output_names {
        g.declare_value_info(name, &TYPE_TENSOR);
    }
    output_names
        .into_iter()
        .map(|n| Output::new(n, &TYPE_TENSOR))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn record_role_op_one(
    g: &mut Graph,
    placeholder: &impl std::any::Any,
    required_trait: &'static str,
    role_domain: &'static str,
    op_type: &str,
    input_names: Vec<String>,
    attribute: Vec<AttributeProto>,
) -> Output {
    record_role_op(
        g,
        placeholder,
        required_trait,
        role_domain,
        op_type,
        input_names,
        1,
        attribute,
    )
    .into_iter()
    .next()
    .expect("record_role_op with n_outputs=1")
}

// ---------------------------------------------------------------
// Model
// ---------------------------------------------------------------

/// Generic Model slot placeholder. Bind a concrete `ModelRuntime` via
/// `Node::with_model(...)`. Exposes the six role-method
/// DSL operations:
/// `Forward`, `Backward`, `ComputeLoss`, `ApplyDelta`,
/// `LoadParameters`, `Params`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelSlot;

impl ModelSlot {
    /// `Forward(input) -> output` - tensor → tensor forward pass.
    pub fn forward(&self, g: &mut Graph, input: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "Forward",
            vec![input.name],
            vec![],
        )
    }

    /// `Backward(grad) -> cmd` - accumulate gradients given upstream gradient.
    pub fn backward(&self, g: &mut Graph, grad: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "Backward",
            vec![grad.name],
            vec![],
        )
    }

    /// `ComputeLoss(input, target) -> loss` - scalar loss score.
    pub fn compute_loss(&self, g: &mut Graph, input: Output, target: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "ComputeLoss",
            vec![input.name, target.name],
            vec![],
        )
    }

    /// `ApplyDelta(delta) -> cmd` - apply parameter delta in-place.
    pub fn apply_delta(&self, g: &mut Graph, delta: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "ApplyDelta",
            vec![delta.name],
            vec![],
        )
    }

    /// `LoadParameters(params) -> cmd` - load parameters wholesale.
    pub fn load_parameters(&self, g: &mut Graph, params: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "LoadParameters",
            vec![params.name],
            vec![],
        )
    }

    /// `Params() -> params` - snapshot the current parameter tensor.
    pub fn params(&self, g: &mut Graph) -> Output {
        record_role_op_one(
            g,
            self,
            "ModelRuntime",
            "ai.bytesandbrains.role.model",
            "Params",
            vec![],
            vec![],
        )
    }
}

// ---------------------------------------------------------------
// Index
// ---------------------------------------------------------------

/// Generic Index slot placeholder. Bind a concrete `IndexRuntime` via
/// `Node::with_index(...)`. Exposes the three role-method DSL
/// operations: `Add`, `Search`,
/// `Remove`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexSlot;

impl IndexSlot {
    /// `Add(vec) -> cmd` - Shape 2 (stateful insert).
    pub fn add(&self, g: &mut Graph, vec: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "IndexRuntime",
            "ai.bytesandbrains.role.index",
            "Add",
            vec![vec.name],
            vec![],
        )
    }

    /// `Search(query, k=...)` - Shape 2 typically, Shape 1 for
    /// in-memory flat indexes.
    pub fn search(&self, g: &mut Graph, query: Output, k: i64) -> Output {
        record_role_op_one(
            g,
            self,
            "IndexRuntime",
            "ai.bytesandbrains.role.index",
            "Search",
            vec![query.name],
            vec![attr_int("k", k)],
        )
    }

    /// `Remove(id) -> cmd` - Shape 2 (stateful delete).
    pub fn remove(&self, g: &mut Graph, id: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "IndexRuntime",
            "ai.bytesandbrains.role.index",
            "Remove",
            vec![id.name],
            vec![],
        )
    }

    /// `Train(samples) -> trigger` — fire-and-forget calibration pass.
    /// The output is `TYPE_TRIGGER` so authors that need to gate body
    /// ops on training completion can wire the trigger through a
    /// `bb.barrier` or place the call in `Module::bootstrap` to run
    /// before body ops fire.
    pub fn train(&self, g: &mut Graph, samples: Output) -> Output {
        let slot_id = g.register_generic(self, "IndexRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "Train".into(),
            domain: "ai.bytesandbrains.role.index".into(),
            input: vec![samples.name],
            output: vec![out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "IndexRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
                kv("ai.bytesandbrains.index.port", "samples"),
            ],
            ..Default::default()
        });
        g.declare_value_info(&out_name, &TYPE_TRIGGER);
        Output::new(out_name, &TYPE_TRIGGER)
    }
}

// ---------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------

/// Generic Aggregator slot placeholder. Bind a concrete
/// `AggregatorRuntime` via `Node::with_aggregator(...)`. Exposes
/// `Contribute` + `Aggregate` per `docs/IR_AND_DSL.md` §5c.2.
///
/// Both ops carry an opaque metadata channel alongside the
/// tensor: `Contribute` takes `(contribution, metadata)`;
/// `Aggregate` returns `(params, metadata)`. This is the channel
/// hierarchical aggregation rides on — a child aggregator emits
/// its summed `num_samples` (or whatever schema the impl uses) so
/// the parent layer's reduction can weight the child's
/// contribution correctly.
#[derive(Debug, Clone, Copy, Default)]
pub struct AggregatorSlot;

impl AggregatorSlot {
    /// `Contribute(contribution, metadata) -> cmd` - Shape 2
    /// (buffer write). `metadata` is impl-defined bytes (e.g. a
    /// sample count for FedAvg); pass an empty `Output` when the
    /// impl has no metadata channel.
    pub fn contribute(&self, g: &mut Graph, contribution: Output, metadata: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "AggregatorRuntime",
            "ai.bytesandbrains.role.aggregator",
            "Contribute",
            vec![contribution.name, metadata.name],
            vec![],
        )
    }

    /// `Aggregate(trigger) -> (params, metadata)` - Shape 1 (mean /
    /// weighted sum / replace expressible as `ai.onnx`). The output
    /// edge fires only when the reduction completes; downstream
    /// consumers read `params` AND the aggregation's accompanying
    /// `metadata` (e.g. summed `num_samples` for hierarchical
    /// FedAvg) directly off the op's two outputs — no separate
    /// `current_tensor` read needed.
    pub fn aggregate(&self, g: &mut Graph, trigger: Output) -> (Output, Output) {
        let mut outs = record_role_op(
            g,
            self,
            "AggregatorRuntime",
            "ai.bytesandbrains.role.aggregator",
            "Aggregate",
            vec![trigger.name],
            2,
            vec![],
        );
        let metadata = outs.pop().expect("two outputs");
        let params = outs.pop().expect("two outputs");
        (params, metadata)
    }
}

// ---------------------------------------------------------------
// Codec
// ---------------------------------------------------------------

/// Generic Codec slot placeholder. Embed as `codec: CodecSlot` in your
/// Module struct; bind a concrete `CodecRuntime` via
/// `Compiler::new().bind_codec::<T>("slot")…`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodecSlot;

impl CodecSlot {
    /// `Train(samples) → trigger` — optional calibration pass.
    /// Quantizers learn scale/zero-point, PQ codebooks run k-means,
    /// dtype casts skip the call. The output rides `TYPE_TRIGGER` so
    /// `Module::bootstrap` (or a downstream `bb.barrier`) can gate
    /// body ops on training completion. Stamps
    /// `ai.bytesandbrains.codec.port = "in"` since samples flow at
    /// the In storage position.
    pub fn train(&self, g: &mut Graph, samples: Output) -> Output {
        let slot_id = g.register_generic(self, "CodecRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "Train".into(),
            domain: "ai.bytesandbrains.role.codec".into(),
            input: vec![samples.name],
            output: vec![out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "CodecRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
                kv("ai.bytesandbrains.codec.port", "in"),
            ],
            ..Default::default()
        });
        g.declare_value_info(&out_name, &TYPE_TRIGGER);
        Output::new(out_name, &TYPE_TRIGGER)
    }

    /// `Encode(input) → output` — In → Out direction. Stamps
    /// `ai.bytesandbrains.codec.port = "out"` on the NodeProto so the
    /// refinement pass (Task 10) knows which port of the bound
    /// concrete's `<In, Out>` to read for the output denotation.
    pub fn encode(&self, g: &mut Graph, input: Output) -> Output {
        let slot_id = g.register_generic(self, "CodecRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "Encode".into(),
            domain: "ai.bytesandbrains.role.codec".into(),
            input: vec![input.name],
            output: vec![out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "CodecRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
                kv("ai.bytesandbrains.codec.port", "out"),
            ],
            ..Default::default()
        });
        g.declare_value_info(&out_name, &TYPE_TENSOR);
        Output::new(out_name, &TYPE_TENSOR)
    }

    /// `Decode(encoded) → output` — Out → In direction. Stamps
    /// `ai.bytesandbrains.codec.port = "in"`.
    pub fn decode(&self, g: &mut Graph, encoded: Output) -> Output {
        let slot_id = g.register_generic(self, "CodecRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "Decode".into(),
            domain: "ai.bytesandbrains.role.codec".into(),
            input: vec![encoded.name],
            output: vec![out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "CodecRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
                kv("ai.bytesandbrains.codec.port", "in"),
            ],
            ..Default::default()
        });
        g.declare_value_info(&out_name, &TYPE_TENSOR);
        Output::new(out_name, &TYPE_TENSOR)
    }
}

// ---------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------

/// Generic DataSource slot placeholder. Bind a concrete
/// `DataSourceRuntime` via `Node::with_data_source(...)`.
/// Exposes `NextBatch`, `Reset`, `OnDataLoaded` per
/// `docs/IR_AND_DSL.md` §5c.2.
#[derive(Debug, Clone, Copy, Default)]
pub struct DataLoaderSlot;

impl DataLoaderSlot {
    /// `NextBatch() -> (batch, labels)` - Shape 2 (data source has
    /// side effects). Returns two `Output` handles; the second is
    /// optional in spec but always materialized in the DSL surface
    /// for shape symmetry.
    pub fn next_batch(&self, g: &mut Graph) -> (Output, Output) {
        let mut outs = record_role_op(
            g,
            self,
            "DataSourceRuntime",
            "ai.bytesandbrains.role.data_source",
            "NextBatch",
            vec![],
            2,
            vec![],
        );
        let labels = outs.pop().expect("two outputs");
        let batch = outs.pop().expect("two outputs");
        (batch, labels)
    }

    /// `Reset(trigger) -> trigger` - Shape 2.
    pub fn reset(&self, g: &mut Graph, trigger: Output) -> Output {
        record_role_op_one(
            g,
            self,
            "DataSourceRuntime",
            "ai.bytesandbrains.role.data_source",
            "Reset",
            vec![trigger.name],
            vec![],
        )
    }

    /// `OnDataLoaded() -> trigger` - Shape 2 (one-shot
    /// notification).
    pub fn on_data_loaded(&self, g: &mut Graph) -> Output {
        record_role_op_one(
            g,
            self,
            "DataSourceRuntime",
            "ai.bytesandbrains.role.data_source",
            "OnDataLoaded",
            vec![],
            vec![],
        )
    }
}

// ---------------------------------------------------------------
// PeerSelector
// ---------------------------------------------------------------

/// Generic PeerSelector slot placeholder. Bind a concrete
/// `PeerSelectorRuntime` via `Node::with_peer_selector(...)`.
/// Exposes `Sample`, `CurrentView`
///
/// `class` tags every `Output<PeerId>` this placeholder yields with
/// the class of peer it samples from. The compiler's
/// `infer_peer_classes` pass reads that tag to attribute downstream
/// `wire.send`s to the right destination class - that's what makes
/// gossip's self-send case partition correctly (1 class → 1
/// partition with both send and synthesized recv).
#[derive(Debug, Clone, Copy)]
pub struct PeerSelectorSlot {
    /// Class identifier the compiler stamps onto every produced
    /// `Output<PeerId>` via `peer_class` metadata. Defaults to
    /// [`bb_ir::peer_class::SELF_CLASS`] for placeholders
    /// constructed via `Default` - that puts samples on the same
    /// class as the registering Node, the natural gossip case.
    pub class: &'static str,
}

impl Default for PeerSelectorSlot {
    fn default() -> Self {
        Self {
            class: bb_ir::peer_class::SELF_CLASS,
        }
    }
}

impl PeerSelectorSlot {
    /// Construct a sampling placeholder bound to the given class. Use
    /// `PeerSelectorSlot::of_class("gossip_peer")` to tag samples as
    /// "gossip peers" so a downstream `wire.send(payload, neighbor)`
    /// puts its `data` output on the same `gossip_peer` partition.
    pub fn of_class(class: &'static str) -> Self {
        Self { class }
    }

    fn record_peer_op(&self, g: &mut Graph, op_type: &str, attrs: Vec<AttributeProto>) -> Output {
        let slot_id = g.register_generic(self, "PeerSelectorRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: op_type.into(),
            domain: "ai.bytesandbrains.role.peer_selector".into(),
            input: vec![],
            output: vec![out_name.clone()],
            attribute: attrs,
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "PeerSelectorRuntime"),
                kv("ai.bytesandbrains.slot_id", &slot_id.to_string()),
                kv(bb_ir::peer_class::PEER_CLASS_KEY, self.class),
            ],
            ..Default::default()
        });
        g.declare_value_info(&out_name, &bb_ir::types::TYPE_PEER_ID);
        Output::new(out_name, &bb_ir::types::TYPE_PEER_ID)
    }

    /// `Sample(n) -> peers` - Shape 2 (state-dependent sampling).
    pub fn sample(&self, g: &mut Graph, n: i64) -> Output {
        self.record_peer_op(g, "Sample", vec![attr_int("n", n)])
    }

    /// `CurrentView() -> view` - Shape 2 (state read).
    pub fn current_view(&self, g: &mut Graph) -> Output {
        self.record_peer_op(g, "CurrentView", vec![])
    }
}

// ---------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------

/// Generic Protocol slot placeholder. Bind a concrete protocol at this
/// slot via the compiler chain
/// (`Compiler::new().bind_protocol::<T>("slot").compile(...)`).
///
/// Protocols have no user-facing DSL ops on the placeholder itself —
/// they're stateful control-plane runtimes that surface via
/// `dispatch_atomic` against their per-impl atomic opset
/// (`register_protocol!{}` emits the DSL methods for the impl's own
/// atomic opset). The placeholder exists solely so Modules can
/// declare a generic Protocol slot.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProtocolSlot;

