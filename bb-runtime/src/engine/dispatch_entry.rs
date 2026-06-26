//! Engine dispatch types
//!
//! `OpDispatch` is the install-time pre-stamped per-OpRef dispatch
//! kind. Each `GraphSlot.op_dispatch[i]` carries one of these,
//! resolved once by `Engine::resolve_dispatch` so runtime invoke is
//! one indirect lookup with no HashMap probes on hot path.

use std::rc::Rc;

use crate::atomic::DispatchResult;
use crate::bus::OpError;
use crate::engine::invoke::ProtocolDispatchFn;
use crate::ids::ComponentRef;
use crate::runtime::RuntimeResourceRef;
use crate::slot_value::SlotValue;
use bb_ir::proto::onnx::NodeProto;

/// Stateless syscall invoke fn pointer
/// Same input/output shape as a role-trait `dispatch_atomic` call;
/// returns `DispatchResult` for uniform handling by `invoke_one`.
pub type StatelessInvokeFn = fn(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError>;

/// Canonical key for a `FunctionProto` in `Node.model.functions[]`.
/// Matches ONNX's `(domain, name, overload)` tuple - the linker
/// dedupes on this key.
pub type FunctionKey = (String, String, String);

/// Per-OpRef dispatch decision, pre-stamped at install time by
/// `Engine::resolve_dispatch`. Runtime invoke is one indirect probe
/// against `GraphSlot.op_dispatch[idx]`. Four variants:
///
/// - `Stateless` - framework syscall.
/// - `Atomic` - bound runtime's `dispatch_atomic`.
/// - `FunctionCall` - splice into another installed function's body
///   via shared `OpRef`s, with input/output rename for call-frame
///   semantics. See `docs/ENGINE.md` §8.4.
/// - `Unresolved` - sentinel for nodes whose dispatch couldn't be
///   resolved at install. Build fails if any survive.
#[derive(Clone, Debug)]
pub enum OpDispatch {
    /// Framework syscall.
    Stateless(StatelessInvokeFn),

    /// Bound runtime impl, routed via `components[component_ref]`.
    /// `dispatch_fn` is the install-time-stamped downcast closure
    /// from `Engine::role_dispatchers[component_type_id]`. Runtime
    /// invoke calls the closure directly; resolve_dispatch only
    /// stamps `Atomic` when the closure is available, otherwise it
    /// stamps `Unresolved`. Test fixtures that bypass install
    /// stamp the closure manually via `bind_slot_id` +
    /// `register_<role>_dispatcher` before `resolve_dispatch`.
    Atomic {
        /// `ComponentRef` of the bound impl.
        component_ref: ComponentRef,
        /// Pre-stamped downcast closure that calls the concrete
        /// `<Role>Runtime::dispatch_atomic` on the bound component.
        dispatch_fn: ProtocolDispatchFn,
    },

    /// Function-call to another installed function. `target` keys
    /// into `engine.graphs` (the symbol table); `input_rename` /
    /// `output_rename` map caller-side ↔ formal value names.
    FunctionCall {
        /// `(domain, name, overload)` of the called function.
        target: FunctionKey,
        /// Pairs `(caller_value_name, formal_parameter_name)` zipped
        /// from this call NodeProto's `input` against the target
        /// function's `input` list.
        input_rename: Rc<[(String, String)]>,
        /// Pairs `(formal_output_name, caller_value_name)` zipped
        /// from the target function's `output` against this call's
        /// `output` list. Resolved into `output_forwarding` site-id
        /// pairs at install time.
        output_rename: Rc<[(String, String)]>,
    },

    /// Sentinel for unresolved dispatch - set by `from_function`
    /// before resolve, replaced by `resolve_dispatch`. Build fails if
    /// any survive past resolution.
    Unresolved,
}
