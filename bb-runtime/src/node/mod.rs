//! `Node` — the framework's runtime construct, installed via
//! [`bytesandbrains::install`] and driven through `Node::poll`.
//!
//! Lifecycle:
//!
//! ```ignore
//! let model    = MyModule.build()?;
//! let compiled = bb::Compiler::new()
//!     .bind_backend::<CpuBackend>("compute")
//!     .compile(model)?;
//! let mut node = bb::install(peer_id, addr, compiled, &["MyModule"], bb::Config::new())?;
//! loop { node.poll(&mut cx); }
//! ```
//!
//! Construction-time validation (passport check, binding-table
//! parse, concrete construction) surfaces `InstallError` at install
//! time — the engine is ready to dispatch by the time `Node::poll`
//! sees it.

use std::collections::HashMap;
use std::sync::Arc;

use prost::Message;

use crate::concrete::ComponentHandle;
use crate::engine::Engine;
use crate::errors::delivery::DeliveryError;
use crate::errors::restore::RestoreError;
use crate::snapshot::transient::TransientSnapshot;
use crate::snapshot::{NamedComponentSnapshot, NamedGraphSnapshot, NodeConfigSnapshot};
use bb_ir::proto::onnx::ModelProto;

/// Type-erased `EngineStep` observer installed onto a `Node` via
/// [`Node::set_telemetry_tap`]. Each `Node::poll` call invokes the
/// closure once per produced step.
pub type TelemetryTap = Box<dyn FnMut(&crate::engine::EngineStep)>;

/// Selector for [`Node::run_bootstrap`]. One enum, four variants
/// covering every bootstrap-kick shape — install-order, by Module
/// target name, by Module target name with staged inputs, and by
/// Component slot.
pub enum BootstrapTarget<'a> {
    /// Drive every install-order Module bootstrap target on this Node.
    /// Equivalent to the F4 host kick: arms + seeds the install-order
    /// queue, then drives the engine until every queued target reaches
    /// `BootstrapComplete` or one suspends on async.
    All,
    /// Drive specific Module bootstrap targets by name (with empty
    /// inputs). Surfaces `BootstrapError::UnknownTarget` when any name
    /// is not a registered Module bootstrap; the batch validates
    /// atomically before any staging happens. Useful when the host
    /// knows every target's bootstrap takes no formals.
    ModuleNames(&'a [&'a str]),
    /// Drive Module bootstrap targets with explicit inputs. Each
    /// `BootstrapRequest`'s `inputs` are validated against the
    /// target's declared formals; the framework copies each
    /// borrowed `&[u8]` into engine-owned storage (Principle 1a).
    /// On any validation failure the engine's bootstrap state stays
    /// untouched.
    ModuleRequests(&'a [crate::engine::BootstrapRequest<'a>]),
    /// Drive Component bootstraps by slot name. Each slot must resolve
    /// to a registered Component bootstrap; the batch validates
    /// atomically before any dispatch fires. Slots fire in slice
    /// order through the registered Bootstrap dispatcher.
    Slots(&'a [&'a str]),
}

/// Constructed BB Node ready to drive ML work. Produced by
/// [`bytesandbrains::install`] — by the time the host holds one,
/// the engine has resolved its dispatch table, registered every
/// bound concrete, and installed the target function as the root
/// graph.
pub struct Node {
    pub(crate) engine: Engine,
    pub(crate) config: NodeConfig,
    pub(crate) incarnation: u64,
    /// Registered target names → the shared compiled artifact. Every
    /// entry references the same underlying `ModelProto` so a
    /// multi-target install (e.g. `Client` + `Server` partitions on
    /// one peer) stores the proto bytes exactly once. The `Arc` is
    /// installed at [`Self::register_module`]; both `deliver_event`
    /// and `invoke` consult `contains_key` for routing, and future
    /// per-target input-name lookups can read through this shared
    /// handle without cloning the proto.
    pub(crate) module_index: HashMap<String, Arc<ModelProto>>,
    /// Single shared handle to the installed model. Equal to the
    /// `Arc` every `module_index` entry references; held separately
    /// so callers (and Debug) can reach the proto without first
    /// resolving a target name. `None` before
    /// [`Self::register_module`] runs.
    pub(crate) model: Option<Arc<ModelProto>>,
    pub(crate) component_handles: Vec<ComponentHandle>,
    /// Optional observer fed every `EngineStep` produced by
    /// `Engine::poll`. Set via [`Node::set_telemetry_tap`]; when
    /// `None` the engine's outputs pass through unchanged.
    pub(crate) telemetry_tap: Option<TelemetryTap>,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("incarnation", &self.incarnation)
            .field("modules", &self.module_index.keys().collect::<Vec<_>>())
            .field("component_count", &self.component_handles.len())
            .finish_non_exhaustive()
    }
}

// --- Construction + chain methods --------------

impl Node {
    /// Bare constructor used by [`bytesandbrains::install`]. Not
    /// part of the public surface — every external caller routes
    /// through `install` so the compilation passport, binding-table
    /// parse, and concrete construction surface a typed
    /// `InstallError` instead of leaving the Node in a half-built
    /// state. Doc-hidden to match the rest of the install-only
    /// surface (`engine_install_handle`, `push_linked_component`,
    /// `set_model`, `register_module`); the facade lives outside
    /// `bb-runtime`, so a `pub(crate)` constructor would not reach
    /// it — `#[doc(hidden)]` keeps the symbol off the rendered
    /// docs while letting the canonical `install` path call it.
    ///
    /// `addresses` is the ordered local-address bag registered in
    /// the engine's AddressBook against `peer_id`. The wire syscall
    /// resolves `peer = self.peer_id()` against the bag without an
    /// explicit `add_peer` call. An empty vec skips self-registration
    /// so downstream "no addresses" errors surface at the protocol
    /// level; passing `vec![Address::empty()]` registers a single
    /// placeholder entry that filters out as empty downstream.
    #[doc(hidden)]
    pub fn new(peer_id: crate::ids::PeerId, addresses: Vec<crate::framework::Address>) -> Self {
        let config = NodeConfig::new(peer_id);
        let mut engine = Engine::with_bus_capacity(config.bus_capacity);
        engine.self_peer = peer_id;
        engine.apply_config_caps(&config);
        engine.register_all_framework_syscalls();
        let real_addresses: Vec<crate::framework::Address> = addresses
            .into_iter()
            .filter(|a| a != &crate::framework::Address::empty())
            .collect();
        if !real_addresses.is_empty() {
            let _ = engine
                .framework
                .address_book
                .add_peer(peer_id, real_addresses);
        }
        Self {
            engine,
            config,
            incarnation: 0,
            module_index: HashMap::new(),
            model: None,
            component_handles: Vec::new(),
            telemetry_tap: None,
        }
    }

    /// Override the default [`NodeConfig`] (the cycle / async /
    /// outbound caps). Must be called before [`crate::install`]
    /// for the new caps to apply at build time.
    pub fn with_config(mut self, cfg: NodeConfig) -> Self {
        self.config = cfg.clone();
        self.engine.apply_config_caps(&cfg);
        self
    }
}

// --- Snapshot / restore / runtime methods -----

impl Node {
    /// Capture the snapshottable state. refuses to
    /// proceed when the bus still carries un-drained events that a
    /// restore would silently drop or re-fire stale; callers drive
    /// `Node::poll` to quiescence first and retry.
    pub fn snapshot(&self) -> Result<crate::snapshot::NodeSnapshot, crate::errors::SnapshotError> {
        // Bus quiescence guard. `len()` reads the queue without
        // draining; `dropped_since_last_drain` would only be non-zero
        // if a publish overflowed the cap since the last poll. Both
        // are surfaced so the host can decide whether the loss is
        // tolerable.
        let queued = self.engine.bus.len();
        if queued > 0 {
            return Err(crate::errors::SnapshotError::BusNotDrained { queued, dropped: 0 });
        }
        Ok(self.snapshot_inner())
    }

    fn snapshot_inner(&self) -> crate::snapshot::NodeSnapshot {
        let mut components: Vec<NamedComponentSnapshot> = Vec::new();
        for handle in &self.component_handles {
            let cref = crate::ids::ComponentRef::from(handle.instance_id);
            let Some(instance) = self.engine.component(cref) else {
                continue;
            };
            let state_bytes = (handle.serialize_fn)(instance);
            components.push(NamedComponentSnapshot {
                type_name: handle.type_name.to_string(),
                instance_id: handle.instance_id,
                package: handle.package,
                state_bytes,
            });
        }

        let mut graphs: Vec<NamedGraphSnapshot> = Vec::new();
        for (name, installed) in self.engine.graphs_named() {
            let function_proto_bytes = installed.function.encode_to_vec();
            graphs.push(NamedGraphSnapshot {
                name: name.to_string(),
                function_proto_bytes,
            });
        }

        let counters: std::collections::HashMap<String, u64> =
            self.engine.framework.counters.clone();

        let event_subscriptions: std::collections::HashMap<String, Vec<u64>> = self
            .engine
            .event_subscriptions
            .iter()
            .map(|(kind, sites)| {
                (
                    kind.clone(),
                    sites.iter().map(|s| s.as_u64()).collect::<Vec<u64>>(),
                )
            })
            .collect();

        let address_book: Vec<crate::snapshot::transient::AddressBookEntrySnapshot> = self
            .engine
            .framework
            .address_book
            .iter()
            .map(
                |(peer, addrs, ref_count)| crate::snapshot::transient::AddressBookEntrySnapshot {
                    peer_id: peer.to_bytes(),
                    addresses: addrs.iter().map(|a| a.to_bytes()).collect(),
                    ref_count,
                },
            )
            .collect();

        let peer_governor = capture_peer_governor(&self.engine.framework.peer_state.governor);
        let backoff_table = self
            .engine
            .framework
            .peer_state
            .backoff
            .iter()
            .map(|(p, s)| crate::snapshot::transient::BackoffEntry {
                peer: p.to_bytes(),
                attempts: s.attempts,
                last_attempt_ns: s.last_attempt_ns,
                next_retry_ns: s.next_retry_ns,
            })
            .collect();
        let pending_async = self
            .engine
            .exec
            .pending_async
            .iter()
            .map(|(cmd, p)| {
                (
                    cmd.as_u64(),
                    crate::snapshot::transient::PendingAsyncSnapshot {
                        op_ref: p.op_ref.as_u64(),
                        exec_id: p.exec_id.as_u64(),
                        output_sites: p.output_sites.iter().map(|s| s.as_u64()).collect(),
                        deadline_ns: p.deadline_ns,
                    },
                )
            })
            .collect();
        let pending_outbound = self
            .engine
            .framework
            .outbound_queue
            .iter_for_snapshot()
            .map(|env| crate::snapshot::transient::PendingOutboundEntry {
                envelope_bytes: crate::envelope::EnvelopeCodec::encode(env),
                redelivered: true,
            })
            .collect();

        let transient = TransientSnapshot {
            framework: crate::snapshot::transient::FrameworkSnapshot {
                counters,
                fired_phases: self.engine.fired_phases.clone(),
                address_book,
                peer_governor,
                backoff_table,
                pending_outbound,
                // multihash bytes + counter
                // persistence. The peer_id_bytes lets a multihash
                // PeerId survive restore; the counters prevent
                // CommandId / ExecId collisions on restart.
                peer_id_bytes: self.engine.self_peer.to_bytes(),
                next_command_id: self.engine.exec.ids.next_command_id,
                next_exec_id: self.engine.exec.ids.next_exec_id,
                spec_version: crate::snapshot::transient::CURRENT_SNAPSHOT_SPEC_VERSION,
            },
            bus: crate::snapshot::transient::TypedBusSnapshot {
                event_subscriptions,
            },
            pending_async,
            ..Default::default()
        };

        crate::snapshot::NodeSnapshot {
            incarnation: self.incarnation,
            config: NodeConfigSnapshot::from(&self.config),
            graphs,
            components,
            transient,
        }
    }

    /// Restore a Node's state from a snapshot.
    pub fn restore(
        &mut self,
        snap: crate::snapshot::NodeSnapshot,
    ) -> Result<(), crate::errors::restore::RestoreError> {
        if snap.transient.framework.spec_version
            != crate::snapshot::transient::CURRENT_SNAPSHOT_SPEC_VERSION
        {
            return Err(RestoreError::SpecVersionMismatch {
                got: snap.transient.framework.spec_version,
                expected: crate::snapshot::transient::CURRENT_SNAPSHOT_SPEC_VERSION,
            });
        }

        // Reinstall each captured graph. The function bytes are
        // prost-encoded FunctionProto blobs from the source Node's
        // snapshot path; decode failures fail the restore loudly
        // so callers don't silently lose a graph.
        for graph_snap in snap.graphs {
            let function = bb_ir::proto::onnx::FunctionProto::decode(
                graph_snap.function_proto_bytes.as_slice(),
            )
            .map_err(|e| {
                RestoreError::SnapshotMismatch(format!(
                    "restore: failed to decode graph `{}`: {e}",
                    graph_snap.name,
                ))
            })?;
            self.engine.install_graph(graph_snap.name, function);
        }

        for comp_snap in snap.components {
            let cref = crate::ids::ComponentRef::from(comp_snap.instance_id);
            let Some(handle) = self.component_handles.iter().find(|h| {
                h.type_name == comp_snap.type_name && h.instance_id == comp_snap.instance_id
            }) else {
                return Err(RestoreError::SnapshotMismatch(format!(
                    "no handle on live Node for component {}@{}",
                    comp_snap.type_name, comp_snap.instance_id,
                )));
            };
            let restored = (handle.restore_fn)(&comp_snap.state_bytes).map_err(|source| {
                RestoreError::ComponentRestoreFailed {
                    type_name: comp_snap.type_name.clone(),
                    source,
                }
            })?;
            self.engine.register_component(cref, restored);
        }

        self.engine.framework.counters.clear();
        for (name, value) in snap.transient.framework.counters {
            self.engine.framework.counters.insert(name, value);
        }
        self.engine.fired_phases = snap.transient.framework.fired_phases;
        self.engine.event_subscriptions.clear();
        for (kind, sites) in snap.transient.bus.event_subscriptions {
            self.engine.event_subscriptions.insert(
                kind,
                sites
                    .into_iter()
                    .map(crate::ids::NodeSiteId::from)
                    .collect(),
            );
        }

        for entry in snap.transient.framework.address_book {
            let mut decoded: Vec<crate::framework::Address> =
                Vec::with_capacity(entry.addresses.len());
            let mut malformed = false;
            for bytes in &entry.addresses {
                match crate::framework::Address::from_bytes(bytes) {
                    Ok(a) => decoded.push(a),
                    Err(_) => {
                        malformed = true;
                        break;
                    }
                }
            }
            if malformed || decoded.is_empty() {
                continue;
            }
            let Ok(peer) = crate::ids::PeerId::from_bytes(&entry.peer_id) else {
                continue;
            };
            self.engine
                .framework
                .address_book
                .restore_entry(peer, decoded, entry.ref_count);
        }
        self.engine.framework.peer_state.governor = crate::framework::PeerGovernor::new();
        let governor_snap = snap.transient.framework.peer_governor;
        self.engine
            .framework
            .peer_state
            .governor
            .set_failure_threshold(governor_snap.failure_threshold);
        for peer_bytes in governor_snap.blocklist {
            let Ok(peer) = crate::ids::PeerId::from_bytes(&peer_bytes) else {
                continue;
            };
            self.engine.framework.peer_state.governor.block(peer);
        }
        if let Some(allow) = governor_snap.allowlist {
            let set: std::collections::HashSet<_> = allow
                .into_iter()
                .filter_map(|b| crate::ids::PeerId::from_bytes(&b).ok())
                .collect();
            self.engine
                .framework
                .peer_state
                .governor
                .set_allowlist(Some(set));
        }
        for (peer_bytes, consecutive_failures, last_event_ns, down) in governor_snap.health {
            let Ok(peer) = crate::ids::PeerId::from_bytes(&peer_bytes) else {
                continue;
            };
            self.engine.framework.peer_state.governor.restore_health(
                peer,
                crate::framework::PeerHealth {
                    consecutive_failures,
                    last_event_ns,
                    down,
                },
            );
        }
        for entry in snap.transient.framework.backoff_table {
            let Ok(peer) = crate::ids::PeerId::from_bytes(&entry.peer) else {
                continue;
            };
            self.engine.framework.peer_state.backoff.restore_state(
                peer,
                crate::framework::backoff_table::BackoffState {
                    attempts: entry.attempts,
                    last_attempt_ns: entry.last_attempt_ns,
                    next_retry_ns: entry.next_retry_ns,
                },
            );
        }
        for (cmd_u64, snap_p) in snap.transient.pending_async {
            self.engine.exec.pending_async.insert(
                crate::ids::CommandId::from(cmd_u64),
                crate::engine::PendingAsync {
                    op_ref: crate::ids::OpRef::from(snap_p.op_ref),
                    exec_id: crate::ids::ExecId::from(snap_p.exec_id),
                    output_sites: snap_p
                        .output_sites
                        .into_iter()
                        .map(crate::ids::NodeSiteId::from)
                        .collect(),
                    deadline_ns: snap_p.deadline_ns,
                },
            );
        }
        for entry in snap.transient.framework.pending_outbound {
            if let Ok(env) = crate::envelope::EnvelopeCodec::decode_capped(
                &entry.envelope_bytes,
                &self.config.envelope_caps,
            ) {
                self.engine.framework.outbound_queue.push(env);
            }
        }

        self.incarnation = snap.incarnation + 1;

        Ok(())
    }

    /// Reset transient runtime state without dropping the bound
    /// components or installed graphs.
    pub fn clear(&mut self) {
        self.engine.exec.frontier.clear();
        self.engine.exec.slot_table.clear();
        self.engine.exec.execution_state.clear();
        self.engine.exec.pending_async.clear();
        self.engine.exec.pending_completions.clear();
        let _ = self.engine.ingress.drain_all();
        self.engine.fired_phases.clear();
        self.engine.framework.counters.clear();
    }

    /// Current incarnation count. Bumped on every `restore()`.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// Names of every module registered at construction time.
    pub fn loaded_modules(&self) -> Vec<&str> {
        self.module_index.keys().map(|s| s.as_str()).collect()
    }

    /// References to every owned component handle.
    pub fn linked_components(&self) -> Vec<&ComponentHandle> {
        self.component_handles.iter().collect()
    }

    /// Local peer identity.
    pub fn peer_id(&self) -> crate::ids::PeerId {
        self.config.peer_id
    }

    /// Read-only execution-state lookup.
    pub fn execution_state(
        &self,
        exec_id: crate::ids::ExecId,
    ) -> Option<&crate::engine::ExecutionState> {
        self.engine.exec.execution_state.get(&exec_id)
    }

    /// Number of `AsyncSuspended` Ops currently awaiting completion.
    pub fn pending_async_count(&self) -> usize {
        self.engine.exec.pending_async.len()
    }

    /// Snapshot of the engine's hot-path counters.
    pub fn engine_stats(&self) -> crate::engine::EngineStats {
        self.engine.engine_stats()
    }

    /// Mutable handle to the engine for install-time setup
    /// (registering components, binding slots, installing the
    /// target graph, resolving dispatch). Doc-hidden because it
    /// crosses the `pub(crate)` field boundary: the user-facing
    /// entry point is `bb::install()`, which calls this accessor
    /// to drive `engine.register_component` / `engine.bind_slot` /
    /// `engine.install_graph` / `engine.resolve_dispatch`.
    #[doc(hidden)]
    pub fn engine_install_handle(&mut self) -> &mut crate::engine::Engine {
        &mut self.engine
    }

    /// Push a `ComponentHandle` onto the linked-components list.
    /// Called from `bb::install()` per supplied component so
    /// `snapshot()` can capture state via the recorded
    /// `serialize_fn`. Doc-hidden — the public surface for setting
    /// up a Node is `bb::install()`.
    #[doc(hidden)]
    pub fn push_linked_component(&mut self, handle: crate::concrete::ComponentHandle) {
        self.component_handles.push(handle);
    }

    /// Set the shared compiled artifact this Node was installed with.
    /// Called once per `bb::install()` from the install path before
    /// the per-target `register_module` calls run; downstream
    /// `register_module` calls share the same `Arc` instead of
    /// cloning the proto per target. Doc-hidden — the user surface
    /// is `bb::install()`.
    #[doc(hidden)]
    pub fn set_model(&mut self, model: ModelProto) {
        self.model = Some(Arc::new(model));
    }

    /// Register `module_name` as a valid target for `deliver_event` /
    /// `invoke`. Called from `bb::install()` once per resolved
    /// target name after the engine's root graph is installed; the
    /// stored `Arc` is the same handle [`Self::set_model`] minted
    /// for every target, so multi-target installs hold the proto
    /// bytes exactly once. Doc-hidden — the user surface is
    /// `bb::install()`.
    #[doc(hidden)]
    pub fn register_module(&mut self, module_name: String) {
        let model = self
            .model
            .clone()
            .expect("Node::set_model must run before register_module");
        self.module_index.insert(module_name, model);
    }

    /// The compiled `ModelProto` the Node was installed with, or
    /// `None` if the Node was built internally by a crate-private
    /// path that bypassed [`crate::install`] (the public entry point
    /// always sets a model). Returns a shared `Arc` so callers can
    /// read per-target metadata without cloning the proto.
    pub fn model(&self) -> Option<Arc<ModelProto>> {
        self.model.clone()
    }

    /// Look up the `ComponentRef` bound at `slot_name` in the
    /// engine's generic slot registry. Returns `None` when no slot
    /// of that name is bound.
    pub fn slot(&self, slot_name: &str) -> Option<crate::ids::ComponentRef> {
        self.engine.slot(slot_name)
    }

    /// Iterate every `(slot_name, ComponentRef)` pair bound in the
    /// engine's slot registry.
    pub fn slots_iter(&self) -> impl Iterator<Item = (&str, crate::ids::ComponentRef)> {
        self.engine.slots_iter()
    }

    /// Inventory-declared roles for a registered component. Read
    /// from the per-Engine `component_roles` map populated by
    /// `ensure_ready` from
    /// `crate::registry::roles_for_component(T::TYPE_NAME)`. Returns
    /// the empty set when the component wasn't registered via a
    /// `#[derive(bb::<Role>)]` chain (test fixtures that hand-implement
    /// role traits surface here as empty).
    pub fn roles_for(
        &self,
        cref: crate::ids::ComponentRef,
    ) -> std::collections::HashSet<crate::registry::ComponentRole> {
        self.engine.roles_for(cref)
    }

    /// Add `peer` to the inbound + outbound blocklist.
    pub fn block_peer(&mut self, peer: crate::ids::PeerId) {
        self.engine.framework.peer_state.governor.block(peer);
    }

    /// Remove `peer` from the blocklist.
    pub fn unblock_peer(&mut self, peer: crate::ids::PeerId) {
        self.engine.framework.peer_state.governor.unblock(peer);
    }

    /// Configure an allowlist for inbound + outbound delivery.
    pub fn set_allowlist(
        &mut self,
        allowlist: Option<std::collections::HashSet<crate::ids::PeerId>>,
    ) {
        self.engine
            .framework
            .peer_state
            .governor
            .set_allowlist(allowlist);
    }

    /// Per-peer health snapshot.
    pub fn peer_health(&self, peer: crate::ids::PeerId) -> Option<crate::framework::PeerHealth> {
        self.engine.framework.peer_state.governor.peer_health(peer)
    }

    /// Resolve a `PeerId` to its ordered address list (cloned).
    pub fn resolve_peer_addresses(
        &self,
        peer: crate::ids::PeerId,
    ) -> Option<Vec<crate::framework::Address>> {
        self.engine
            .framework
            .address_book
            .lookup(peer)
            .map(|addrs| addrs.to_vec())
    }

    /// Install an `EngineStep` observer onto the live Node.
    pub fn set_telemetry_tap<F>(&mut self, tap: F)
    where
        F: FnMut(&crate::engine::EngineStep) + 'static,
    {
        self.telemetry_tap = Some(Box::new(tap));
    }

    /// Announce a peer with one or more reachable addresses.
    pub fn add_peer(
        &mut self,
        peer: crate::ids::PeerId,
        addresses: Vec<crate::framework::Address>,
    ) -> Result<(), crate::framework::AddressBookError> {
        self.engine.framework.address_book.add_peer(peer, addresses)
    }

    /// Concern 2 - cheap-clone handle to the shared
    /// `IngressQueue`. Used by off-thread transport / clock
    /// adapters to push events into the Node.
    pub fn ingress_handle(&self) -> crate::ingress::IngressQueueRef {
        crate::ingress::IngressQueueRef::new(self.engine.ingress_queue_handle())
    }

    /// Read-only view on the registered NodeConfig.
    pub fn config(&self) -> &NodeConfig {
        &self.config
    }

    /// First entry of the local-address bag, or [`Address::empty`]
    /// when none registered. The AddressBook entry for
    /// `self.peer_id()` is the source of truth.
    pub fn peer_address(&self) -> crate::framework::Address {
        self.local_addresses()
            .first()
            .cloned()
            .unwrap_or_else(crate::framework::Address::empty)
    }

    /// Ordered local-address bag for this Node. Reads directly from
    /// the AddressBook entry keyed by `self.peer_id()`; returns an
    /// empty slice when the Node was installed with no addresses or
    /// the bag was pruned to zero entries.
    pub fn local_addresses(&self) -> &[crate::framework::Address] {
        self.engine
            .framework
            .address_book
            .lookup(self.engine.self_peer)
            .unwrap_or(&[])
    }

    /// Append `address` to the local-address bag. Creates the
    /// AddressBook entry if none exists; idempotent on duplicates.
    pub fn add_local_address(
        &mut self,
        address: crate::framework::Address,
    ) -> Result<(), crate::framework::AddressBookError> {
        let self_peer = self.engine.self_peer;
        let book = &mut self.engine.framework.address_book;
        if book.lookup(self_peer).is_some() {
            book.register_address(self_peer, address)
        } else {
            book.add_peer(self_peer, vec![address])
        }
    }

    /// Prune `address` from the local-address bag. Errors when no
    /// entry exists; succeeds (no-op) when the bag has no matching
    /// address.
    pub fn forget_local_address(
        &mut self,
        address: &crate::framework::Address,
    ) -> Result<(), crate::framework::AddressBookError> {
        let self_peer = self.engine.self_peer;
        self.engine
            .framework
            .address_book
            .forget_address(self_peer, address)
    }

    /// Drive bootstrap targets to completion. Returns every
    /// `EngineStep` the bootstrap path emitted, including each
    /// drained phase's `BootstrapComplete` or the yielding
    /// `WaitingOnBootstrap`. Idempotent: a Node whose bootstrap
    /// already completed (or a Node with no install-order recording)
    /// returns an empty vec.
    ///
    /// Variant semantics:
    ///
    /// - [`BootstrapTarget::All`] arms + seeds the install-order
    ///   queue, then polls until every queued target reaches
    ///   `BootstrapComplete` or one suspends on an async completion.
    ///   Equivalent to the F4 host kick.
    /// - [`BootstrapTarget::ModuleNames`] / [`BootstrapTarget::ModuleRequests`]
    ///   validate target names + input formals atomically — the whole
    ///   batch rejects with `BootstrapError::UnknownTarget` /
    ///   `BootstrapError::AlreadyTransitivelyQueued` /
    ///   `BootstrapError::UnknownInput` / `BootstrapError::MissingInput`
    ///   before any staging happens. On success the framework copies
    ///   each request's borrowed `&[u8]` inputs into engine-owned
    ///   storage (Principle 1a: caller slices may drop the moment
    ///   this call returns), then drives the staged bodies to
    ///   completion.
    /// - [`BootstrapTarget::Slots`] resolves each slot to a registered
    ///   Component bootstrap, validates the whole batch atomically,
    ///   then fires each in slice order through the dispatcher. The
    ///   drain loop runs after all slots fire so synchronous
    ///   `Immediate` dispatches retire inline and async ones surface
    ///   `WaitingOnBootstrap`.
    ///
    /// On async suspension the host posts the matured completion via
    /// the ingress and re-invokes `run_bootstrap` to drain the
    /// resumed op.
    ///
    /// Body-phase ops touching a locked `ComponentRef` do not fire
    /// during this call; the `is_op_locked` gate in
    /// `pop_frontier_fireable` keeps them parked until the bootstrap
    /// in-flight set drops.
    pub fn run_bootstrap(
        &mut self,
        target: BootstrapTarget<'_>,
    ) -> Result<Vec<crate::engine::EngineStep>, crate::errors::BootstrapError> {
        match target {
            BootstrapTarget::All => {
                // Arm + seed the install-order queue (no-op when no
                // targets remain). `Engine::run_bootstrap` returns
                // false on a fully drained Node so the drain loop
                // exits immediately.
                let armed = self.engine.run_bootstrap();
                if !armed && !self.engine.bootstrap_pending() {
                    return Ok(Vec::new());
                }
            }
            BootstrapTarget::ModuleNames(targets) => {
                let requests: Vec<crate::engine::BootstrapRequest<'_>> = targets
                    .iter()
                    .map(|t| crate::engine::BootstrapRequest {
                        target: t,
                        inputs: &[],
                    })
                    .collect();
                self.stage_module_requests(&requests)?;
            }
            BootstrapTarget::ModuleRequests(requests) => {
                self.stage_module_requests(requests)?;
            }
            BootstrapTarget::Slots(slots) => {
                // Pre-flight: every slot must resolve to a registered
                // Component bootstrap. Atomic — no firing happens
                // until every slot in the batch passes.
                for slot in slots {
                    if !self.engine.has_component_bootstrap(slot) {
                        return Err(crate::errors::BootstrapError::UnknownTarget {
                            target_name: (*slot).to_string(),
                            available: self
                                .engine
                                .module_bootstrap_target_names()
                                .into_iter()
                                .chain(std::iter::empty())
                                .collect(),
                        });
                    }
                }
                for slot in slots {
                    self.engine.fire_component_bootstrap_by_slot(slot)?;
                }
            }
        }
        Ok(self.drain_bootstrap())
    }

    /// Pre-flight + stage a batch of `BootstrapRequest`s through the
    /// engine's F5 immediate-fire entry point. Validates target
    /// registration and duplicate detection atomically before any
    /// per-input copy happens; on any error path the engine's
    /// bootstrap state stays untouched.
    fn stage_module_requests(
        &mut self,
        requests: &[crate::engine::BootstrapRequest<'_>],
    ) -> Result<(), crate::errors::BootstrapError> {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for req in requests {
            if !self.engine.module_bootstrap_registered(req.target) {
                return Err(crate::errors::BootstrapError::UnknownTarget {
                    target_name: req.target.to_string(),
                    available: self.engine.module_bootstrap_target_names(),
                });
            }
            if !seen.insert(req.target) {
                return Err(crate::errors::BootstrapError::AlreadyTransitivelyQueued {
                    target_name: req.target.to_string(),
                });
            }
        }
        for req in requests {
            // Borrow each request through its declared lifetime; the
            // engine copies bytes into framework-owned storage so the
            // borrow ends when this call returns.
            self.enqueue_bootstrap_request(crate::engine::BootstrapRequest {
                target: req.target,
                inputs: req.inputs,
            })?;
        }
        Ok(())
    }

    /// Drive the engine's poll loop until every staged bootstrap
    /// drains to `BootstrapComplete` or one suspends with
    /// `WaitingOnBootstrap`. Forwards each step through the
    /// telemetry tap (matching `Node::poll` semantics).
    fn drain_bootstrap(&mut self) -> Vec<crate::engine::EngineStep> {
        let mut steps = Vec::new();
        loop {
            let batch = self.engine.poll();
            let waiting = matches!(
                batch.last(),
                Some(crate::engine::EngineStep::WaitingOnBootstrap)
            );
            if let Some(tap) = self.telemetry_tap.as_mut() {
                for step in &batch {
                    tap(step);
                }
            }
            steps.extend(batch);
            if !self.engine.bootstrap_pending() || waiting {
                break;
            }
        }
        steps
    }

    /// Stage a host-supplied [`crate::engine::BootstrapRequest`] +
    /// fire the addressed Module bootstrap immediately. Crate-internal
    /// helper for [`Self::run_bootstrap`]'s
    /// [`BootstrapTarget::ModuleRequests`] arm. Forwards to the
    /// engine's F5 immediate-fire entry point: validates
    /// `request.inputs` against the target's declared formals, runs
    /// the Principle 1a copy (`try_charge` → `try_reserve_exact` →
    /// `extend_from_slice`) against the engine's ingress byte budget,
    /// and pushes the body ops onto the frontier. The caller's
    /// borrowed `&[u8]` slices may drop the moment this call returns.
    ///
    /// Returns `BootstrapError::UnknownTarget` /
    /// `BootstrapError::UnknownInput` / `BootstrapError::MissingInput`
    /// when the request fails validation, and
    /// `BootstrapError::AllocationFailed` when the engine's budget /
    /// allocator rejects a staged payload. On any error path the
    /// engine's bootstrap state stays untouched — partially admitted
    /// byte charges are released before the error surfaces.
    pub(crate) fn enqueue_bootstrap_request(
        &mut self,
        request: crate::engine::BootstrapRequest<'_>,
    ) -> Result<(), crate::errors::BootstrapError> {
        self.engine.enqueue_bootstrap_request(request)
    }

    /// Snapshot of the engine's bootstrap lifecycle. Returns
    /// `BootstrapStatus::Idle` when no bootstrap is queued or
    /// in-flight, `Running` when at least one bootstrap is in-flight
    /// (the body gate is parking touched ops), and `WaitingForInput`
    /// when the install-order queue still has unseeded targets or
    /// host-staged requests sit on `pending_requests` / `waiting`.
    /// Pure read — does not advance any queue.
    pub fn bootstrap_status(&self) -> crate::engine::bootstrap::BootstrapStatus {
        self.engine.bootstrap_status()
    }

    /// Drive one poll cycle. Returns `Pending` when the engine
    /// drains to quiescence (the ingress waker is registered with
    /// `cx`); otherwise yields the steps the engine made progress
    /// on. Construction-time validation lives in [`crate::node::Node`]'s
    /// installation path (see `bytesandbrains::install`) and surfaces
    /// `InstallError` before the first poll.
    pub fn poll(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Vec<crate::engine::EngineStep>> {
        let steps = self.engine.poll();
        if let Some(tap) = self.telemetry_tap.as_mut() {
            for step in &steps {
                tap(step);
            }
        }
        if steps.is_empty() {
            self.engine.ingress.register_waker(cx.waker());
            return std::task::Poll::Pending;
        }
        std::task::Poll::Ready(steps)
    }

    /// Decode and push inbound wire bytes onto the ingress queue.
    /// Routes through `EnvelopeCodec::decode_capped` so malformed,
    /// schema-mismatched, or oversize buffers fail with
    /// `DeliveryError::InvalidEnvelope` BEFORE any prost allocation.
    /// The transport layer supplies `src_peer` so the engine can
    /// consult `PeerGovernor::check_inbound` before routing.
    pub fn deliver_inbound(
        &mut self,
        src_peer: crate::ids::PeerId,
        bytes: &[u8],
    ) -> Result<(), crate::errors::delivery::DeliveryError> {
        let envelope =
            crate::envelope::EnvelopeCodec::decode_capped(bytes, &self.config.envelope_caps)
                .map_err(|e| {
                    crate::errors::delivery::DeliveryError::InvalidEnvelope(e.to_string())
                })?;
        self.engine
            .ingress
            .push(crate::ingress::IngressEvent::EnvelopeFrom {
                src_peer,
                envelope,
                // The raw-bytes deliver path doesn't surface an
                // observed address — host adapters that want
                // reflexive-address discovery push via the typed
                // ingress queue directly.
                src_observed_address: None,
            })
            .map_err(|_| crate::errors::delivery::DeliveryError::IngressClosed)
    }

    /// Push an app-event onto the ingress queue.
    ///
    /// Per Principle 1a (`docs/internal/superpowers/specs/2026-06-24-engine-boundary-fallibility-and-backend-owned-tensors.md`)
    /// the byte payload crosses the engine boundary as a BORROWED
    /// slice: the framework caps `value_bytes.len()` against
    /// `NodeConfig::max_app_event_bytes`, charges the length against
    /// `NodeConfig::ingress_byte_budget`, fallibly reserves a fresh
    /// framework-owned `Vec<u8>`, and copies the caller's bytes in.
    /// The caller may free `value_bytes` the moment this call
    /// returns. Cap / budget / alloc failures return a synchronous
    /// `DeliveryError::*` AND publish a matching
    /// `InfraEvent::AppIngressError` on the bus for observers.
    pub fn deliver_event(
        &mut self,
        module: &str,
        input: &str,
        value_bytes: &[u8],
    ) -> Result<(), crate::errors::delivery::DeliveryError> {
        if !self.module_index.contains_key(module) {
            return Err(DeliveryError::UnknownModule(module.to_string()));
        }
        let byte_count = value_bytes.len();
        let cap = self.config.max_app_event_bytes;
        let source = || crate::bus::AppIngressSource::AppEvent {
            module: module.to_string(),
            input: input.to_string(),
        };
        if byte_count > cap {
            self.emit_app_ingress_error(
                source(),
                byte_count,
                crate::bus::AppIngressErrorKind::PerItemCapExceeded { cap },
            );
            return Err(DeliveryError::OversizePayload { byte_count, cap });
        }
        if let Err(reason) = self.engine.try_charge(byte_count) {
            self.emit_app_ingress_error(
                source(),
                byte_count,
                crate::bus::AppIngressErrorKind::BudgetExceeded {
                    budget_remaining: reason.budget_remaining,
                },
            );
            return Err(DeliveryError::BudgetExceeded {
                byte_count,
                budget_remaining: reason.budget_remaining,
            });
        }
        let mut owned: Vec<u8> = Vec::new();
        // Route through `crate::fallible::try_reserve_exact` so the
        // thread-local fault seam (used by `tests/fallible_ingress.rs`)
        // intercepts the boundary reservation.
        if crate::fallible::try_reserve_exact(&mut owned, byte_count).is_err() {
            // Release the budget charge before surfacing the alloc
            // failure — the bytes never made it past the boundary.
            self.engine.release(byte_count);
            self.emit_app_ingress_error(
                source(),
                byte_count,
                crate::bus::AppIngressErrorKind::AllocationFailed {
                    reason: crate::bus::AllocFailReason::HeapExhausted,
                },
            );
            return Err(DeliveryError::AllocationFailed {
                byte_count,
                reason: crate::bus::AllocFailReason::HeapExhausted,
            });
        }
        owned.extend_from_slice(value_bytes);
        self.engine
            .ingress
            .push(crate::ingress::IngressEvent::AppEvent {
                module_name: module.to_string(),
                input_name: input.to_string(),
                value_bytes: owned,
            })
            .map_err(|_| {
                // Ingress queue closed — release the charge so the
                // counter does not leak the rejected push.
                self.engine.release(byte_count);
                DeliveryError::IngressClosed
            })
    }

    /// Invoke a Module with the given pre-encoded inputs.
    ///
    /// Inputs cross as borrowed `&[(&str, &[u8])]` per Principle 1a;
    /// the framework caps count + cumulative bytes, charges against
    /// `ingress_byte_budget`, and per-input fallibly reserves +
    /// copies into framework-owned `Vec<u8>` storage. Any failure
    /// releases prior charges and emits the matching
    /// `InfraEvent::AppIngressError`.
    pub fn invoke(
        &mut self,
        module: &str,
        inputs: &[(&str, &[u8])],
    ) -> Result<crate::ids::ExecId, crate::errors::delivery::DeliveryError> {
        if !self.module_index.contains_key(module) {
            return Err(DeliveryError::UnknownModule(module.to_string()));
        }
        let input_count = inputs.len();
        let input_cap = self.config.max_invoke_inputs;
        let source = || crate::bus::AppIngressSource::Invoke {
            module: module.to_string(),
            input_count,
        };
        if input_count > input_cap {
            // `byte_count` reports the requested input count for the
            // count-cap rejection — the byte cap will surface
            // separately below for cumulative-byte rejections.
            self.emit_app_ingress_error(
                source(),
                input_count,
                crate::bus::AppIngressErrorKind::PerItemCapExceeded { cap: input_cap },
            );
            return Err(DeliveryError::TooManyInputs {
                count: input_count,
                cap: input_cap,
            });
        }
        let total_bytes: usize = inputs
            .iter()
            .fold(0usize, |acc, (_, b)| acc.saturating_add(b.len()));
        let bytes_cap = self.config.max_invoke_bytes;
        if total_bytes > bytes_cap {
            self.emit_app_ingress_error(
                source(),
                total_bytes,
                crate::bus::AppIngressErrorKind::PerItemCapExceeded { cap: bytes_cap },
            );
            return Err(DeliveryError::OversizePayload {
                byte_count: total_bytes,
                cap: bytes_cap,
            });
        }
        if let Err(reason) = self.engine.try_charge(total_bytes) {
            self.emit_app_ingress_error(
                source(),
                total_bytes,
                crate::bus::AppIngressErrorKind::BudgetExceeded {
                    budget_remaining: reason.budget_remaining,
                },
            );
            return Err(DeliveryError::BudgetExceeded {
                byte_count: total_bytes,
                budget_remaining: reason.budget_remaining,
            });
        }
        // Per-input fallible reservation. Track how many bytes we've
        // already copied so a mid-loop failure can release the full
        // `total_bytes` charge in one shot (the slot-table never
        // observed any of these payloads).
        let mut owned: Vec<(String, Vec<u8>)> = Vec::new();
        if crate::fallible::try_reserve_exact(&mut owned, input_count).is_err() {
            self.engine.release(total_bytes);
            self.emit_app_ingress_error(
                source(),
                total_bytes,
                crate::bus::AppIngressErrorKind::AllocationFailed {
                    reason: crate::bus::AllocFailReason::HeapExhausted,
                },
            );
            return Err(DeliveryError::AllocationFailed {
                byte_count: total_bytes,
                reason: crate::bus::AllocFailReason::HeapExhausted,
            });
        }
        for (name, bytes) in inputs.iter() {
            let mut buf: Vec<u8> = Vec::new();
            if crate::fallible::try_reserve_exact(&mut buf, bytes.len()).is_err() {
                self.engine.release(total_bytes);
                self.emit_app_ingress_error(
                    source(),
                    total_bytes,
                    crate::bus::AppIngressErrorKind::AllocationFailed {
                        reason: crate::bus::AllocFailReason::HeapExhausted,
                    },
                );
                return Err(DeliveryError::AllocationFailed {
                    byte_count: total_bytes,
                    reason: crate::bus::AllocFailReason::HeapExhausted,
                });
            }
            buf.extend_from_slice(bytes);
            owned.push(((*name).to_string(), buf));
        }
        let exec_id = self.engine.allocate_exec_id();
        self.engine
            .ingress
            .push(crate::ingress::IngressEvent::Invoke {
                module_name: module.to_string(),
                inputs: owned,
                exec_id,
            })
            .map_err(|_| {
                self.engine.release(total_bytes);
                DeliveryError::IngressClosed
            })?;
        Ok(exec_id)
    }

    /// Publish a freshly-built [`crate::bus::InfraEvent::AppIngressError`]
    /// onto the in-Node bus. Internal helper used by the
    /// application-ingress entry points (`deliver_event`, `invoke`)
    /// to mirror their synchronous `DeliveryError` returns on the
    /// observer surface.
    fn emit_app_ingress_error(
        &mut self,
        source: crate::bus::AppIngressSource,
        byte_count: usize,
        kind: crate::bus::AppIngressErrorKind,
    ) {
        self.engine.bus.publish(crate::bus::NodeEvent::Infra(
            crate::bus::InfraEvent::AppIngressError {
                source,
                byte_count,
                kind,
            },
        ));
    }
}

/// Snapshot the PeerGovernor's policy + health state.
fn capture_peer_governor(
    governor: &crate::framework::PeerGovernor,
) -> crate::snapshot::transient::PeerGovernorSnapshot {
    crate::snapshot::transient::PeerGovernorSnapshot {
        blocklist: governor.blocklist().iter().map(|p| p.to_bytes()).collect(),
        allowlist: governor
            .allowlist()
            .map(|s| s.iter().map(|p| p.to_bytes()).collect()),
        health: governor
            .iter_health()
            .map(|(p, h)| {
                (
                    p.to_bytes(),
                    h.consecutive_failures,
                    h.last_event_ns,
                    h.down,
                )
            })
            .collect(),
        failure_threshold: governor.failure_threshold(),
    }
}

pub mod config;
pub mod derivation;
pub use config::{
    NodeConfig, DEFAULT_BUS_CAPACITY, DEFAULT_CYCLE_OP_BUDGET, DEFAULT_INGRESS_BYTE_BUDGET,
    DEFAULT_MAX_APP_EVENT_BYTES, DEFAULT_MAX_COMPLETION_RESULT_BYTES, DEFAULT_MAX_INVOKE_BYTES,
    DEFAULT_MAX_INVOKE_INPUTS, DEFAULT_MAX_OUTBOUND_QUEUE, DEFAULT_MAX_PENDING_ASYNC,
    EDGE_INGRESS_BYTE_BUDGET, EDGE_MAX_APP_EVENT_BYTES, EDGE_MAX_COMPLETION_RESULT_BYTES,
    EDGE_MAX_INVOKE_BYTES, EDGE_MAX_INVOKE_INPUTS,
};


#[cfg(test)]
#[path = "snapshot_fidelity_tests.rs"]
mod snapshot_fidelity_tests;

#[cfg(test)]
#[path = "shared_model_tests.rs"]
mod shared_model_tests;
