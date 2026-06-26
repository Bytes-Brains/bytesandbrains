//! `EventSource` - registered `event_kind → ComponentTag`
//! subscription table consulted by the engine's Phase 3 bus event
//! routing per ENGINE.md §10.7.
//!
//! Syscall ops with `event_subscriptions()` declarations register
//! themselves here at install time; the engine's `poll()` Phase 3
//! consults `subscribers(event_kind)` to deliver each drained
//! `NodeEvent` to the matching consumer ops.

use std::collections::HashMap;

use crate::ids::ComponentTag;

/// `event_kind → Vec<ComponentTag>` subscription table.
#[derive(Default)]
pub struct EventSource {
    subscriptions: HashMap<String, Vec<ComponentTag>>,
}

impl EventSource {
    /// Construct a fresh, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe `tag` to events of `event_kind`. Idempotent -
    /// re-subscribing the same tag is a no-op.
    pub fn subscribe(&mut self, event_kind: &str, tag: ComponentTag) {
        let entry = self
            .subscriptions
            .entry(event_kind.to_string())
            .or_default();
        if !entry.contains(&tag) {
            entry.push(tag);
        }
    }

    /// Remove `tag`'s subscription to `event_kind` if present.
    pub fn unsubscribe(&mut self, event_kind: &str, tag: ComponentTag) {
        if let Some(entry) = self.subscriptions.get_mut(event_kind) {
            entry.retain(|t| *t != tag);
            if entry.is_empty() {
                self.subscriptions.remove(event_kind);
            }
        }
    }

    /// Subscribers to `event_kind`. Returns an empty slice when no
    /// component is subscribed.
    pub fn subscribers(&self, event_kind: &str) -> &[ComponentTag] {
        self.subscriptions
            .get(event_kind)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of distinct event kinds with at least one subscriber.
    pub fn len(&self) -> usize {
        self.subscriptions.len()
    }

    /// `true` when no subscriptions are registered.
    pub fn is_empty(&self) -> bool {
        self.subscriptions.is_empty()
    }
}

