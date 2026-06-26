//! Global lattice of registered `TypeNode`s.
//!
//! Built once at process startup from `inventory::iter::<TypeNodeReg>`
//! and cached for the process lifetime. Subtype queries
//! (`is_subtype_of`) walk the parent chain with memoization. The
//! lattice is never mutated after construction; no locks, no
//! allocation on the query path.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use super::TypeNode;

/// Process-wide lattice. Single instance via [`Lattice::get`].
pub struct Lattice {
    /// `id → TypeNode`. Constructed from inventory submissions.
    by_id: &'static HashMap<&'static str, &'static TypeNode>,
    /// Subtype-query memo: `(child_ptr, parent_ptr) → bool`.
    /// Lazily populated as queries arrive. RwLock is taken
    /// briefly on cold-cache writes; reads use a snapshot.
    cache: RwLock<HashMap<(usize, usize), bool>>,
}

impl Lattice {
    /// Get the process-wide lattice. Builds on first access.
    pub fn get() -> &'static Lattice {
        static LATTICE: OnceLock<Lattice> = OnceLock::new();
        LATTICE.get_or_init(|| Lattice {
            by_id: super::id_to_node_map(),
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// `true` iff `child` is `parent` or a (transitive) descendant
    /// of `parent` in the lattice. Walks the parent chain on the
    /// first call for a given pair; subsequent calls hit the cache.
    pub fn is_subtype(&self, child: &'static TypeNode, parent: &'static TypeNode) -> bool {
        if std::ptr::eq(child, parent) {
            return true;
        }
        let key = (child as *const _ as usize, parent as *const _ as usize);
        if let Some(&hit) = self.cache.read().expect("lattice cache poisoned").get(&key) {
            return hit;
        }

        // Walk child's parent chain looking for `parent`.
        let mut cursor = child.parent;
        let mut found = false;
        while let Some(parent_id) = cursor {
            let Some(node) = self.by_id.get(parent_id).copied() else {
                break;
            };
            if std::ptr::eq(node, parent) {
                found = true;
                break;
            }
            cursor = node.parent;
        }

        self.cache
            .write()
            .expect("lattice cache poisoned")
            .insert(key, found);
        found
    }

    /// Resolve a registered `TypeNode` by its `id` string. `None`
    /// when no inventory submission named that id.
    pub fn lookup(&self, id: &str) -> Option<&'static TypeNode> {
        self.by_id.get(id).copied()
    }

    /// Iterate every registered `TypeNode`. Useful for diagnostics
    /// and `cargo doc`-style introspection of the type tree.
    pub fn nodes(&self) -> impl Iterator<Item = &'static TypeNode> + '_ {
        self.by_id.values().copied()
    }
}
