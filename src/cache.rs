// KV Cache — shared by all transformer architectures.
//
// LEGACY: the graph path owns KV in persistent per-layer regions (alloc.rs),
// so this type survives only as the `ModelDef::forward` signature's vestigial
// `&mut KVCache` parameter (main.rs / tests still construct it, nothing reads
// it). Kept until the trait signature is refactored away.
//
// The storage was deleted in [#244] rather than only annotated: `KVCacheLayer`
// allocated `max_seq_len * dim` f32 words per layer and direction on **every
// load** (`KVCache::new` in main.rs), and rustc proved every one of those
// fields dead in every configuration — the graph path never consults them. What
// remains is an empty handle, so the `ModelDef::forward` signature does not
// move; removing *that* parameter (and this module with it) is the trait change
// [#244] reports, not a dead-code cleanup.
//
// [#244]: https://github.com/yusiwen/minfer/issues/244

/// The legacy KV-cache handle.
///
/// Empty on purpose: [`crate::models::ModelDef::forward`] takes `&mut KVCache`
/// and ignores it, and nothing read the rows this type used to own.
#[derive(Clone)]
pub struct KVCache;

impl KVCache {
    /// Kept with its arity so the load sites and tests do not move; the
    /// arguments describe a cache this type no longer stores.
    pub fn new(_n_layers: usize, _dim: usize, _max_seq_len: usize) -> Self {
        Self
    }
}
