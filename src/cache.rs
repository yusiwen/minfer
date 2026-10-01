// KV Cache — shared by all transformer architectures.
//
// LEGACY: the graph path owns KV in persistent per-layer regions (alloc.rs),
// so this type survives only as the `ModelDef::forward` signature's vestigial
// `&mut KVCache` parameter (main.rs / tests still construct it, nothing reads
// it). Kept until the trait signature is refactored away.
#![allow(dead_code)]

/// KV cache for a single transformer layer.
#[derive(Clone)]
pub struct KVCacheLayer {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub size: usize,
    pub max_size: usize,
    pub dim: usize,
}

impl KVCacheLayer {
    pub fn new(max_size: usize, dim: usize) -> Self {
        Self {
            k: vec![0.0f32; max_size * dim],
            v: vec![0.0f32; max_size * dim],
            size: 0,
            max_size,
            dim,
        }
    }
}

/// KV cache for all layers.
#[derive(Clone)]
pub struct KVCache {
    pub layers: Vec<KVCacheLayer>,
}

impl KVCache {
    pub fn new(n_layers: usize, dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| KVCacheLayer::new(max_seq_len, dim))
            .collect();
        Self { layers }
    }
}
