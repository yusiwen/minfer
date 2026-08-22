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

    pub fn store(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let dim = self.dim;
        let offset = pos * dim;
        self.k[offset..offset + dim].copy_from_slice(k);
        self.v[offset..offset + dim].copy_from_slice(v);
        if pos + 1 > self.size {
            self.size = pos + 1;
        }
    }

    pub fn get_k(&self) -> &[f32] {
        &self.k[..self.size * self.dim]
    }

    pub fn get_v(&self) -> &[f32] {
        &self.v[..self.size * self.dim]
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }

    /// Store K/V for multiple positions at once.
    pub fn store_multi(&mut self, positions: &[usize], k_rope: &[f32], v: &[f32]) {
        let dim = self.dim;
        for (i, &pos) in positions.iter().enumerate() {
            let offset = pos * dim;
            self.k[offset..offset + dim].copy_from_slice(&k_rope[i * dim..(i + 1) * dim]);
            self.v[offset..offset + dim].copy_from_slice(&v[i * dim..(i + 1) * dim]);
            if pos + 1 > self.size {
                self.size = pos + 1;
            }
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

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }
}
