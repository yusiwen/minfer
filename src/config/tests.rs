//! `#[cfg(test)] mod tests` for `src/config.rs`.

use super::*;

#[test]
fn defaults_keep_every_optimisation_on() {
    let c = EngineConfig::default();
    assert!(c.fuse_qkv && c.fuse_ffn && c.fused_q8_kv && c.split_attn);
    assert!(c.kv_defrag && c.cuda_graph && c.prefill_capture && c.mmq_q4k_dsc);
    assert!(c.avx2 && c.avx512 && c.neon && c.f16_rowb && c.context_shift);
    assert!(!c.packed_kv_stage && !c.metal_sync_ops && !c.sync_copies && !c.op_profile);
    assert!(c.kv_cache_type.is_none() && c.gpu_layers.is_none() && c.gpu_mem.is_none());
}

#[test]
fn from_env_maps_the_documented_spellings() {
    // The environment is process-wide, so this test sets and restores exactly
    // the variables it owns and does not rely on a clean slate.
    let names = [
        "MINFER_NO_FUSE_QKV",
        "MINFER_NO_FUSE_FFN",
        "MINFER_FFN_COMPOSITION",
        "MINFER_CACHE_TYPE",
        "MINFER_GPU_LAYERS",
        "MINFER_GPU_MEM",
        "MINFER_SYNC_COPIES",
        "MINFER_GRAPH_TRACE",
        "MINFER_GRAPH_DUMP",
        "MINFER_ATTN_CHUNKS",
        "MINFER_TEST_CAPTURE_WARMUP",
        "MINFER_MMQ_Q4K_DSC",
        "MINFER_NO_CONTEXT_SHIFT",
        "MINFER_NO_NEON",
    ];
    let saved: Vec<_> = names.iter().map(|n| (*n, std::env::var(n).ok())).collect();
    for (n, _) in &saved {
        std::env::remove_var(n);
    }
    // `set_var` is unsafe in edition 2024; this crate is 2021 and the batch
    // runner is serial enough for a pure mapping test that restores its keys.
    std::env::set_var("MINFER_NO_FUSE_QKV", "1");
    std::env::set_var("MINFER_FFN_COMPOSITION", "1");
    std::env::set_var("MINFER_CACHE_TYPE", "f16");
    std::env::set_var("MINFER_GPU_LAYERS", "4");
    std::env::set_var("MINFER_GPU_MEM", "512");
    std::env::set_var("MINFER_SYNC_COPIES", "1");
    std::env::set_var("MINFER_GRAPH_TRACE", "x");
    std::env::set_var("MINFER_GRAPH_DUMP", "/tmp/d");
    std::env::set_var("MINFER_ATTN_CHUNKS", "8");
    std::env::set_var("MINFER_TEST_CAPTURE_WARMUP", "1");
    std::env::set_var("MINFER_MMQ_Q4K_DSC", "0");
    std::env::set_var("MINFER_NO_CONTEXT_SHIFT", "1");
    std::env::set_var("MINFER_NO_NEON", "1");

    let c = EngineConfig::from_env();
    assert!(!c.fuse_qkv, "NO_FUSE_QKV=1 disables the fusion");
    assert!(c.fuse_ffn, "an unset NO_* keeps the optimisation on");
    assert_eq!(c.ffn_composition.as_deref(), Some("1"));
    assert_eq!(c.kv_cache_type.as_deref(), Some("f16"));
    assert_eq!(c.gpu_layers.as_deref(), Some("4"));
    assert_eq!(c.gpu_mem.as_deref(), Some("512"));
    assert!(c.sync_copies && c.graph_trace && c.ffndebug == false);
    assert_eq!(c.graph_dump.as_deref(), Some("/tmp/d"));
    assert_eq!(c.attn_chunks, Some(8));
    assert_eq!(c.capture_warmup, Some(1));
    assert!(!c.mmq_q4k_dsc, "MMQ_Q4K_DSC=0 disables it");
    assert!(!c.context_shift && !c.neon);

    for (name, value) in saved {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}
