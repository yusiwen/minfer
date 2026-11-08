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
    assert_eq!(c.backends, crate::graph::registry::BackendFilter::all());
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

#[test]
fn device_policy_knobs_parse_like_the_environment_they_replace() {
    // #769 moved the device backends' knobs onto `EngineConfig`. The parse is the
    // half a CPU box can check: on-unless-`0` gates, the inverted `NO_*` flags, the
    // `MMQ_A_FUSE` mode table (unset = 2, anything unrecognised = 0), and the two
    // filtered values (`MV_MMA_NT` only accepts 1/2/4, an empty `METALLIB_FILE` is
    // unset).
    let names = [
        "MINFER_MMQ",
        "MINFER_MMQ_RAW",
        "MINFER_MMQ_RAW_NB",
        "MINFER_MMQ_A_TRANSPOSE",
        "MINFER_MMQ_Q6K_NB",
        "MINFER_Q6K_DPL",
        "MINFER_MMQ_A_FUSE",
        "MINFER_Q6K_PF",
        "MINFER_NO_FLASH",
        "MINFER_NO_RMS_256",
        "MINFER_MV_MMA",
        "MINFER_MV_MMA_NSG",
        "MINFER_MV_MMA_NT",
        "MINFER_METALLIB_FILE",
        "MINFER_WEIGHT_COPY",
        "MINFER_DISABLE_MPS",
        "MINFER_DISABLE_CUDA",
        "MINFER_MMQ_RAW_KD",
        "MINFER_MMQ_KSPLIT_TARGET",
    ];
    let saved: Vec<_> = names.iter().map(|n| (*n, std::env::var(n).ok())).collect();
    for (n, _) in &saved {
        std::env::remove_var(n);
    }

    // Every default: MMQ family on, diagnostics/opt-ins off, the A-fuse mode 2.
    let d = EngineConfig::from_env();
    assert!(
        d.mmq
            && d.mmq_raw
            && d.mmq_raw_nb
            && d.mmq_a_transpose
            && d.mmq_q6k_nb
            && d.q6k_dpl
            && d.q6k_prefetch
            && d.flash_attn
            && d.rms_256
    );
    assert_eq!(d.mmq_a_fuse, 2);
    assert_eq!(d.mmq_raw_kd, 8);
    assert_eq!(d.mmq_ksplit_target, None);
    assert!(!d.mv_mma && d.mv_mma_nsg.is_none() && d.mv_mma_nt.is_none());
    assert!(!d.weight_copy && !d.disable_mps && !d.disable_cuda);
    assert!(d.metallib_file.is_none());

    // `0` opts out of each on-unless-0 gate; a non-`0` value leaves it on.
    std::env::set_var("MINFER_MMQ", "0");
    std::env::set_var("MINFER_MMQ_RAW", "0");
    std::env::set_var("MINFER_MMQ_RAW_NB", "0");
    std::env::set_var("MINFER_MMQ_A_TRANSPOSE", "0");
    std::env::set_var("MINFER_MMQ_Q6K_NB", "0");
    std::env::set_var("MINFER_Q6K_DPL", "0");
    std::env::set_var("MINFER_Q6K_PF", "0");
    std::env::set_var("MINFER_MMQ_A_FUSE", "1");
    std::env::set_var("MINFER_MMQ_RAW_KD", "12");
    std::env::set_var("MINFER_MMQ_KSPLIT_TARGET", "512");
    let c = EngineConfig::from_env();
    assert!(
        !c.mmq
            && !c.mmq_raw
            && !c.mmq_raw_nb
            && !c.mmq_a_transpose
            && !c.mmq_q6k_nb
            && !c.q6k_dpl
            && !c.q6k_prefetch,
        "each on-unless-0 gate opts out on \"0\""
    );
    assert_eq!(c.mmq_a_fuse, 1);
    assert_eq!(c.mmq_raw_kd, 12);
    assert_eq!(c.mmq_ksplit_target, Some(512));

    // The inverted flags: the `NO_*` spelling turns the optimisation off.
    std::env::set_var("MINFER_NO_FLASH", "1");
    std::env::set_var("MINFER_NO_RMS_256", "1");
    std::env::set_var("MINFER_MV_MMA", "1");
    std::env::set_var("MINFER_MV_MMA_NSG", "2");
    std::env::set_var("MINFER_MV_MMA_NT", "3"); // not 1/2/4 → filtered out
    std::env::set_var("MINFER_METALLIB_FILE", "");
    std::env::set_var("MINFER_WEIGHT_COPY", "1");
    std::env::set_var("MINFER_DISABLE_MPS", "x");
    std::env::set_var("MINFER_DISABLE_CUDA", "x");
    let c = EngineConfig::from_env();
    assert!(!c.flash_attn && !c.rms_256);
    assert!(c.mv_mma && c.mv_mma_nsg == Some(2) && c.mv_mma_nt.is_none());
    assert!(c.metallib_file.is_none(), "an empty override is unset");
    assert!(c.weight_copy && c.disable_mps && c.disable_cuda);

    // The mode table: "2" = 2, and an unrecognised value degrades to 0 (off).
    std::env::set_var("MINFER_MMQ_A_FUSE", "2");
    assert_eq!(EngineConfig::from_env().mmq_a_fuse, 2);
    std::env::set_var("MINFER_MMQ_A_FUSE", "3");
    assert_eq!(EngineConfig::from_env().mmq_a_fuse, 0);
    std::env::set_var("MINFER_MMQ_A_FUSE", "0");
    assert_eq!(EngineConfig::from_env().mmq_a_fuse, 0);

    for (name, value) in saved {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}
