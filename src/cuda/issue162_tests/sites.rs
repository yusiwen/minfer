//! The union driver: every audited `<<<` site is armed, reached, named by its own report and the fixture fragment, and no latch survives.
//!
//! Split out of `src/cuda/issue162_tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// Issue #162 acceptance, site by site: arming a site drives a REAL failing
/// launch, and the site names itself and its instantiation and clears the
/// latch. Every `<<<` site in the audit fixture must be reached.
#[test]
fn cuda_issue162_every_launch_site_names_itself_and_leaves_no_latch() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    let s = device().unwrap();
    let rows = fixture();
    let frag: HashMap<String, String> = rows.iter().map(|r| (r.2.clone(), r.3.clone())).collect();
    let ctx = Ctx::new(s);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let st = ctx.stream;

    macro_rules! go {
        ($arm:expr, $body:expr) => {
            run(s, &frag, $arm, &mut seen, $body)
        };
    }

    // ── the split-attention families ────────────────────────────────
    for (mode, tag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
        let arm: Vec<&str> = vec![
            match tag {
                "map" => "launch:gqa_attn_split_batched_kv__partial_map",
                "span" => "launch:gqa_attn_split_batched_kv__partial_span",
                _ => "launch:gqa_attn_split_batched_kv__partial_causal",
            },
            "launch:gqa_attn_split_batched_kv__combine",
        ];
        go!(&arm, || unsafe {
            launch_gqa_attn_split_batched_f16kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.f(4),
                ctx.ci32(5),
                mode,
                4,
                2,
                64,
                0.125,
                68,
                4,
                st,
            );
        });
    }

    // ── embedding gather (8 quantized cases) ────────────────────────
    for (type_id, token) in [
        (0, "launch:embed_rows__q8_0"),
        (1, "launch:embed_rows__q4_0"),
        (2, "launch:embed_rows__q4_k"),
        (7, "launch:embed_rows__q4_1"),
        (4, "launch:embed_rows__q5_1"),
        (5, "launch:embed_rows__q5_k"),
        (6, "launch:embed_rows__q5_0"),
        (3, "launch:embed_rows__q6_k"),
    ] {
        go!(&[token], || unsafe {
            launch_embed_rows(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, type_id, 210, st);
        });
    }
    go!(&["launch:embed_rows_f16"], || unsafe {
        launch_embed_rows_f16(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, st);
    });

    // ── f32 / f16 matmul shape branches ─────────────────────────────
    for (id, token) in [
        (8, "launch:f32_f32_matmul__vec"),
        (7, "launch:f32_f32_matmul__scalar"),
    ] {
        go!(&[token], || unsafe {
            launch_f32_f32_matmul(ctx.cf(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
        });
    }
    for (id, token) in [
        (8, "launch:f16_f32_matmul_vec"),
        (7, "launch:f16_f32_matmul_scalar"),
    ] {
        go!(&[token], || unsafe {
            launch_f16_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
        });
    }

    // ── every single-site launcher ──────────────────────────────────
    go!(&["launch:q4_0_q8_0_matmul"], || unsafe {
        launch_q4_0_q8_0_matmul(ctx.u(0), ctx.u(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_0_f32_matmul"], || unsafe {
        launch_q4_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q8_0_f32_matmul"], || unsafe {
        launch_q8_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_1_f32_matmul"], || unsafe {
        launch_q4_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_k_f32_matmul"], || unsafe {
        launch_q4_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_1_f32_matmul"], || unsafe {
        launch_q5_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_0_f32_matmul"], || unsafe {
        launch_q5_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_k_f32_matmul"], || unsafe {
        launch_q5_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q6_k_f32_matmul"], || unsafe {
        launch_q6_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q6_k_f32_matmul_padded"], || unsafe {
        launch_q6_k_f32_matmul_padded(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:swiglu_f32_off"], || unsafe {
        launch_swiglu_f32_off(ctx.f(0), 16, 0, st);
    });
    go!(&["launch:swiglu_quant_pad40"], || unsafe {
        launch_swiglu_quant_pad40(ctx.f(0), ctx.mu(1), 16, 0, st);
    });
    go!(&["launch:gather_rows_f32"], || unsafe {
        launch_gather_rows_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 2, st);
    });
    go!(&["launch:quantize_q8_0_pad40"], || unsafe {
        launch_quantize_q8_0_pad40(ctx.cf(0), ctx.mu(1), 256, 2, st);
    });
    go!(&["launch:quantize_q8_0_pad40_t"], || unsafe {
        launch_quantize_q8_0_pad40_t(ctx.cf(0), ctx.mu(1), ctx.mu(2), 256, 2, 8, 1, st);
    });
    go!(&["launch:quantize_q8_0"], || unsafe {
        launch_quantize_q8_0(ctx.cf(0), ctx.mu(1), 256, 2, st);
    });
    go!(&["launch:rms_norm_f32"], || unsafe {
        launch_rms_norm_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 1e-6, 2, st);
    });
    go!(&["launch:rms_norm_quant_pad40"], || unsafe {
        launch_rms_norm_quant_pad40(ctx.cf(0), ctx.cf(1), ctx.f(2), ctx.mu(3), 64, 1e-6, 8, st);
    });
    go!(&["launch:add_bias_f32"], || unsafe {
        launch_add_bias_f32(ctx.f(0), ctx.cf(1), 64, 8, st);
    });
    go!(&["launch:add_f32"], || unsafe {
        launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:mul_f32"], || unsafe {
        launch_mul_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:silu_f32"], || unsafe {
        launch_silu_f32(ctx.f(0), 64, st);
    });
    go!(&["launch:swiglu_f32"], || unsafe {
        launch_swiglu_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:rms_norm_quant_f32_t"], || unsafe {
        launch_rms_norm_quant_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.f(2),
            ctx.mu(3),
            ctx.mu(4),
            256,
            1e-6,
            8,
            8,
            1,
            st,
        );
    });
    go!(&["launch:rms_norm_quant_nw_f32_t"], || unsafe {
        launch_rms_norm_quant_nw_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.mu(2),
            ctx.mu(3),
            256,
            1e-6,
            8,
            8,
            1,
            st,
        );
    });
    go!(&["launch:swiglu_quant_f32_t"], || unsafe {
        launch_swiglu_quant_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.f(2),
            ctx.mu(3),
            ctx.mu(4),
            256,
            2,
            8,
            1,
            st,
        );
    });
    go!(&["launch:swiglu_quant_nw_f32_t"], || unsafe {
        launch_swiglu_quant_nw_f32_t(ctx.cf(0), ctx.cf(1), ctx.mu(2), ctx.mu(3), 256, 2, 8, 1, st);
    });
    go!(&["launch:f32_bits_to_i32"], || unsafe {
        launch_f32_bits_to_i32(ctx.cf(0), ctx.i32(1), 64, st);
    });
    go!(&["launch:rope_f32"], || unsafe {
        launch_rope_f32(ctx.f(0), 4, 64, 2, 10000.0, 1.0, ctx.ci32(1), st);
    });
    go!(&["launch:store_kv_f32"], || unsafe {
        launch_store_kv_f32(ctx.cf(0), ctx.f(1), 64, 2, ctx.ci32(2), st);
    });
    go!(&["launch:store_kv_f16"], || unsafe {
        launch_store_kv_f16(ctx.cf(0), ctx.p(1), 64, 2, ctx.ci32(2), st);
    });
    go!(&["launch:store_kv_q8_0"], || unsafe {
        launch_store_kv_q8_0(ctx.cf(0), ctx.p(1), 64, 2, 68, ctx.ci32(2), st);
    });
    go!(&["launch:attn_bias_rope_store"], || unsafe {
        launch_attn_bias_rope_store(
            ctx.f(0),
            ctx.f(1),
            ctx.f(2),
            ctx.p(3),
            ctx.p(4),
            ctx.p(5),
            ctx.p(6),
            ctx.p(7),
            2,
            4,
            64,
            10000.0,
            1.0,
            ctx.ci32(8),
            ctx.ci32(9),
            0,
            st,
        );
    });
    // #144 item 1: the packed arm of the same epilogue.
    go!(&["launch:attn_bias_rope_store_q8_0"], || unsafe {
        launch_attn_bias_rope_store_q8_0(
            ctx.f(0),
            ctx.cf(1),
            ctx.cf(2),
            ctx.p(3),
            ctx.p(4),
            ctx.p(5),
            ctx.p(6),
            ctx.p(7),
            2,
            4,
            64,
            10000.0,
            1.0,
            ctx.ci32(8),
            ctx.ci32(9),
            68,
            st,
        );
    });
    for (mode, token) in [
        (MAP, "launch:gqa_attn_f32_f16kv__map"),
        (SPAN, "launch:gqa_attn_f32_f16kv__span"),
        (CAUSAL, "launch:gqa_attn_f32_f16kv__causal"),
    ] {
        go!(&[token], || unsafe {
            launch_gqa_attn_f32_f16kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                mode,
                4,
                2,
                64,
                0.125,
                2,
                st,
            );
        });
    }
    for (mode, token) in [
        (MAP, "launch:gqa_attn_split_f32kv__map"),
        (SPAN, "launch:gqa_attn_split_f32kv__span"),
        (CAUSAL, "launch:gqa_attn_split_f32kv__causal"),
    ] {
        go!(
            &[token, "launch:gqa_attn_split_f32kv__combine"],
            || unsafe {
                launch_gqa_attn_split_f32kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    st,
                );
            }
        );
    }
    for (mode, base_token, dp4a_token, wide_token) in [
        (
            MAP,
            "launch:gqa_attn_split_q8_0__map",
            "launch:gqa_attn_split_q8_0__map_dp4a",
            "launch:gqa_attn_split_q8_0__map_wide",
        ),
        (
            SPAN,
            "launch:gqa_attn_split_q8_0__span",
            "launch:gqa_attn_split_q8_0__span_dp4a",
            "launch:gqa_attn_split_q8_0__span_wide",
        ),
        (
            CAUSAL,
            "launch:gqa_attn_split_q8_0__causal",
            "launch:gqa_attn_split_q8_0__causal_dp4a",
            "launch:gqa_attn_split_q8_0__causal_wide",
        ),
    ] {
        // #186/#202: the Q8_0 decode launcher picks one of three instantiations
        // from the `dp4a`/`wide` arguments. Every audited site must be driven,
        // so all three arms are driven explicitly — the env resolves to one of
        // them for production, but `run`'s armed set is a value here, not a
        // process-global answer.
        for (token, dp4a, wide) in [
            (base_token, 0i32, 0i32),
            (dp4a_token, 1, 0),
            (wide_token, 1, 1),
        ] {
            go!(&[token, "launch:gqa_attn_split_q8_0__combine"], || unsafe {
                launch_gqa_attn_split_q8_0(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    68,
                    dp4a,
                    wide,
                    st,
                );
            });
        }
    }
    // hd == 128 takes the dual-kernel (h4w + hybrid) path; hd != 128 the
    // single incumbent one. Both share `__combine`.
    for (mode, mtag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
        let h4w = format!("launch:gqa_attn_split_f16kv__{mtag}_h4w");
        let hyb = format!("launch:gqa_attn_split_f16kv__hybrid_{mtag}");
        go!(
            &[
                h4w.as_str(),
                hyb.as_str(),
                "launch:gqa_attn_split_f16kv__combine"
            ],
            || unsafe {
                launch_gqa_attn_split_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    128,
                    0.125,
                    68,
                    st,
                );
            }
        );
        let plain = format!("launch:gqa_attn_split_f16kv__{mtag}");
        go!(
            &[plain.as_str(), "launch:gqa_attn_split_f16kv__combine"],
            || unsafe {
                launch_gqa_attn_split_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    st,
                );
            }
        );
    }
    // The layout-tagged general kernel: one source site, nine instantiations
    // (the message names the instantiation). Drive one.
    go!(&["launch:gqa_attn_f32"], || unsafe {
        launch_gqa_attn_f32(
            ctx.cf(0),
            ctx.p(1),
            ctx.p(2),
            ctx.f(3),
            ctx.ci32(4),
            CAUSAL,
            KV_LAYOUT_F32,
            4,
            2,
            64,
            0.125,
            256,
            2,
            st,
        );
    });
    // #144: the FA launcher serves both staged layouts from one source site
    // per (layout, mode) — six sites, all driven here.
    for (mode, layout, token) in [
        (
            MAP,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_map",
        ),
        (
            SPAN,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_span",
        ),
        (
            CAUSAL,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_causal",
        ),
        (
            MAP,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_map",
        ),
        (
            SPAN,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_span",
        ),
        (
            CAUSAL,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_causal",
        ),
    ] {
        go!(&[token], || unsafe {
            launch_fa_prefill_kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                mode,
                4,
                2,
                128,
                0.125,
                2,
                layout,
                136,
                st,
            );
        });
    }
    for (type_id, token) in [
        (0, "launch:dequant_f16__q8_0"),
        (1, "launch:dequant_f16__q4_0"),
        (2, "launch:dequant_f16__q4_1"),
        (3, "launch:dequant_f16__q5_0"),
        (4, "launch:dequant_f16__q5_1"),
        (5, "launch:dequant_f16__q4_k"),
        (6, "launch:dequant_f16__q5_k"),
        (7, "launch:dequant_f16__q6_k"),
    ] {
        go!(&[token], || unsafe {
            launch_dequant_f16(type_id, ctx.u(0), ctx.p(1), 8, 64, 210, st);
        });
    }
    go!(&["launch:convert_f16"], || unsafe {
        launch_convert_f16(ctx.cf(0), ctx.p(1), 256, st);
    });
    go!(&["launch:gemm_qb_nt"], || unsafe {
        launch_gemm_qb_nt(ctx.p(0), ctx.u(1), ctx.f(2), 2, 8, 64, 5, 212, st);
    });
    // `launch_gemm_f16`'s site token is chosen by `af32`; the audit resolves
    // the ternary to the first arm, so drive the `af32 = true` one.
    go!(&["launch:gemm_f16_a32"], || unsafe {
        launch_gemm_f16(ctx.p(0), ctx.p(1), ctx.f(2), 2, 8, 64, st, true);
    });
    // The MMQ fast paths: arm the launch token (not the attribute token), so
    // the opt-in succeeds and the launch itself is what fails.
    go!(&["launch:mmq_raw_nb"], || unsafe {
        launch_mmq_raw_nb_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 1, 8, 64, st, 8);
    });
    for (kd, token) in [
        (4, "launch:mmq_raw_wide_kd4"),
        (8, "launch:mmq_raw_wide_kd8"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_wide_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
        });
    }
    for (kd, token) in [(4, "launch:mmq_raw_nt_kd4"), (8, "launch:mmq_raw_nt_kd8")] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
        });
    }
    go!(&["launch:mmq_nt"], || unsafe {
        launch_mmq_nt(0, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, 40, st);
    });
    // The two NB-BT launchers: `w_dsc`/`w_exp` selects the instantiation, and
    // the k-split reduce is its own token (the kernel's token returns before
    // it, so the reduce can only be reached alone).
    for (dsc, token) in [
        (false, "launch:mmq_raw_nb_bt"),
        (true, "launch:mmq_raw_nb_bt"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nb_bt_nt(
                5,
                ctx.u(0),
                if dsc { ctx.u(1) } else { std::ptr::null() },
                ctx.u(2),
                ctx.u(3),
                ctx.f(4),
                1,
                8,
                64,
                8,
                st,
                8,
                ctx.f(5),
                1,
            );
        });
    }
    go!(&["launch:mmq_raw_nb_bt_ksplit"], || unsafe {
        launch_mmq_raw_nb_bt_nt(
            5,
            ctx.u(0),
            ctx.u(1),
            ctx.u(2),
            ctx.u(3),
            ctx.f(4),
            1,
            8,
            64,
            16,
            st,
            8,
            ctx.f(5),
            2,
        );
    });
    for (exp, token) in [
        (false, "launch:mmq_raw_nb_bt_q6k"),
        (true, "launch:mmq_raw_nb_bt_q6k"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nb_bt_q6k_nt(
                7,
                ctx.u(0),
                if exp { ctx.u(1) } else { std::ptr::null() },
                ctx.u(2),
                ctx.u(3),
                ctx.u(4),
                ctx.f(5),
                1,
                8,
                64,
                8,
                212,
                st,
                8,
                ctx.f(6),
                1,
            );
        });
    }
    go!(&["launch:mmq_raw_nb_bt_q6k_ksplit"], || unsafe {
        launch_mmq_raw_nb_bt_q6k_nt(
            7,
            ctx.u(0),
            ctx.u(1),
            ctx.u(2),
            ctx.u(3),
            ctx.u(4),
            ctx.f(5),
            1,
            8,
            64,
            16,
            212,
            st,
            8,
            ctx.f(6),
            2,
        );
    });
    // ── the multi-token MMVQ family ─────────────────────────────────
    for (token, extra) in [
        ("launch:q4_k_q8_mmvq", 0),
        ("launch:q4_k_q8_mmvq_v2", 0),
        ("launch:q4_k_q8_mmvq_multi", 0),
        ("launch:q4_k_q8_mmvq_v2_multi", 0),
        ("launch:q5_k_q8_mmvq", 0),
        ("launch:q5_k_q8_mmvq_v2", 0),
        ("launch:q5_k_q8_mmvq_multi", 0),
        ("launch:q5_k_q8_mmvq_v2_multi", 0),
        ("launch:q4_0_q8_mmvq", 0),
        ("launch:q4_0_q8_mmvq_multi", 0),
        ("launch:q8_0_q8_mmvq", 0),
        ("launch:q8_0_q8_mmvq_multi", 0),
        ("launch:q6_k_q8_mmvq", 1),
        ("launch:q6_k_q8_mmvq_v2", 1),
        ("launch:q6_k_q8_mmvq_v2_pf", 1),
        ("launch:q6_k_q8_mmvq_multi", 1),
        ("launch:q6_k_q8_mmvq_v2_multi", 1),
        ("launch:q6_k_q8_mmvq_v2_dpl", 2),
        ("launch:q6_k_q8_mmvq_v2_pf_dpl", 2),
    ] {
        go!(&[token], || unsafe {
            let (w, a, o) = (ctx.u(0), ctx.u(1), ctx.f(2));
            match token {
                "launch:q4_k_q8_mmvq" => launch_q4_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_v2" => launch_q4_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_multi" => launch_q4_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_v2_multi" => {
                    launch_q4_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                }
                "launch:q5_k_q8_mmvq" => launch_q5_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_v2" => launch_q5_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_multi" => launch_q5_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_v2_multi" => {
                    launch_q5_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                }
                "launch:q4_0_q8_mmvq" => launch_q4_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q4_0_q8_mmvq_multi" => launch_q4_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q8_0_q8_mmvq" => launch_q8_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q8_0_q8_mmvq_multi" => launch_q8_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q6_k_q8_mmvq" => launch_q6_k_q8_mmvq(w, a, o, 8, 64, 2, 210, st),
                "launch:q6_k_q8_mmvq_v2" => launch_q6_k_q8_mmvq_v2(w, a, o, 8, 64, 2, 210, st),
                "launch:q6_k_q8_mmvq_v2_pf" => {
                    launch_q6_k_q8_mmvq_v2_pf(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_multi" => {
                    launch_q6_k_q8_mmvq_multi(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_v2_multi" => {
                    launch_q6_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_v2_dpl" => {
                    launch_q6_k_q8_mmvq_v2_dpl(w, a, o, 8, 64, 2, 1, st)
                }
                _ => launch_q6_k_q8_mmvq_v2_pf_dpl(w, a, o, 8, 64, 2, 1, st),
            }
            let _ = extra;
        });
    }
    go!(&["launch:q8_0_p32_q8_mmvq"], || unsafe {
        launch_q8_0_p32_q8_mmvq(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
    });
    go!(&["launch:q8_0_p32_q8_mmvq_multi"], || unsafe {
        launch_q8_0_p32_q8_mmvq_multi(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
    });
    go!(&["launch:kv_move_rows"], || unsafe {
        launch_kv_move_rows(ctx.f(0), ctx.cf(1), 1, 0, 1, 64, st);
    });

    // ── the union assertion (rule 1: the expected set is a value, not a
    //    relation between two code paths) ─────────────────────────────
    let expected: BTreeSet<String> = rows.iter().map(|r| r.2.clone()).collect();
    let missing: Vec<&String> = expected.difference(&seen).collect();
    let extra: Vec<&String> = seen.difference(&expected).collect();
    assert!(
        missing.is_empty(),
        "the driver never reached {} audited <<< site(s): {missing:?}",
        missing.len()
    );
    assert!(
        extra.is_empty(),
        "the driver observed {} site(s) absent from the audit fixture: {extra:?}",
        extra.len()
    );
    assert_eq!(
        s.take_last_error(),
        0,
        "no latch may survive the gate: a latched error reaching CudaState::sync is the bug"
    );
}
