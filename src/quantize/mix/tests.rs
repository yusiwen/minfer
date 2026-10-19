//! The mixture resolver, pinned against llama.cpp's `llama_tensor_get_type_impl`
//! and against the two reference files whose type histograms are recorded in
//! `docs/GGUF-TOOLING.md` §2.2.

use super::*;

fn facts(cat: TensorCategory, layer: Option<usize>, row: i64) -> TensorFacts {
    TensorFacts {
        category: cat,
        layer,
        row,
    }
}

fn tied_ctx(n_layers: usize) -> MixCtx {
    MixCtx {
        n_layers,
        n_attention_wv: n_layers,
        has_tied_embeddings: true,
    }
}

/// llama.cpp's `use_more_bits`, layer by layer, for the two measured models.
///
/// The numbers are not decoration: 12 of 24 (the 0.5B) and 14 of 28 (the 0.6B)
/// are exactly the `q6_K` halves of the measured `q4_K` mixtures.
#[test]
fn use_more_bits_matches_llama_cpps_layer_sets() {
    let want_24: Vec<usize> = vec![0, 1, 2, 5, 8, 11, 14, 17, 20, 21, 22, 23];
    let got_24: Vec<usize> = (0..24).filter(|i| use_more_bits(*i, 24)).collect();
    assert_eq!(got_24, want_24);

    let want_28: Vec<usize> = vec![0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27];
    let got_28: Vec<usize> = (0..28).filter(|i| use_more_bits(*i, 28)).collect();
    assert_eq!(got_28, want_28);

    // The rule's three clauses, at their boundaries: `n/8` is the first layer not
    // in the leading run, `7*n/8` the first in the trailing one, and the modulo
    // resumes after the leading run only.
    assert!(
        use_more_bits(2, 24) && !use_more_bits(3, 24),
        "n/8 boundary"
    );
    assert!(use_more_bits(21, 24), "7n/8 boundary");
    assert!(!use_more_bits(4, 24) && use_more_bits(5, 24), "(i-n/8)%3");
}

/// The category table, including the order-sensitive pairs: a fused or `kv_b`
/// name must not be read as plain `attn_v`, and `attn_output` must not be read as
/// `attn_q`/`attn_v` just because it contains those letters as a prefix.
#[test]
fn categories_come_from_the_gguf_names() {
    let cases = [
        ("output.weight", TensorCategory::Output),
        ("token_embd.weight", TensorCategory::TokenEmbd),
        ("per_layer_token_embd.weight", TensorCategory::TokenEmbd),
        ("blk.0.attn_qkv.weight", TensorCategory::AttnQkv),
        ("blk.0.attn_kv_b.weight", TensorCategory::AttnKvB),
        ("blk.0.attn_v.weight", TensorCategory::AttnV),
        ("blk.0.attn_k.weight", TensorCategory::AttnK),
        ("blk.0.attn_q.weight", TensorCategory::AttnQ),
        ("blk.0.attn_output.weight", TensorCategory::AttnOutput),
        ("blk.0.ffn_up.weight", TensorCategory::FfnUp),
        ("blk.0.ffn_gate.weight", TensorCategory::FfnGate),
        ("blk.0.ffn_down.weight", TensorCategory::FfnDown),
        ("blk.0.attn_norm.weight", TensorCategory::Other),
        ("blk.0.attn_q.bias", TensorCategory::Other),
        ("output_norm.weight", TensorCategory::Other),
        ("blk.3.ffn_down.weight", TensorCategory::FfnDown),
    ];
    for (name, want) in cases {
        assert_eq!(category(name), want, "{name}");
    }
    // `attn_v`/`attn_qkv`/`attn_kv_b` are the categories llama.cpp counts with
    // `i_attention_wv`; `attn_k`/`attn_q` are not.
    assert!(category_is_attn_v(TensorCategory::AttnV));
    assert!(category_is_attn_v(TensorCategory::AttnQkv));
    assert!(category_is_attn_v(TensorCategory::AttnKvB));
    assert!(!category_is_attn_v(TensorCategory::AttnK));
    assert!(!category_is_attn_v(TensorCategory::AttnQ));
    // The block index, and the tensors outside one.
    assert_eq!(layer("blk.23.ffn_down.weight"), Some(23));
    assert_eq!(layer("blk.0.attn_q.weight"), Some(0));
    assert_eq!(layer("token_embd.weight"), None);
    assert_eq!(layer("output.weight"), None);
}

/// `q4_K_M`, every rule at once, read off the reference the gate compares against.
#[test]
fn q4_k_m_selects_llama_cpps_per_tensor_types() {
    let ctx = tied_ctx(24);
    let t = Target::Mixture(Mixture::Q4_K_M);
    let at = |cat, layer: usize| t.select(&facts(cat, Some(layer), 896), &ctx);

    // The OUTPUT branch: the 0.6B's tied embedding (hidden 1024, 256-divisible)
    // is asked for Q6_K, and the 0.5B's (hidden 896) for Q8_0 with no demotion.
    assert_eq!(
        t.select(&facts(TensorCategory::TokenEmbd, None, 1024), &ctx),
        Selection {
            target: QuantTarget::Q6_K,
            reason: Reason::OutputBranch
        }
    );
    assert_eq!(
        t.select(&facts(TensorCategory::TokenEmbd, None, 896), &ctx),
        Selection {
            target: QuantTarget::Q8_0,
            reason: Reason::OutputBranch
        }
    );
    // An untied model's `output.weight` takes the same branch; its
    // `token_embd.weight` stays the base type.
    let untied = MixCtx {
        has_tied_embeddings: false,
        ..ctx
    };
    assert_eq!(
        t.select(&facts(TensorCategory::Output, None, 1024), &untied),
        Selection {
            target: QuantTarget::Q6_K,
            reason: Reason::OutputBranch
        }
    );
    assert_eq!(
        t.select(&facts(TensorCategory::TokenEmbd, None, 1024), &untied)
            .target,
        QuantTarget::Q4_K
    );
    // use_more_bits bumps `attn_v` and `ffn_down` on 12 of the 24 layers, and
    // only there.
    assert_eq!(
        at(TensorCategory::AttnV, 0),
        Selection {
            target: QuantTarget::Q6_K,
            reason: Reason::Mixture
        }
    );
    assert_eq!(at(TensorCategory::AttnV, 3).target, QuantTarget::Q4_K);
    assert_eq!(at(TensorCategory::AttnV, 3).reason, Reason::Requested);
    assert_eq!(at(TensorCategory::FfnDown, 21).target, QuantTarget::Q6_K);
    assert_eq!(at(TensorCategory::FfnDown, 4).target, QuantTarget::Q4_K);
    // The fused-qkv category goes to Q5_K under Q4_K_M; everything else is the
    // base type.
    assert_eq!(
        at(TensorCategory::AttnQkv, 0),
        Selection {
            target: QuantTarget::Q5_K,
            reason: Reason::Mixture
        }
    );
    for cat in [
        TensorCategory::AttnQ,
        TensorCategory::AttnK,
        TensorCategory::AttnOutput,
        TensorCategory::FfnUp,
        TensorCategory::FfnGate,
        TensorCategory::Other,
    ] {
        assert_eq!(at(cat, 0).target, QuantTarget::Q4_K, "{cat:?}");
    }
}

/// `q5_K_M` is the same policy over a `Q5_K` base, with a different fused-qkv
/// answer — and the same `Q6_K` bumps.
#[test]
fn q5_k_m_selects_llama_cpps_per_tensor_types() {
    let ctx = tied_ctx(24);
    let t = Target::Mixture(Mixture::Q5_K_M);
    let at = |cat, layer: usize| t.select(&facts(cat, Some(layer), 896), &ctx);
    assert_eq!(t.base(), QuantTarget::Q5_K);
    assert_eq!(at(TensorCategory::AttnQkv, 0).target, QuantTarget::Q6_K);
    assert_eq!(at(TensorCategory::AttnV, 1).target, QuantTarget::Q6_K);
    assert_eq!(at(TensorCategory::FfnDown, 22).target, QuantTarget::Q6_K);
    assert_eq!(at(TensorCategory::AttnV, 3).target, QuantTarget::Q5_K);
    assert_eq!(at(TensorCategory::FfnDown, 4).target, QuantTarget::Q5_K);
    assert_eq!(at(TensorCategory::TokenEmbd, 0).target, QuantTarget::Q8_0);
    for cat in [
        TensorCategory::AttnQ,
        TensorCategory::AttnK,
        TensorCategory::AttnOutput,
        TensorCategory::FfnUp,
        TensorCategory::FfnGate,
        TensorCategory::Other,
    ] {
        assert_eq!(at(cat, 0).target, QuantTarget::Q5_K, "{cat:?}");
    }
}

/// The ftype-independent half: the legacy sub-8-bit targets take the
/// `OUTPUT`/tied branch (issue #203's measured divergence on the 0.6B), while the
/// pure K-quant uniforms, `q8_0` and the element casts do not move.
#[test]
fn legacy_targets_take_the_output_branch_and_pure_targets_do_not() {
    let ctx = tied_ctx(24);
    for t in [
        QuantTarget::Q4_0,
        QuantTarget::Q4_1,
        QuantTarget::Q5_0,
        QuantTarget::Q5_1,
    ] {
        // 1024 % 32 == 0, so the legacy request is answered with Q6_K (the 0.6B
        // divergence: the pre-#203 rule asked for q8_0 here).
        assert_eq!(
            Target::from(t).select(&facts(TensorCategory::TokenEmbd, None, 1024), &ctx),
            Selection {
                target: QuantTarget::Q6_K,
                reason: Reason::OutputBranch
            },
            "{}",
            t.name()
        );
        // A row the 32-block type cannot divide answers with Q8_0 instead — the
        // branch's row-length half, which is where the 0.5B's tied embedding
        // lands.
        assert_eq!(
            Target::from(t)
                .select(&facts(TensorCategory::TokenEmbd, None, 60), &ctx)
                .target,
            QuantTarget::Q8_0,
            "{}",
            t.name()
        );
        assert_eq!(
            Target::from(t)
                .select(&facts(TensorCategory::AttnV, Some(0), 1024), &ctx)
                .target,
            t,
            "{}: only the OUTPUT branch applies to a legacy target",
            t.name()
        );
    }
    assert_eq!(
        Target::from(QuantTarget::Q8_0)
            .select(&facts(TensorCategory::TokenEmbd, None, 1024), &ctx)
            .target,
        QuantTarget::Q8_0
    );
    for t in [
        QuantTarget::Q4_K,
        QuantTarget::Q5_K,
        QuantTarget::Q6_K,
        QuantTarget::F16,
        QuantTarget::F32,
    ] {
        let sel = Target::from(t).select(&facts(TensorCategory::TokenEmbd, None, 1024), &ctx);
        assert_eq!(sel.target, t, "{}", t.name());
        assert_eq!(sel.reason, Reason::Requested, "{}", t.name());
    }
}

/// The target's metadata: the spellings, the ftype numbers a mixture shares with
/// its uniform sibling, and the row-length answer taken from the base type.
#[test]
fn mixture_metadata_matches_llama_cpp() {
    for (name, t) in [
        ("q4_K_M", Target::Mixture(Mixture::Q4_K_M)),
        ("q5_K_M", Target::Mixture(Mixture::Q5_K_M)),
    ] {
        assert_eq!(Target::parse(name).unwrap(), t, "{name}");
        assert_eq!(
            Target::parse(&name.to_ascii_lowercase()).unwrap(),
            t,
            "{name}"
        );
        assert_eq!(t.name(), name);
        assert!(t.is_mixture());
    }
    assert_eq!(
        Target::parse("q4_K_M").unwrap().file_type(),
        15,
        "LLAMA_FTYPE_MOSTLY_Q4_K_M, the same number --pure records"
    );
    assert_eq!(
        Target::parse("q5_K_M").unwrap().file_type(),
        17,
        "LLAMA_FTYPE_MOSTLY_Q5_K_M"
    );
    let m = Target::parse("q4_K_M").unwrap();
    assert_eq!(m.base(), QuantTarget::Q4_K);
    assert_eq!(m.base().blck_size(), 256);
    assert_eq!(m.base().row_len_fallback(), Some(QuantTarget::Q5_0));
    let m5 = Target::parse("q5_K_M").unwrap();
    assert_eq!(m5.base(), QuantTarget::Q5_K);
    assert_eq!(m5.base().row_len_fallback(), Some(QuantTarget::Q5_1));
    // A concrete target keeps its own metadata, unchanged by the wrapper.
    let c = Target::parse("q8_0").unwrap();
    assert_eq!(c, Target::Concrete(QuantTarget::Q8_0));
    assert!(!c.is_mixture());
    assert_eq!(c.name(), "q8_0");
    assert_eq!(c.file_type(), 7);
    // A name no encoder knows is still refused by name, with the advertised list.
    let e = Target::parse("q4_K_S").unwrap_err();
    assert!(e.contains("unknown quant target"), "{e}");
    assert!(e.contains(crate::quantize::SUPPORTED_TARGETS), "{e}");
}

/// The row-length rule llama.cpp applies *after* the selection: the fallback
/// chain, one step at a time (a K-quant's 256-block falls to a 32-block legacy
/// type, and a 32-block type to `F16`).
fn demote(mut t: QuantTarget, row: i64) -> QuantTarget {
    while row % t.blck_size() as i64 != 0 {
        match t.row_len_fallback() {
            Some(f) => t = f,
            None => break,
        }
    }
    t
}

/// Run the selection + demotion over a synthetic model and return the 2-D type
/// histogram as `name:count` pairs, sorted.
fn histogram(t: Target, n_layers: usize, hidden: i64, ffn: i64) -> Vec<(String, usize)> {
    let ctx = tied_ctx(n_layers);
    let mut list: Vec<TensorFacts> = vec![facts(TensorCategory::TokenEmbd, None, hidden)];
    for i in 0..n_layers {
        for (cat, row) in [
            (TensorCategory::AttnQ, hidden),
            (TensorCategory::AttnK, hidden),
            (TensorCategory::AttnV, hidden),
            (TensorCategory::AttnOutput, hidden),
            (TensorCategory::FfnGate, hidden),
            (TensorCategory::FfnUp, hidden),
            (TensorCategory::FfnDown, ffn),
        ] {
            list.push(facts(cat, Some(i), row));
        }
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for f in &list {
        let sel = t.select(f, &ctx);
        let got = demote(sel.target, f.row).name().to_string();
        match counts.iter_mut().find(|(n, _)| *n == got) {
            Some((_, c)) => *c += 1,
            None => counts.push((got, 1)),
        }
    }
    counts.sort();
    counts
}

/// Both measured reference histograms fall out of the rules alone — no file, no
/// encoder.
///
/// The expected counts are the ones `llama-quantize <f16> <out> q4_K` writes
/// (`docs/GGUF-TOOLING.md` §2.2): 12 `q4_K` + 12 `q6_K` + 13 `q8_0` + 132 `q5_0`
/// (the 2-D half; the other 121 tensors are the 1-D norms) on the 0.5B, and
/// 168 `q4_K` + 29 `q6_K` with **no** demotion on the 0.6B, whose hidden size is
/// a multiple of 256.
#[test]
fn both_reference_histograms_follow_from_the_rules() {
    let q4km = Target::Mixture(Mixture::Q4_K_M);
    assert_eq!(
        histogram(q4km, 24, 896, 4864),
        vec![
            ("q4_K".to_string(), 12),
            ("q5_0".to_string(), 132),
            ("q6_K".to_string(), 12),
            ("q8_0".to_string(), 13),
        ],
        "the 0.5B (hidden 896)"
    );
    assert_eq!(
        histogram(q4km, 28, 1024, 3072),
        vec![("q4_K".to_string(), 168), ("q6_K".to_string(), 29),],
        "the 0.6B (hidden 1024)"
    );
}
