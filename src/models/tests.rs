//! `#[cfg(test)] mod tests` for `src/models/mod.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// D3: the gate's whole matrix, with no device — the same reason E6 made
/// `batch_mode` pure: CI has no GPU, and this decides which graph a GPU run
/// builds.
#[test]
fn ffn_composition_is_opt_in_and_refuses_where_it_cannot_work() {
    // Default: the hand-written node, on every device.
    for d in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert!(!ffn_composition(None, d), "{d:?}");
    }
    // Forced on: allowed where offset views exist (CUDA), refused elsewhere
    // (loudly, in the function) so a Metal graph is never built with a
    // partial window it cannot express.
    assert!(ffn_composition(Some("1"), Device::Cuda));
    assert!(!ffn_composition(Some("1"), Device::Metal));
    assert!(!ffn_composition(Some("1"), Device::Cpu));
    // Forced off, and anything unrecognised, keeps the default.
    for v in ["0", "true", "banana", ""] {
        for d in [Device::Cpu, Device::Metal, Device::Cuda] {
            assert!(!ffn_composition(Some(v), d), "{v:?} {d:?}");
        }
    }
}

/// C8b S4 / #362: `gathers_attn_map` is the single authority for the set-valued
/// window, and every backend now answers yes — the CPU and CUDA kernels always
/// did, and Metal's sibling `kernel_gqa_attn_map_f32/_f16` closed the gap this
/// ticket is about. Pure, so CI (no GPU) covers it.
#[test]
fn every_device_gathers_the_attn_map() {
    for d in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert!(d.gathers_attn_map(), "{d:?} must gather a kv_map window");
    }
}

/// #723: a Llama file can declare an eos that is **not** its `<|eot_id|>` (the
/// cached 3.2 file sets eos = eot, which is why the real-model run cannot show
/// this), so the roles have to stand on their own rather than riding on the eos
/// fallback.
#[test]
fn the_stop_set_is_eos_plus_the_declared_roles() {
    let special = SpecialTokens {
        eos: 1,
        im_end: None,
        eot: Some(2),
        eom: Some(3),
    };
    assert!(special.is_stop(1), "eos is a stop id");
    assert!(special.is_stop(2), "eot is a stop id");
    assert!(special.is_stop(3), "eom is a stop id");
    assert!(!special.is_stop(4), "an ordinary id is not");

    // A Qwen-shaped set has no eot/eom, and its im_end still stops.
    let qwen = SpecialTokens {
        eos: 151645,
        im_end: Some(151645),
        eot: None,
        eom: None,
    };
    assert!(qwen.is_stop(151645));
    assert!(!qwen.is_stop(151646));
}

/// [#725]: the Llama 3.x real-model pin.
///
/// A gate in the [`docs/GATE-CONTRACT.md`] pattern: it loads a cached Llama GGUF,
/// feeds the **prompt ids llama.cpp itself produced**, greedily decodes 32 tokens
/// and compares the ids against a committed reference. The reference was generated
/// once from the same file with llama.cpp's own `/tokenize` + `/completion`
/// (`n_probs=1`), and its command and version are committed beside it.
///
/// Why ids and not text: the comparison then says nothing about the tokenizer — a
/// tokenizer change and a model change fail different gates. The template itself is
/// gated by `model_templates_render_byte_for_byte` (#734).
///
/// Unset `MINFER_LLAMA_TEST_MODEL` is a **named skip**, never a silent pass; a set
/// path whose bytes are not the pinned file is a failure (the ids are only
/// meaningful for that exact artifact). It lives here, not under
/// `models/qwen2/graph/`, because it is a fixture plus a loop over the public
/// `forward` — it contains no per-backend code of its own. Since #723 it does
/// **drive** the box's device on purpose (see the `MpsState::init` call in the
/// body): a gate that claims a device class must actually run there.
///
/// [#725]: https://github.com/yusiwen/minfer/issues/725
/// [`docs/GATE-CONTRACT.md`]: https://github.com/yusiwen/minfer/blob/master/docs/GATE-CONTRACT.md
const LLAMA_FIXTURE: &str = include_str!("../../tests/fixtures/llama/reference_tokens.json");

/// `sha2` is a dev-dependency (the same one #205 uses for the fixture manifest);
/// hashing here is what makes "the pinned file" a property instead of a hope.
fn sha256_of(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn fixture_ids(v: &serde_json::Value, key: &str) -> Vec<u32> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture lacks `{key}`"))
        .iter()
        .map(|x| {
            x.as_u64()
                .unwrap_or_else(|| panic!("`{key}` has a non-integer")) as u32
        })
        .collect()
}

/// One greedy step's answer: the arg-max and the runner-up, so a near-tie can be
/// told apart from a confident pick (the class difference lives in the margin).
fn top_two(x: &[f32]) -> ((u32, f32), (u32, f32)) {
    let mut b1 = (0u32, f32::NEG_INFINITY);
    let mut b2 = (0u32, f32::NEG_INFINITY);
    for (i, &v) in x.iter().enumerate() {
        if v > b1.1 {
            b2 = b1;
            b1 = (i as u32, v);
        } else if v > b2.1 {
            b2 = (i as u32, v);
        }
    }
    (b1, b2)
}

struct GreedySteps {
    ids: Vec<u32>,
    top2: Vec<[(u32, f32); 2]>,
}

/// The production decode loop (prefill, then one cached forward per token),
/// recording what the sampler sees at each step. The cache is local to the run,
/// never a process global ([#799]); prefill and decode share it so the KV
/// persists across the shape change.
///
/// [#799]: https://github.com/yusiwen/minfer/issues/799
fn greedy_steps(model: &dyn ModelDef, prompt_ids: &[u32], n: usize, n_ctx: usize) -> GreedySteps {
    let n_ctx = n_ctx.min(model.max_seq_len());
    let positions: Vec<usize> = (0..prompt_ids.len()).collect();
    let mut cache = crate::graph::cache::GraphCache::new();
    let mut logits = model.forward_graph_cached(prompt_ids, &positions, 1, n_ctx, &mut cache);
    let mut ids = Vec::with_capacity(n);
    let mut top2 = Vec::with_capacity(n);
    for step in 0..n {
        let (t1, t2) = top_two(&logits);
        ids.push(t1.0);
        top2.push([t1, t2]);
        let pos = prompt_ids.len() + step;
        logits = model.forward_graph_cached(&[t1.0], &[pos], 1, n_ctx, &mut cache);
    }
    GreedySteps { ids, top2 }
}

#[test]
#[ignore = "real model: set MINFER_LLAMA_TEST_MODEL to the pinned Llama 3.x GGUF"]
fn llama_greedy_tokens_match_the_pinned_llama_cpp_reference() {
    let fx: serde_json::Value = serde_json::from_str(LLAMA_FIXTURE).expect("llama fixture parses");
    let model_name = fx["model"].as_str().expect("model");
    let path = match std::env::var("MINFER_LLAMA_TEST_MODEL") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "llama_pin: MINFER_LLAMA_TEST_MODEL unset; skipping the {model_name} pin \
                 (set it to {} and re-run with --ignored)",
                fx["file"].as_str().unwrap_or("<the cached GGUF>")
            );
            return;
        }
    };
    let path = std::path::PathBuf::from(path);
    let want_sha = fx["sha256"].as_str().expect("sha256");
    let got_sha = sha256_of(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        got_sha,
        want_sha,
        "{} is not the pinned {model_name} file — the committed ids belong to the \
         artifact with sha256 {want_sha}; point MINFER_LLAMA_TEST_MODEL at that file",
        path.display()
    );

    // #723: on macOS nothing initialises MPS in a test process, so `model.device()`
    // answered `Cpu`, the forwards ran on the CPU, and the device arm this file grew in
    // #740 could never run on the only box that has Metal — the gate measured the CPU
    // under a device-shaped selector. The CLI initialises MPS before the load
    // (`main.rs`), and the loaders register weights against the live state, so the order
    // matters: init, then load. `MINFER_DISABLE_MPS=1` keeps the CPU arm reachable, which
    // is what that variable means everywhere else.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    crate::metal::MpsState::init();
    // [#752] run the pinned arm *after* a Metal reload: load, forward, drop, so
    // the zero-copy registration-release path has run in this process before the
    // Llama model is loaded and the 32 pinned ids are checked.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let toy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
        let toy_gguf = crate::gguf::load_gguf_model(&toy).expect("toy GGUF for the reload");
        let toy_model = crate::models::load_model_with(
            &toy_gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(1),
        )
        .expect("toy model for the reload");
        let mut cache = crate::graph::cache::GraphCache::new();
        let _ = toy_model.forward_graph_cached(&[1], &[0], 1, 16, &mut cache);
        drop(toy_model);
        drop(cache);
        drop(toy_gguf);
    }

    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    // A parallel test loading another architecture swaps same-named weight-registry
    // entries and would flip the device decision mid-run (same reason as
    // `graph_logits_match_forward_real_model`).
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();

    let prompt_ids = fixture_ids(&fx, "prompt_ids");
    let want = fixture_ids(&fx, "continuation_ids");
    let n_ctx = 2048usize;
    let got = greedy_steps(&*model, &prompt_ids, want.len(), n_ctx);

    // #740 gave the gate a device arm; #723's Metal half found the *selector* behind
    // it was still a build property. `cfg!(feature = "cuda")` asks what the crate was
    // compiled with, not where the forwards ran: a macOS release build carries no
    // `cuda` feature and still executes every `forward` on Metal (`Qwen2Graph::device`),
    // so it silently took the CPU arm and demanded token-exactness from a numeric
    // class that cannot promise it (rule 9: the CPU quantizes activations to Q8_0, the
    // devices read f32). `ModelDef::device` is the single authority for "a device
    // participates" (E6), so the arm follows it, and the fixture carries one golden
    // per class.
    let class = match model.device() {
        Device::Cpu => None,
        Device::Metal => Some("metal"),
        Device::Cuda => Some("cuda"),
    };

    if let Some(key) = class {
        // On a device the ids are pinned as that device's **golden** (its logits
        // differ from the CPU's by class, so a near-tie can legitimately flip), and
        // the prefix on which the two arms agree must stay inside the measured
        // cross-class tolerance.
        let dev = &fx["device"][key];
        assert!(
            !dev.is_null(),
            "llama_pin({key}): this run's forwards execute on the {key} device, but \
             tests/fixtures/llama/reference_tokens.json records no `device.{key}` golden. \
             Run this gate on that box and record the ids it produces (the `cuda` block is \
             the shape) — a device arm never falls back to the CPU's ids, because that asks \
             one numeric class to answer for another"
        );
        let golden = fixture_ids(dev, "ids");
        let tol = dev["tolerance"].as_f64().expect("device.tolerance") as f32;
        let cpu_top2 = fx["cpu_top2"].as_array().expect("cpu_top2");
        assert_eq!(cpu_top2.len(), want.len(), "fixture cpu_top2 length");

        let mut bad = Vec::new();
        for (i, (g, w)) in got.ids.iter().zip(golden.iter()).enumerate() {
            if g != w {
                bad.push(i);
            }
        }
        assert!(
            bad.is_empty(),
            "llama_pin({key}): {model_name} diverged from the recorded {key} golden at {} of \
             {} ids (first difference at index {}, got {}, expected {})\n\
             got:      {:?}\n\
             expected: {:?}\n\
             recorded: {} on {}",
            bad.len(),
            golden.len(),
            bad[0],
            got.ids[bad[0]],
            golden[bad[0]],
            got.ids,
            golden,
            dev["reference"].as_str().unwrap_or("?"),
            dev["box"].as_str().unwrap_or("?")
        );

        let mut prefix = 0usize;
        let mut worst = 0f32;
        for i in 0..got.ids.len() {
            let c1 = cpu_top2[i][0][0].as_u64().expect("cpu top1 id") as u32;
            if got.ids[i] != c1 {
                break;
            }
            let cl = cpu_top2[i][0][1].as_f64().expect("cpu top1 logit") as f32;
            worst = worst.max((got.top2[i][0].1 - cl).abs());
            prefix += 1;
        }
        assert!(
            prefix > 0,
            "llama_pin({key}): no agreeing prefix — {key} and the CPU reference disagree from \
             step 0, so the cross-class tolerance cannot be checked"
        );
        assert!(
            worst <= tol,
            "llama_pin({key}): the agreeing prefix's top logit moved {worst:.4} > the class \
             tolerance {tol} at step {}",
            prefix - 1
        );
        // A full agreement is the good case, and it is also the case where indexing
        // `cpu_top2[prefix]` would panic: the step that would report the divergence
        // does not exist.
        if prefix < got.ids.len() {
            eprintln!(
                "llama_pin({key}): {model_name} matched the recorded {key} golden on {} ids; \
                 the first {prefix} steps agree with the recorded CPU reference within {worst:.4} (class \
                 tolerance {tol}), and the trajectories diverge at step {prefix} (a near-tie: \
                 CPU margin {:.4}, {key} margin {:.4})",
                golden.len(),
                (cpu_top2[prefix][0][1].as_f64().unwrap()
                    - cpu_top2[prefix][1][1].as_f64().unwrap()) as f32,
                (got.top2[prefix][0].1 - got.top2[prefix][1].1)
            );
        } else {
            eprintln!(
                "llama_pin({key}): {model_name} matched the recorded {key} golden on {} ids and \
                 agreed with the recorded CPU reference on all {prefix} steps (worst top-logit |Δ| {worst:.4}, \
                 class tolerance {tol})",
                golden.len()
            );
        }
    } else {
        // The CPU arm: token-exact against llama.cpp **up to a near-tie the file's
        // rope table creates**. #724 made this engine divide each pair's angle by
        // the file's `rope_freqs.weight`, which is what llama.cpp has always done;
        // on this file the table moves step 5 from a 0.20 margin to a 0.037 tie on
        // llama.cpp's own numbers, and this engine lands 0.064 further (the CPU/CPU
        // numeric class). So the arm requires the recorded agreeing prefix, the
        // recorded candidate pair, and margins inside the class tolerance — never a
        // silently relaxed id comparison.
        let cpu = &fx["cpu"];
        let want_prefix = cpu["agreeing_prefix"]
            .as_u64()
            .expect("fixture cpu.agreeing_prefix") as usize;
        let tie = &cpu["tie"];
        let tie_cands: Vec<u32> = tie["candidates"]
            .as_array()
            .expect("fixture cpu.tie.candidates")
            .iter()
            .map(|v| v.as_u64().expect("candidate id") as u32)
            .collect();
        let tie_margin = tie["llama_cpp_margin"]
            .as_f64()
            .expect("fixture cpu.tie.llama_cpp_margin") as f32;
        let tol = cpu["tolerance"].as_f64().expect("fixture cpu.tolerance") as f32;

        let prefix = got
            .ids
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(want.len());
        if prefix < want_prefix {
            let (t1, t2) = (got.top2[prefix][0], got.top2[prefix][1]);
            panic!(
                "llama_pin: {model_name} agreed with llama.cpp {} on only {prefix} ids, short of \
                 the recorded {want_prefix}\n\
                 got:      {:?}\nexpected: {:?}\n\
                 at the first difference (step {prefix}) this run chose {}({:.4}) over \
                 {}({:.4}) — margin {:.4}",
                fx["llama_cpp"].as_str().unwrap_or("?"),
                got.ids,
                want,
                t1.0,
                t1.1,
                t2.0,
                t2.1,
                t1.1 - t2.1
            );
        }
        if prefix < want.len() {
            // The divergence must be the fixture's recorded near-tie: the same two
            // candidates, a margin below the class tolerance on both sides.
            let mut cands = [got.ids[prefix], want[prefix]];
            cands.sort_unstable();
            let mut recorded = tie_cands.clone();
            recorded.sort_unstable();
            assert_eq!(
                cands.to_vec(),
                recorded,
                "llama_pin: the divergence at step {prefix} is between {:?}, not the recorded \
                 near-tie candidates {:?}",
                cands,
                tie_cands
            );
            let mine = got.top2[prefix][0].1 - got.top2[prefix][1].1;
            assert!(
                tie_margin.abs() <= tol && mine.abs() <= tol,
                "llama_pin: the divergence at step {prefix} is not a tie any more — llama.cpp's \
                 recorded margin {tie_margin:.4}, this run's {mine:.4}, class tolerance {tol}"
            );
            eprintln!(
                "llama_pin: {model_name} matched llama.cpp on the first {prefix} ids and landed on \
                 the other side of the recorded near-tie at step {prefix} ({} vs {}; \
                 llama.cpp margin {tie_margin:.4}, this run {mine:.4}) — #724 applies the file's \
                 rope table, and the 8192-token gate carries the exact-id claim",
                got.ids[prefix], want[prefix]
            );
            return;
        }
        eprintln!(
            "llama_pin: {model_name} matched all {} greedy ids (prompt {} ids, {} + {})",
            want.len(),
            prompt_ids.len(),
            fx["llama_cpp"].as_str().unwrap_or("?"),
            fx["provenance"]["date"].as_str().unwrap_or("?")
        );
    }
}

const LLAMA_SCALED_FIXTURE: &str =
    include_str!("../../tests/fixtures/llama/scaled_reference_tokens.json");

/// [#724]: the long-context parity gate — a real Llama 3.1 GGUF at 8192 tokens.
///
/// It forwards 8192 tokens and compares 16 greedy ids against llama.cpp's own
/// output on the same file, so it covers the prefill path (positions, KV, rope at
/// 8192 distinct positions) against the reference rather than against itself.
///
/// **Measured limit, 2026-11-05: this gate does not discriminate the rope table.**
/// With `rope_scaling::resolve` stubbed to `Ok(None)` — the pre-#724 reading — the
/// same 16 ids still match (1251 s), because the pairs the table moves are the
/// low-frequency ones and this model's greedy trajectory has margins wider than
/// their effect at 8192 tokens. So it is a parity gate, not a scaling gate: the
/// table is pinned by the unit gates above (its bytes, the division, bit-identity
/// when absent) and by the 1B pin's arms, where applying it moves step 5 across a
/// 0.037-logit tie. A gate that discriminates the table needs a longer context or
/// a needle-style prompt — filed as a follow-up on #724.
///
/// `MINFER_LLAMA_SCALED_TEST_MODEL` unset is a **named skip**; a set path whose
/// bytes are not the pinned file fails (the ids belong to that artifact only).
#[test]
#[ignore = "real model: set MINFER_LLAMA_SCALED_TEST_MODEL to the pinned Llama-3.1-8B GGUF"]
fn llama_scaled_long_context_matches_the_pinned_reference() {
    let fx: serde_json::Value =
        serde_json::from_str(LLAMA_SCALED_FIXTURE).expect("scaled llama fixture parses");
    let model_name = fx["model"].as_str().expect("model");
    let path = match std::env::var("MINFER_LLAMA_SCALED_TEST_MODEL") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "llama_scaled: MINFER_LLAMA_SCALED_TEST_MODEL unset; skipping the {model_name} \
                 8k pin (set it to {} and re-run with --ignored)",
                fx["file"].as_str().unwrap_or("<the cached GGUF>")
            );
            return;
        }
    };
    let path = std::path::PathBuf::from(path);
    let want_sha = fx["sha256"].as_str().expect("sha256");
    let got_sha = sha256_of(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        got_sha,
        want_sha,
        "{} is not the pinned {model_name} file — the committed ids belong to the artifact \
         with sha256 {want_sha}",
        path.display()
    );

    #[cfg(all(target_os = "macos", feature = "metal"))]
    crate::metal::MpsState::init();
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();

    let prompt_ids = fixture_ids(&fx, "prompt_ids");
    let want = fixture_ids(&fx, "continuation_ids");
    let n_ctx = fx["n_ctx"].as_u64().expect("n_ctx") as usize;
    assert!(
        prompt_ids.len() + want.len() <= n_ctx,
        "the fixture's prompt {} + {} tokens must fit n_ctx {n_ctx}",
        prompt_ids.len(),
        want.len()
    );
    let got = greedy_steps(&*model, &prompt_ids, want.len(), n_ctx);
    // The 8192-token prompt is the point: the table's low-frequency pairs rotate by
    // tens of radians across it, so a dropped or mis-indexed table cannot survive.
    let first_bad = got
        .ids
        .iter()
        .zip(want.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(want.len());
    if first_bad != want.len() {
        let (t1, t2) = (got.top2[first_bad][0], got.top2[first_bad][1]);
        panic!(
            "llama_scaled: {model_name} diverged from llama.cpp at {first_bad} of {} ids\n\
             got:      {:?}\nexpected: {:?}\n\
             at that step this run's top-2 were {}({:.4}) and {}({:.4}) — margin {:.4}",
            want.len(),
            got.ids,
            want,
            t1.0,
            t1.1,
            t2.0,
            t2.1,
            t1.1 - t2.1
        );
    }
    eprintln!(
        "llama_scaled: {model_name} matched all {} greedy ids at {} prompt tokens",
        want.len(),
        prompt_ids.len()
    );
}

// ── #763 S3: a load failure is a value with a reason ────────────────────────

/// The toy fixture's bytes, for the two tests that need a *slightly* broken
/// file. A same-length byte edit keeps the GGUF structurally valid, so the test
/// exercises the intended refusal rather than a parse failure.
fn toy_gguf_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
    std::fs::read(&path).expect("the toy GGUF fixture")
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn write_patched(name: &str, edit: impl FnOnce(&mut Vec<u8>)) -> std::path::PathBuf {
    let mut bytes = toy_gguf_bytes();
    edit(&mut bytes);
    let path = std::env::temp_dir().join(format!("minfer-763-{name}-{}.gguf", std::process::id()));
    std::fs::write(&path, &bytes).expect("write the patched fixture");
    path
}

/// `general.architecture`'s *value* (5 bytes), not the `qwen2.*` keys that also
/// contain the word.
fn patch_architecture(bytes: &mut [u8], new: &[u8; 5]) {
    let key = b"general.architecture";
    let k = find_bytes(bytes, key).expect("the fixture declares an architecture");
    let mut p = k + key.len();
    let ty = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
    assert_eq!(ty, 8, "GGUF string type");
    p += 4;
    let len = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    assert_eq!(len, 5, "the toy architecture is five bytes");
    p += 8;
    assert_eq!(&bytes[p..p + 5], b"qwen2");
    bytes[p..p + 5].copy_from_slice(new);
}

#[test]
fn an_unsupported_architecture_is_a_named_error() {
    let path = write_patched("arch", |b| patch_architecture(b, b"bogus"));
    let gguf = crate::gguf::load_gguf_model(&path).expect("the patched file still parses");
    let err = load_model(&gguf).err().expect("a bogus arch must refuse");
    assert_eq!(err.arch, "bogus");
    assert!(
        err.to_string().contains("unsupported architecture"),
        "{err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_missing_token_embedding_is_a_named_error() {
    // 17 bytes in, 17 bytes out: the tensor-info name changes, the layout does not.
    let path = write_patched("no-embd", |b| {
        let at = find_bytes(b, b"token_embd.weight").expect("the fixture has token_embd");
        b[at..at + 17].copy_from_slice(b"token_embd.weighX");
    });
    let gguf = crate::gguf::load_gguf_model(&path).expect("the patched file still parses");
    let err = load_model(&gguf)
        .err()
        .expect("a missing tensor must refuse");
    assert!(
        err.reason.contains("missing tensor") && err.reason.contains("token_embd.weight"),
        "{err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unsupported_kv_cache_type_is_a_named_error() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
    let gguf = crate::gguf::load_gguf_model(&path).expect("the toy GGUF loads");
    let err = load_model_configured(
        &gguf,
        "",
        crate::graph::offload::OffloadRequest::Layers(0),
        Some("bogus"),
    )
    .err()
    .expect("a bogus KV cache type must refuse");
    assert!(err.reason.contains("bogus"), "{err}");
}
