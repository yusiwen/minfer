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
/// `models/qwen2/graph/`, because that path is a Metal-path trigger and this gate
/// needs no device code.
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

/// The production decode loop (prefill, then one `ModelDef::forward` per token),
/// recording what the sampler sees at each step.
fn greedy_steps(model: &dyn ModelDef, prompt_ids: &[u32], n: usize, n_ctx: usize) -> GreedySteps {
    let positions: Vec<usize> = (0..prompt_ids.len()).collect();
    let mut logits = model.forward(prompt_ids, &positions, 1, n_ctx);
    let mut ids = Vec::with_capacity(n);
    let mut top2 = Vec::with_capacity(n);
    for step in 0..n {
        let (t1, t2) = top_two(&logits);
        ids.push(t1.0);
        top2.push([t1, t2]);
        let pos = prompt_ids.len() + step;
        logits = model.forward(&[t1.0], &[pos], 1, n_ctx);
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

    if cfg!(feature = "cuda") {
        // #740: on a device build the ids are pinned as the **device golden** (the
        // device reads f32 activations where the CPU quantizes to Q8_0, so a
        // near-tie can legitimately flip), and the prefix on which the two arms
        // agree must stay inside the measured cross-class tolerance.
        let dev = &fx["device"];
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
            "llama_pin(device): {model_name} diverged from the recorded device golden at {} of \
             {} ids (first difference at index {}, got {}, expected {})\n\
             got:      {:?}\n\
             expected: {:?}",
            bad.len(),
            golden.len(),
            bad[0],
            got.ids[bad[0]],
            golden[bad[0]],
            got.ids,
            golden
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
            "llama_pin(device): no agreeing prefix — the device golden and the CPU reference \
             disagree from step 0, so the cross-class tolerance cannot be checked"
        );
        assert!(
            worst <= tol,
            "llama_pin(device): the agreeing prefix's top logit moved {worst:.4} > the class \
             tolerance {tol} at step {}",
            prefix - 1
        );
        eprintln!(
            "llama_pin(device): {model_name} matched the recorded device golden on {} ids; the \
             first {prefix} steps agree with the CPU arm within {worst:.4} (class tolerance \
             {tol}), and the trajectories diverge at step {prefix} (a near-tie: CPU margin \
             {:.4}, device margin {:.4})",
            golden.len(),
            (cpu_top2[prefix][0][1].as_f64().unwrap() - cpu_top2[prefix][1][1].as_f64().unwrap())
                as f32,
            (got.top2[prefix][0].1 - got.top2[prefix][1].1)
        );
    } else {
        // The CPU arm is the correctness gate: token-exact against llama.cpp.
        let mut bad = Vec::new();
        for (i, (g, w)) in got.ids.iter().zip(want.iter()).enumerate() {
            if g != w {
                bad.push(i);
            }
        }
        if !bad.is_empty() {
            panic!(
                "llama_pin: {model_name} diverged from llama.cpp {} at {} of {} ids\n\
                 got:      {:?}\n\
                 expected: {:?}\n\
                 first difference at index {} (got {}, expected {})",
                fx["llama_cpp"].as_str().unwrap_or("?"),
                bad.len(),
                want.len(),
                got.ids,
                want,
                bad[0],
                got.ids[bad[0]],
                want[bad[0]]
            );
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
