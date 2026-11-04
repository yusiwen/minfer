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

fn greedy_argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
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

    // Prefill then one greedy step per token, exactly the production entry point.
    let positions: Vec<usize> = (0..prompt_ids.len()).collect();
    let logits = model.forward(&prompt_ids, &positions, 1, n_ctx);
    let mut got = vec![greedy_argmax(&logits)];
    for step in 1..want.len() {
        let pos = prompt_ids.len() + step - 1;
        let logits = model.forward(&[got[step - 1]], &[pos], 1, n_ctx);
        got.push(greedy_argmax(&logits));
    }

    let mut bad = Vec::new();
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            bad.push(i);
        }
    }
    if !bad.is_empty() {
        let device = if cfg!(feature = "cuda") {
            "cuda build"
        } else {
            "cpu build"
        };
        panic!(
            "llama_pin: {model_name} ({device}) diverged from llama.cpp {} at {} of {} ids\n\
             got:      {got:?}\n\
             expected: {want:?}\n\
             first difference at index {} (got {}, expected {})",
            fx["llama_cpp"].as_str().unwrap_or("?"),
            bad.len(),
            want.len(),
            bad[0],
            got[bad[0]],
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
