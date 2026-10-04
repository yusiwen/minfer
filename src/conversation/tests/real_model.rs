//! The real-model smoke and the context-shift measurement.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// Real-model 2-turn smoke test (part of L2; ignored by default, consistent with the existing realdata tests):
///   cargo test --bin minfer conversation_real_model_smoke -- --ignored
/// Requires the locally cached Qwen2.5-0.5B q4_0 (skips if absent).
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn conversation_real_model_smoke() {
    let Some(path) = cached_qwen05_q4_0() else {
        eprintln!("0.5B q4_0 not cached; skipping conversation smoke");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ctx = &gguf.parts[0].ctx;
    let template = ctx
        .kv
        .iter()
        .find(|kv| kv.key == "tokenizer.chat_template")
        .map(|kv| kv.get_val_str(0).to_string());
    let special = model.special_tokens();
    let bos_text = tok
        .id_to_token
        .get(tok.bos_token as usize)
        .cloned()
        .unwrap_or_default();

    let spec = ConversationSpec {
        template,
        bos_text,
        eog: {
            let mut v = vec![special.eos];
            if let Some(im) = special.im_end {
                v.push(im);
            }
            v
        },
        eot: special.im_end.unwrap_or(special.eos),
        seed: 42,
        n_ctx: 512,
        mirostat_tau: 5.0,
        system_prompt: None,
    };
    let mut conv = Conversation::new(spec);
    let mut engine = GraphEngine::new(&*model, 512);
    let tp = TurnParams {
        n_predict: 16, // a short reply suffices (0.5B usually EOGs within a few dozen tokens)
        sampler: sampler::SamplerConfig {
            temp: 0.0, // greedy: deterministic and fast
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            ..sampler::SamplerConfig::default()
        },
        stop_strings: Vec::new(),
    };
    let mut emitted: Vec<u8> = Vec::new();
    let t1 = conv
        .start(Some("hi"), &tok, &tp, &mut engine, &mut |b| {
            emitted.extend_from_slice(b)
        })
        .unwrap()
        .expect("first turn ran");
    let t2 = conv
        .user_turn("what is 2+2?", &tok, &tp, &mut engine, &mut |b| {
            emitted.extend_from_slice(b)
        })
        .unwrap();
    eprintln!("t1 text: {:?}", t1.text);
    eprintln!("t2 text: {:?}", t2.text);
    assert!(!t1.text.is_empty(), "turn 1 must answer");
    assert!(!t2.text.is_empty(), "turn 2 must answer");
    assert!(
        !t1.text.contains('\u{FFFD}') && !t2.text.contains('\u{FFFD}'),
        "no U+FFFD"
    );
    // Incrementality: t2's delta prefill must be much smaller than t1's full prefill
    assert!(
        t2.prefill_tokens < t1.prefill_tokens,
        "t2 delta ({}) must be < t1 full render ({})",
        t2.prefill_tokens,
        t1.prefill_tokens
    );
    // Strong invariant: current_pos == stream_tokens.len() (the KV mirror is consistent)
    assert_eq!(conv.current_pos, conv.stream_tokens.len());
    assert_eq!(conv.messages.len(), 4, "user, assistant, user, assistant");
    // The model's own stop behaviour is **not** the engine's contract: on these
    // prompts the greedy 0.5B runs to the 16-token cap instead of emitting EOG
    // (t1 stops mid-sentence), so `need_insert_eot` is true. That is a fact about
    // a small model, not a defect — both stop branches are pinned by the
    // scripted-engine tests `first_turn_full_render_and_eog` (EOG -> no EOT) and
    // `n_predict_exhaustion_sets_eot` (cap -> EOT). What this real-model run must
    // hold is the rule that ties the flag to the state the *next* turn reads:
    // the EOT is owed exactly when the stream does not end on an EOG.
    let last = *conv.stream_tokens.last().expect("the session wrote tokens");
    eprintln!(
        "[smoke] t1 stopped_by_eog={} t2 stopped_by_eog={} last_stream_token={last} \
         need_insert_eot={}",
        t1.stopped_by_eog, t2.stopped_by_eog, conv.need_insert_eot
    );
    assert_eq!(
        conv.need_insert_eot,
        !conv.eog.contains(&last),
        "need_insert_eot ({}) must mirror whether the stream ends on an EOG (last stream \
         token {last}, eog {:?})",
        conv.need_insert_eot,
        conv.eog
    );
}
/// C2 real-model measurement: with a small context the conversation must
/// overflow, and the overflowing turn must *shift* the KV window — prefilling
/// only its own delta — instead of re-prefilling the retained render.
///
///   cargo test --release --bin minfer context_shift_real_model -- --ignored --nocapture
///
/// This is the check that the shift is actually reachable on a real model:
/// the token boundaries are re-derived from the chat template and verified
/// against the KV stream, which a byte-level test codec cannot exercise. The
/// printed numbers are the ones recorded in the execution plan's C2 record;
/// the assertion is that the shift fires, not that its logits match a fresh
/// prefill (they cannot — that is C2's tolerance class).
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn context_shift_real_model_measurement() {
    let Some(path) = cached_qwen05_q4_0() else {
        eprintln!("0.5B q4_0 not cached; skipping the context-shift measurement");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ctx = &gguf.parts[0].ctx;
    let template = ctx
        .kv
        .iter()
        .find(|kv| kv.key == "tokenizer.chat_template")
        .map(|kv| kv.get_val_str(0).to_string());
    let special = model.special_tokens();
    let bos_text = tok
        .id_to_token
        .get(tok.bos_token as usize)
        .cloned()
        .unwrap_or_default();

    // A context this small overflows after a handful of short turns; the
    // system prompt makes the retained prefix non-empty, which is the
    // conversation case the shift exists for.
    let n_ctx = 192;
    let spec = ConversationSpec {
        template,
        bos_text,
        eog: {
            let mut v = vec![special.eos];
            if let Some(im) = special.im_end {
                v.push(im);
            }
            v
        },
        eot: special.im_end.unwrap_or(special.eos),
        seed: 42,
        n_ctx,
        mirostat_tau: 5.0,
        system_prompt: Some("You are a terse assistant: answer in one short sentence.".into()),
    };
    let mut conv = Conversation::new(spec);
    let mut engine = GraphEngine::new(&*model, n_ctx);
    let tp = TurnParams {
        n_predict: 8, // keep the run short; a few tokens per reply still overflow n_ctx
        sampler: sampler::SamplerConfig {
            temp: 0.0,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            ..sampler::SamplerConfig::default()
        },
        stop_strings: Vec::new(),
    };
    let prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Egypt is",
        "The capital of Peru is",
        "The capital of Kenya is",
        "The capital of Norway is",
        "The capital of Sweden is",
        "The capital of Greece is",
        "The capital of Poland is",
        "The capital of Portugal is",
    ];
    let mut shifted = 0usize;
    let mut rehydrated = 0usize;
    for (i, p) in prompts.iter().enumerate() {
        let before = conv.stream_tokens.len();
        let out = conv
            .user_turn(p, &tok, &tp, &mut engine, &mut |_| {})
            .expect("turn");
        eprintln!(
            "[c2] real-model turn {i}: prefill {} tokens (stream {before} -> {}), \
             dropped {} turn(s), reply {:?}",
            out.prefill_tokens,
            conv.stream_tokens.len(),
            out.dropped_turns,
            out.text
        );
        if out.dropped_turns > 0 {
            if out.prefill_tokens < 40 {
                shifted += 1;
            } else {
                rehydrated += 1;
            }
        }
        assert_eq!(conv.current_pos, conv.stream_tokens.len());
        assert!(
            conv.stream_tokens.len() <= n_ctx,
            "the KV must stay inside n_ctx"
        );
    }
    assert_eq!(
        rehydrated, 0,
        "an overflow must shift the window, not re-prefill the retained render"
    );
    assert!(shifted > 0, "no turn overflowed n_ctx = {n_ctx}");
    eprintln!("[c2] context shift fired on {shifted} overflow(s); no full re-render was needed");
}
