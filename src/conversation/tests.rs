//! `#[cfg(test)] mod tests` for `src/conversation.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use std::collections::VecDeque;

const EOS: u32 = 2;
const IM_END: u32 = 7;

/// Programmable mock engine: each forward pops the next token id from the program,
/// returning logits spiked at that id (temp=0 → greedy is forced to pick it).
/// The program must cover every forward call (including EOT insertion and delta prefill).
struct MockEngine {
    program: VecDeque<u32>,
    calls: Vec<(Vec<u32>, Vec<usize>, usize)>,
    resets: usize,
    vocab: usize,
    /// doc 97: when non-empty the engine carries a speculative draft;
    /// each spec_round pops one pre-accepted batch.
    spec_batches: VecDeque<Vec<u32>>,
    /// C2: when true the mock accepts KV row removals and records them;
    /// false is the plain engine (and GraphEngine without a shiftable
    /// backend), which makes the conversation re-render instead.
    shiftable: bool,
    /// C2: virtual written rows, so `kv_rm` can report the new count.
    rows: usize,
    shifts: Vec<(usize, usize)>,
}

impl MockEngine {
    fn new(program: Vec<u32>) -> Self {
        Self {
            program: program.into(),
            calls: Vec::new(),
            resets: 0,
            vocab: 4096,
            spec_batches: VecDeque::new(),
            shiftable: false,
            rows: 0,
            shifts: Vec::new(),
        }
    }
    fn call_tokens(&self) -> Vec<u32> {
        self.calls.iter().flat_map(|c| c.0.clone()).collect()
    }
}

impl Engine for MockEngine {
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
        self.calls
            .push((tokens.to_vec(), positions.to_vec(), n_out));
        if let Some(&p) = positions.last() {
            self.rows = self.rows.max(p + 1);
        }
        let id = self.program.pop_front().unwrap_or(EOS);
        let mut logits = vec![0.0f32; self.vocab];
        logits[id as usize] = 100.0;
        logits
    }
    fn reset_cache(&mut self) {
        self.resets += 1;
        self.rows = 0;
    }
    fn kv_rm(&mut self, start: usize, len: usize) -> Result<usize, String> {
        if !self.shiftable {
            return Err("mock engine without a KV".to_string());
        }
        assert!(start + len <= self.rows, "removal past the written rows");
        self.shifts.push((start, len));
        self.rows -= len;
        Ok(self.rows)
    }
    fn has_spec(&self) -> bool {
        !self.spec_batches.is_empty()
    }
    fn spec_round(
        &mut self,
        _seed: u32,
        _pos: usize,
        _s: &crate::spec::SpecSampler,
        _prev_tokens: &mut Vec<u32>,
        _rng: &mut StdRng,
    ) -> Option<Vec<u32>> {
        Some(self.spec_batches.pop_front().expect("batch queued"))
    }
}

/// Fake codec: same semantics as the real tokenizer — template special markers are **single** token ids
/// (`<|im_end|>` = IM_END, `<|im_start|>` = 7000), everything else is encoded byte-wise.
/// This aligns the canonical form of `tokenize(render(...))` with the single-token EOG/EOT in the KV,
/// so the §5.4 invariant can be asserted at the token level (a byte-wise codec would split `<|im_end|>` into 10 tokens).
struct FakeCodec;

impl FakeCodec {
    const IM_START: u32 = 7000;
}

impl TokenCodec for FakeCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(r) = rest.strip_prefix("<|im_end|>") {
                out.push(IM_END);
                rest = r;
            } else if let Some(r) = rest.strip_prefix("<|im_start|>") {
                out.push(Self::IM_START);
                rest = r;
            } else {
                let b = rest.as_bytes()[0];
                out.push(b as u32);
                rest = &rest[1..];
            }
        }
        out
    }
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            match id {
                IM_END => out.extend_from_slice(b"<|im_end|>"),
                Self::IM_START => out.extend_from_slice(b"<|im_start|>"),
                b => out.push(b as u8),
            }
        }
        out
    }
}

fn cfg() -> TurnParams {
    TurnParams {
        n_predict: 512,
        // greedy: the mock spike is always picked
        sampler: sampler::SamplerConfig {
            temp: 0.0,
            top_k: 4096,
            top_p: 1.0,
            repeat_penalty: 1.0,
            ..sampler::SamplerConfig::default()
        },
        stop_strings: Vec::new(),
    }
}

fn spec(n_ctx: usize) -> ConversationSpec {
    ConversationSpec {
        template: None, // ChatML fallback
        bos_text: String::new(),
        eog: vec![EOS, IM_END],
        eot: IM_END,
        seed: 42,
        n_ctx,
        mirostat_tau: 5.0,
        system_prompt: None,
    }
}

fn conv(n_ctx: usize) -> Conversation {
    Conversation::new(spec(n_ctx))
}

/// The snapshot JSON with its `version` field replaced (the refuse-not-guess test).
fn json_with_version(json: &str, version: u32) -> String {
    let mut v: serde_json::Value = serde_json::from_str(json).unwrap();
    v["version"] = serde_json::json!(version);
    v.to_string()
}

fn noop_emit() -> impl FnMut(&[u8]) {
    |_| {}
}

/// Byte-level tokenization of the canonical render (ChatML fallback, no generation prompt).
fn canonical(messages: &[(String, Option<String>)]) -> Vec<u32> {
    FakeCodec.encode(&template::fallback_chatml_messages(messages, false))
}

fn fallback_full(messages: &[(String, Option<String>)]) -> Vec<u32> {
    FakeCodec.encode(&template::fallback_chatml_messages(messages, true))
}

#[test]
fn start_no_input_waits() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![]);
    let out = c
        .start(None, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.is_none());
    assert!(c.messages.is_empty());
    assert_eq!(c.current_pos, 0);
}

#[test]
fn first_turn_full_render_and_eog() {
    let mut c = conv(512);
    // 'H','i', EOG, EOG-decode placeholder
    let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS]);
    let mut emitted: Vec<u8> = Vec::new();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut |b| {
            emitted.extend_from_slice(b)
        })
        .unwrap()
        .unwrap();
    assert!(out.stopped_by_eog);
    assert_eq!(out.text, "Hi");
    // First turn: full render (with generation prompt) + generation + EOG (§5.4: EOG enters the KV)
    let full = fallback_full(&[("user".into(), Some("hi".into()))]);
    assert_eq!(out.prefill_tokens, full.len());
    assert_eq!(c.stream_tokens, [&full[..], &[72, 105, IM_END]].concat());
    assert_eq!(
        c.messages,
        vec![
            ("user".into(), Some("hi".into())),
            ("assistant".into(), Some("Hi".into())),
        ]
    );
    assert!(!c.need_insert_eot);
    assert_eq!(emitted, b"Hi");
}

#[test]
fn second_turn_appends_only_delta() {
    let mut c = conv(512);
    // t1: EOG, EOG-decode placeholder; t2: EOG, EOG-decode placeholder
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let len_after_t1 = c.stream_tokens.len();
    let t2 = c
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert!(t2.stopped_by_eog);
    // delta = "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n" (with newline compensation)
    let delta = FakeCodec.encode("\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(t2.prefill_tokens, delta.len());
    // Incrementality: only the delta + this turn's EOG are appended (hit immediately, no generated tokens)
    assert_eq!(c.stream_tokens.len(), len_after_t1 + delta.len() + 1);
    assert_eq!(
        &c.stream_tokens[len_after_t1..len_after_t1 + delta.len()],
        &delta[..]
    );
    assert_eq!(c.stream_tokens.last(), Some(&IM_END));
    // turn_pos points to the start of t2's delta (= the position after t1)
    assert_eq!(c.turn_pos, len_after_t1);
    // The assistant message is recorded (immediate EOG this turn → empty text)
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some(""));
}

#[test]
fn stop_string_truncates_and_sets_eot() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
    let mut tp = cfg();
    tp.stop_strings = vec!["!".to_string()];
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.stopped_by_string);
    assert!(!out.stopped_by_eog);
    assert_eq!(out.text, "Hi");
    assert!(
        c.need_insert_eot,
        "stop-string termination → EOT needed before the next turn"
    );

    // The next turn inserts EOT first (the engine receives the [eot] call), then runs the delta.
    let mut eng2 = MockEngine::new(vec![999, IM_END]); // 999 = dummy for the EOT insertion
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(eng2.calls[0].0, vec![IM_END], "first call must insert EOT");
    assert_eq!(
        eng2.calls[0].1,
        vec![c.turn_pos - 1],
        "EOT written at the position before the delta"
    );
    // After EOT, before delta: the stream prefix == the canonical prefix (missing the trailing template newline)
    let canon = canonical(&[
        ("user".into(), Some("hi".into())),
        ("assistant".into(), Some("Hi".into())),
    ]);
    assert_eq!(c.stream_tokens[..canon.len() - 1], canon[..canon.len() - 1]);
}

/// C5 S2's host-state half: a conversation restored from a session snapshot starts
/// with **nothing prefilled** and its next turn issues exactly the engine calls the
/// in-memory run's does (delta prefill + decodes — never a re-render of the history).
/// The KV bytes that make that legitimate are the C5 container's own gate.
#[test]
fn a_resumed_snapshot_prefills_nothing_and_continues_alike() {
    // The in-memory run: one turn, then the snapshot `--session` would write on exit.
    // The mock pops one program token per `forward` — including prefill calls — so the
    // program below is the one that makes turn 2 sample '!' on both sides.
    let mut base = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, EOS, 33, 33, EOS]); // 'H','i',EOG,(EOG write),'!',EOG
    base.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let snap = base.snapshot();
    let json = base.snapshot_to_json();
    let parsed = Conversation::snapshot_from_json(&json).expect("the snapshot round-trips");
    assert_eq!(parsed, snap, "JSON must carry the snapshot losslessly");

    // The resumed run: a fresh conversation and a fresh (empty) engine, as a second
    // `--session` process would have.
    let mut resumed = conv(512);
    let mut eng2 = MockEngine::new(vec![33, EOS]);
    resumed.restore_snapshot(&parsed).expect("restore");
    assert!(
        eng2.calls.is_empty() && eng2.resets == 0,
        "restoring a snapshot must not touch the engine at all"
    );
    assert_eq!(resumed.messages, base.messages);
    assert_eq!(resumed.stream_tokens, base.stream_tokens);
    assert_eq!(resumed.current_pos, base.current_pos);
    assert_eq!(resumed.prev_tokens, base.prev_tokens);
    assert_eq!(resumed.need_insert_eot, base.need_insert_eot);

    // The next turn on both: same input, same engine calls, same output.
    let before = eng.calls.len();
    let out_base = base
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let base_calls: Vec<(Vec<u32>, Vec<usize>)> = eng.calls[before..]
        .iter()
        .map(|(t, p, _)| (t.clone(), p.clone()))
        .collect();
    let out_resumed = resumed
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    let resumed_calls: Vec<(Vec<u32>, Vec<usize>)> = eng2
        .calls
        .iter()
        .map(|(t, p, _)| (t.clone(), p.clone()))
        .collect();
    assert_eq!(
        base_calls, resumed_calls,
        "a resumed turn must issue the same forwards, at the same positions"
    );
    assert_eq!(out_base.text, out_resumed.text, "greedy tokens must agree");
    assert_eq!(out_resumed.text, "!");
    // And the first call is the turn's own delta, not the history re-rendered.
    let full_render = fallback_full(&base.messages);
    assert!(
        resumed_calls[0].0.len() < full_render.len(),
        "the resumed turn prefilled {} token(s) — a re-render would be {}",
        resumed_calls[0].0.len(),
        full_render.len()
    );
    assert_eq!(resumed.messages, base.messages);
    assert_eq!(resumed.stream_tokens, base.stream_tokens);
    assert_eq!(resumed.current_pos, base.current_pos);
}

/// C5 S2: a snapshot that contradicts the host mirror is refused *before* it is
/// applied — the caller then re-seeds, which is always safe.
#[test]
fn a_snapshot_that_contradicts_the_host_mirror_is_refused() {
    let mut c = conv(512);
    let good = ConversationSnapshot {
        messages: vec![("user".into(), Some("hi".into()))],
        stream_tokens: vec![1, 2, 3],
        current_pos: 3,
        turn_pos: 0,
        prev_tokens: vec![2, 3],
        need_insert_eot: false,
    };
    c.restore_snapshot(&good)
        .expect("a consistent snapshot applies");

    // current_pos must equal the token stream's length.
    let mut bad = good.clone();
    bad.current_pos = 4;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("host mirror"), "{err}");
    // ... and must fit this run's n_ctx.
    let mut bad = good.clone();
    bad.stream_tokens = vec![1; 600];
    bad.current_pos = 600;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("n_ctx"), "{err}");
    // ... and turn_pos may not run past it.
    let mut bad = good.clone();
    bad.turn_pos = 9;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("turn_pos"), "{err}");

    // A JSON this build does not understand is `None`, never a guess.
    assert!(Conversation::snapshot_from_json("{}").is_none());
    assert!(Conversation::snapshot_from_json("not json").is_none());
    let bumped = json_with_version(&c.snapshot_to_json(), SNAPSHOT_VERSION + 1);
    assert!(Conversation::snapshot_from_json(&bumped).is_none());
    // The refused snapshots above must not have touched the conversation.
    assert_eq!(c.current_pos, 3);
    assert_eq!(c.stream_tokens, vec![1, 2, 3]);
}

/// C5 S2: an engine with no arena (the mocks, and any backend that cannot hand its KV
/// to the host) refuses both calls, so the CLI falls back to re-seeding instead of
/// pretending the session was resumed.
#[test]
fn an_engine_without_a_kv_refuses_the_session_calls() {
    let mut eng = MockEngine::new(vec![]);
    let path = std::path::Path::new("/tmp/minfer-c5s2-not-a-kv-file");
    assert!(eng.kv_save(path, b"{}").is_err());
    assert!(eng.kv_load(path).is_err());
}

#[test]
fn n_predict_exhaustion_sets_eot() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33, 34]);
    let mut tp = cfg();
    tp.n_predict = 2;
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.hit_n_predict);
    assert_eq!(out.text, "Hi");
    assert_eq!(out.tokens_generated, 2);
    assert!(c.need_insert_eot);
}

#[test]
fn regen_rolls_back_and_regenerates() {
    let mut c = conv(512);
    // t1: EOG, placeholder; t2: 'P', EOG, placeholder; regen: 'W', EOG, placeholder
    let mut eng = MockEngine::new(vec![IM_END, EOS, 80, IM_END, EOS, 87, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.messages.last().unwrap().0, "assistant");
    let turn2_start = c.turn_pos;

    // regen t2: roll back to turn2_start, replay t2's delta, generate 'W'
    let mut eng2 = MockEngine::new(vec![87, IM_END, EOS]); // 'W', EOG, placeholder
    let out = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(out.text, "W");
    assert!(out.stopped_by_eog);
    // messages: user hi, assistant "", user Q, assistant W
    assert_eq!(c.messages.len(), 4);
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("W"));
    // t2's old content 'P' has been removed from the stream
    assert!(!c.stream_tokens.contains(&80));
    assert_eq!(c.turn_pos, turn2_start);
}

#[test]
fn spec_rounds_commit_batches_and_stop_at_eog() {
    // doc 97: the spec loop must mirror the plain loop's stream/window
    // semantics. The mock's spec batches replace the sampling: the seed
    // comes from the prefill spike, then each round commits one batch.
    let mut c = conv(512);
    // program[0] spikes the prefill logits → the turn's seed ('z');
    // the spares back the mock's per-forward logits afterwards.
    let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS, EOS]);
    // The prefill consumes program[0] (its spike seeds the turn); the
    // remaining spikes back the mock's per-call logits if a plain forward
    // ever runs (it should not, beyond the EOG slot write).
    eng.spec_batches = vec![
        vec![b'a' as u32, b'b' as u32, b'c' as u32], // round 1 batch
        vec![IM_END],                                // round 2: EOG in-batch
    ]
    .into();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap()
        .unwrap();
    // The seed ('z', sampled from the prefill spike) is the first
    // committed token; the batch follows it.
    assert_eq!(out.text, "zabc");
    assert!(out.stopped_by_eog);
    assert!(!out.stopped_by_string);
    // The EOG is not counted (plain-loop mirror).
    assert_eq!(out.tokens_generated, 4);
    // The assistant message holds the batch text; the stream holds the
    // committed tokens plus the EOG.
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("zabc"));
    assert!(c.stream_tokens.contains(&(b'a' as u32)));
    assert!(c.stream_tokens.contains(&IM_END));
    // need_insert_eot stays false: the turn reached EOG.
    assert!(!c.need_insert_eot);
}

#[test]
fn spec_round_mid_batch_stop_string_truncates() {
    // The stop string lands mid-batch: committed tokens stop at the cut,
    // the stop tokens are not part of the canonical text.
    let mut cfgv = cfg();
    cfgv.stop_strings = vec!["bc".to_string()];
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS]);
    eng.spec_batches = vec![vec![b'a' as u32, b'b' as u32, b'c' as u32, b'd' as u32]].into();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfgv, &mut eng, &mut noop_emit())
        .unwrap()
        .unwrap();
    // The cut lands mid-batch (at 'c'): committed text stops before the
    // stop string, the batch tail ('d') is discarded uncommitted. The
    // tokens through the stop-completing one are counted (the plain loop
    // increments n_gen before its stop check — mirror exactly).
    assert_eq!(out.text, "za");
    assert!(out.stopped_by_string);
    assert!(!out.stopped_by_eog);
    assert_eq!(out.tokens_generated, 4);
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("za"));
}

#[test]
fn regen_first_turn_uses_full_render() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.turn_pos, 0);

    // regen t1: turn_pos == 0 → full render (the engine's first call must be the full render, not a suffix)
    let mut eng2 = MockEngine::new(vec![88, IM_END, EOS]); // 'X', EOG, placeholder
    let out = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(out.text, "X");
    let full = fallback_full(&[("user".into(), Some("hi".into()))]);
    assert_eq!(
        eng2.calls[0].0, full,
        "first call must be the full first-turn render"
    );
}

#[test]
fn regen_without_assistant_errors() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![]);
    let err = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::NothingToRegen));
}

#[test]
fn clear_resets_state_keeps_system() {
    let mut s = spec(512);
    s.system_prompt = Some("Be nice.".into());
    let mut c = Conversation::new(s);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.clear();
    assert_eq!(c.messages, vec![("system".into(), Some("Be nice.".into()))]);
    assert!(c.stream_tokens.is_empty());
    assert_eq!(c.current_pos, 0);
    assert_eq!(c.turn_pos, 0);
    assert!(!c.need_insert_eot);
}

#[test]
fn context_full_errors_before_prefill() {
    let mut c = conv(32); // a very small n_ctx
    let mut eng = MockEngine::new(vec![IM_END]);
    let long_input = "x".repeat(64);
    let err = c
        .user_turn(&long_input, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::ContextFull { .. }));
    // State is not polluted: no forward calls at all
    assert!(eng.calls.is_empty());
    assert!(c.messages.is_empty());
}

#[test]
fn context_fill_during_decode_stops_cleanly() {
    // fallback([user hi], true) tokenizes to 21 tokens (special markers are 1 token each)
    let mut c = conv(30);
    // The first turn's delta takes 21; decoding fills all the way up to n_ctx=30
    let mut eng = MockEngine::new(vec![72, 105, 33, 34, 35, 36, 37, 38, 39, 40, IM_END]);
    let mut tp = cfg();
    tp.n_predict = 100;
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(
        out.hit_n_predict,
        "context full should stop as cleanly as n_predict exhaustion"
    );
    assert!(c.current_pos <= 30);
    assert!(c.need_insert_eot);
}

#[test]
fn prev_tokens_reseeded_per_turn() {
    let mut c = conv(512);
    // t1: 'H','i', EOG, placeholder; t2: EOG, placeholder
    let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    // After t1, prev_tokens = the last 64 stream tokens (including delta + generated + EOG)
    assert_eq!(
        c.prev_tokens.len(),
        c.stream_tokens.len().min(REPEAT_LAST_N)
    );
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(
        c.prev_tokens.len(),
        c.stream_tokens.len().min(REPEAT_LAST_N)
    );
}

#[test]
fn eot_inserted_before_first_user_turn_after_no_eog() {
    // After start, the first turn ends via a stop string → the second user_turn inserts EOT first
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
    let mut tp = cfg();
    tp.stop_strings = vec!["!".to_string()];
    c.start(Some("hi"), &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(c.need_insert_eot);

    let mut eng2 = MockEngine::new(vec![999, IM_END]);
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(eng2.calls[0].0, vec![IM_END]);
}

// === Phase 3: overflow truncation + session persistence ===

/// C2: with a shiftable engine, an overflowing turn removes the dropped
/// turn's rows from the KV instead of re-rendering, so it prefills only its
/// own delta — and the resulting stream is still the canonical render of the
/// new message list (the §5.4 invariant), which is what makes the shortcut
/// safe to take.
#[test]
fn overflow_shift_removes_the_turn_and_prefills_only_the_delta() {
    let mut c = conv(50);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 44);

    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    eng3.shiftable = true;
    eng3.rows = c.current_pos; // the stub mirrors the session's written rows
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();

    assert_eq!(out.dropped_turns, 1, "the oldest turn is dropped");
    assert_eq!(eng3.resets, 0, "a shift must not reset the cache");
    assert_eq!(eng3.shifts.len(), 1, "exactly one KV removal");
    let (start, len) = eng3.shifts[0];
    // "hi" and its reply are gone; Q's turn is still at the stream head.
    assert_eq!(start, 0, "Q's turn starts at the stream head");
    assert_eq!(
        len, 23,
        "the dropped turn's span, including its trailing newline"
    );
    // The overflow turn prefills its own delta, not the whole render.
    // (`c.messages` now ends with the new reply; the canonical prompt is the
    // messages as they were when the turn's prefill ran.)
    let full = fallback_full(&c.messages[..c.messages.len() - 1]);
    let delta = format_single(
        None,
        &c.messages[..c.messages.len() - 2],
        ("user".to_string(), Some("X".to_string())),
        true,
        "",
    )
    .expect("diff render");
    assert_eq!(
        out.prefill_tokens,
        FakeCodec.encode(&delta.text).len(),
        "only the overflow turn's own delta is prefilled"
    );
    assert!(
        out.prefill_tokens < full.len(),
        "the shift must not re-prefill the render ({} vs {})",
        out.prefill_tokens,
        full.len()
    );

    // The §5.4 invariant survives the removal: the stream is exactly the
    // canonical render of the final message list.
    assert_eq!(c.messages.len(), 4);
    assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
    assert_eq!(c.current_pos, c.stream_tokens.len());
}

/// C2: a system prompt is *not* part of the removed region — the drop starts
/// after it, so the shift is a middle removal (`start > 0`), and the system
/// prompt's rows are not even re-roped.
#[test]
fn overflow_shift_keeps_the_system_prompt_in_place() {
    let mut spec = spec(60);
    spec.system_prompt = Some("sys".to_string());
    let mut c = Conversation::new(spec);
    // The system prompt is part of the first turn's delta; the boundary check
    // must still resolve it (ChatML renders it as its own block).
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    eng.shiftable = true;
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let sys_tokens = canonical(&[("system".into(), Some("sys".into()))]);
    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    eng3.shiftable = true;
    eng3.rows = c.current_pos; // the stub mirrors the session's written rows
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();
    assert_eq!(out.dropped_turns, 1);
    assert_eq!(eng3.shifts.len(), 1, "the shift must fire");
    let (start, _) = eng3.shifts[0];
    assert_eq!(
        start,
        sys_tokens.len(),
        "the removed region starts right after the system prompt"
    );
    assert_eq!(
        &c.stream_tokens[..sys_tokens.len()],
        &sys_tokens[..],
        "the system prompt's tokens stay at the head"
    );
    assert_eq!(
        c.messages[0],
        ("system".into(), Some("sys".into())),
        "the system prompt survives the drop"
    );
}

#[test]
fn overflow_truncates_oldest_turns_and_rehydrates() {
    // n_ctx=50: turn1(22) + turn2(44) both fit; turn3's delta makes
    // current_pos + delta > 50 → drop the oldest user+assistant pair and fully re-render.
    let mut c = conv(50);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 22, "t1: 21-token render + EOG");
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 44, "t2: +21 delta + EOG");

    // t3 triggers overflow: re-render (engine.reset) + drop 1 turn + full render
    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();
    assert_eq!(out.dropped_turns, 1, "oldest turn must be dropped");
    assert_eq!(eng3.resets, 1, "rehydrate must reset the engine cache");
    // Messages: the oldest [user hi, assistant] pair has been dropped
    assert_eq!(
        c.messages,
        vec![
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
            ("user".into(), Some("X".into())),
            ("assistant".into(), Some("".into())),
        ]
    );
    // KV = the full render after re-render (turn_pos reset to zero) + EOG
    let canon = FakeCodec.encode(&template::fallback_chatml_messages(
        &[
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
        ],
        false,
    ));
    assert_eq!(c.turn_pos, 0);
    let full = FakeCodec.encode(&template::fallback_chatml_messages(
        &[
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
            ("user".into(), Some("X".into())),
        ],
        true,
    ));
    assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
    assert_eq!(
        c.stream_tokens.len(),
        canon.len() + full.len() - canon.len() + 1
    );
}

#[test]
fn overflow_single_message_still_errors() {
    let mut c = conv(20); // smaller than a single message's render length
    let mut eng = MockEngine::new(vec![]);
    let err = c
        .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::ContextFull { .. }));
}

#[test]
fn session_json_round_trip() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let json = c.messages_to_json();
    let parsed = Conversation::messages_from_json(&json).expect("parse");
    assert_eq!(parsed, c.messages);
    // null content is preserved (OpenAI-style object format)
    let with_null = vec![
        ("user".into(), Some("hi".into())),
        ("assistant".into(), None),
    ];
    let with_null_json = serde_json::json!([
        { "role": "user", "content": "hi" },
        { "role": "assistant", "content": null },
    ])
    .to_string();
    let j2 = Conversation::messages_from_json(&with_null_json).unwrap();
    assert_eq!(j2, with_null);
    // Invalid input → None
    assert!(Conversation::messages_from_json("not json").is_none());
}

#[test]
fn load_history_rehydrates_full_render() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let saved = c.messages_to_json();

    // A new session loads the history → full re-render (render(messages, false))
    let mut c2 = conv(512);
    let mut eng2 = MockEngine::new(vec![IM_END, EOS]);
    let msgs = Conversation::messages_from_json(&saved).unwrap();
    c2.load_history(msgs, &FakeCodec, &mut eng2)
        .expect("load history");
    assert_eq!(c2.messages, c.messages);
    assert_eq!(c2.current_pos, c2.stream_tokens.len());
    let canon = canonical(&c.messages);
    assert_eq!(
        c2.stream_tokens, canon,
        "KV = canonical render of loaded history"
    );
    assert_eq!(c2.turn_pos, 0);

    // Continue the conversation after loading (the incremental delta continues from the re-rendered KV)
    let out = c2
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert!(out.stopped_by_eog);
    assert_eq!(c2.messages.len(), 4);
    assert!(c2.current_pos > c2.stream_tokens.len().saturating_sub(1));
    assert_eq!(c2.current_pos, c2.stream_tokens.len());
}

/// The very first `user_turn` has no KV yet, so it must prefill the whole
/// render — including anything already in `messages`, i.e. the `--system`
/// prompt. Before C2 the delta was prefilled on its own, which silently
/// dropped the system prompt from the KV.
#[test]
fn first_user_turn_prefills_the_system_prompt() {
    let mut sp = spec(128);
    sp.system_prompt = Some("be brief".to_string());
    let mut c = Conversation::new(sp);
    let mut eng = MockEngine::new(vec![IM_END]);
    let out = c
        .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();

    let full = fallback_full(&[
        ("system".into(), Some("be brief".into())),
        ("user".into(), Some("hi".into())),
    ]);
    assert_eq!(
        c.stream_tokens,
        [&full[..], &[IM_END]].concat(),
        "the first turn's KV must be the canonical render + EOG"
    );
    assert_eq!(out.prefill_tokens, full.len());
    let sys = canonical(&[("system".into(), Some("be brief".into()))]);
    assert_eq!(
        &c.stream_tokens[..sys.len()],
        &sys[..],
        "the system prompt must reach the KV"
    );
}

/// The locally cached Qwen2.5-0.5B q4_0 the real-model tests run against.
fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}

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
