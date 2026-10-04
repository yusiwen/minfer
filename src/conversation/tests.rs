//! `#[cfg(test)] mod tests` for `src/conversation.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use std::collections::VecDeque;
const EOS: u32 = 2;
const IM_END: u32 = 7;

mod overflow;
mod real_model;
mod regen;
mod snapshot;
mod spec;
mod turns;
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
/// The locally cached Qwen2.5-0.5B q4_0 the real-model tests run against.
fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `conversation.rs` (bucket B of the dead-code census —
// the only test caller lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl Conversation {
    /// The host state a KV session belongs to (C5 S2).
    ///
    /// The container carries the KV *rows*; this is everything the host needs to continue
    /// the conversation those rows were written for — which is what lets a resume prefill
    /// **nothing**. The message list is part of it (not just the token bookkeeping), so a
    /// resumed session can render its next turn's delta from the same history the KV was
    /// built from.
    ///
    /// Test-only (#239): driven by
    /// `conversation::tests::a_resumed_snapshot_prefills_nothing_and_continues_alike`
    /// (the `snap` the JSON round-trip and `restore_snapshot` compare against).
    pub fn snapshot(&self) -> ConversationSnapshot {
        ConversationSnapshot {
            messages: self.messages.clone(),
            stream_tokens: self.stream_tokens.clone(),
            current_pos: self.current_pos,
            turn_pos: self.turn_pos,
            prev_tokens: self.prev_tokens.clone(),
            need_insert_eot: self.need_insert_eot,
        }
    }
}
