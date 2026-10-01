//! `#[cfg(test)] mod tests` for `src/tokenizer.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// Gives a synthetic tokenizer the 256 byte tokens every real byte-level
/// BPE vocabulary has (and which `Tokenizer::load` verifies). Without them
/// the byte fallback has nothing to emit, which is a load-time refusal in
/// production and an invariant violation here.
fn fill_byte_tokens(t: &mut Tokenizer) {
    for (b, c) in t.byte_to_unicode.clone() {
        t.vocab.entry(c.to_string()).or_insert(1000 + b as u32);
    }
}

/// Rebuilds `special_by_first` the way `load()` does: GGUF special tokens
/// plus the hardcoded `<|im_start|>` / `<|im_end|>` fallbacks.
fn rebuild_special_index(t: &mut Tokenizer) {
    let mut merged: HashMap<String, u32> = t.special_tokens.clone();
    if !merged.contains_key("<|im_start|>") {
        merged.insert("<|im_start|>".into(), t.im_start);
    }
    if !merged.contains_key("<|im_end|>") {
        merged.insert("<|im_end|>".into(), t.im_end);
    }
    for (pat, id) in merged {
        let first = pat.chars().next().unwrap_or('\0');
        t.special_by_first.entry(first).or_default().push((pat, id));
    }
    for group in t.special_by_first.values_mut() {
        group.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }
}

/// Minimal tokenizer for decode tests: id 2 is the byte-level encoding of
/// the CJK char U+4E2D (E4 B8 AD in UTF-8, i.e. bytes 0xE4 0xB8 0xAD).
fn test_tokenizer() -> Tokenizer {
    let byte_to_unicode = build_byte_to_unicode();
    let unicode_to_byte = byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();
    let mut t = Tokenizer {
        id_to_token: vec!["hello".into(), " world".into(), "ä¸Ń".into()],
        id_to_score: vec![0.0; 3],
        id_to_type: vec![1; 3],
        vocab: HashMap::new(),
        merges: HashMap::new(),
        byte_to_unicode,
        unicode_to_byte,
        special_tokens: HashMap::new(),
        pre: PreTokenizer::Qwen2,
        special_by_first: HashMap::new(),
        bos_token: 0,
        eos_token: 0,
        im_start: 0,
        im_end: 0,
    };
    fill_byte_tokens(&mut t);
    rebuild_special_index(&mut t);
    t
}

#[test]
fn decode_bytes_reverses_byte_encoding() {
    let t = test_tokenizer();
    assert_eq!(t.decode_bytes(&[0]), b"hello");
    assert_eq!(t.decode_bytes(&[0, 1]), b"hello world");
    assert_eq!(t.decode(&[0, 1]), "hello world");
}

#[test]
fn decode_bytes_keeps_multibyte_bytes() {
    let t = test_tokenizer();
    let bytes = t.decode_bytes(&[2]);
    assert_eq!(bytes, vec![0xE4, 0xB8, 0xAD]); // bytes of U+4E2D
    assert_eq!(t.decode(&[2]), "中");
    assert!(!t.decode(&[2]).contains('\u{FFFD}'));
}

#[test]
fn decode_out_of_range_id_is_skipped() {
    let t = test_tokenizer();
    assert_eq!(t.decode_bytes(&[99]), Vec::<u8>::new());
    assert_eq!(t.decode(&[0, 99, 1]), "hello world");
}

#[test]
fn complete_utf8_prefix_len_holds_incomplete_trailing() {
    // U+4E2D = E4 B8 AD
    let full = [0xE4u8, 0xB8, 0xAD];
    assert_eq!(
        complete_utf8_prefix_len(&full[..1]),
        0,
        "1 of 3 bytes: incomplete"
    );
    assert_eq!(
        complete_utf8_prefix_len(&full[..2]),
        0,
        "2 of 3 bytes: incomplete"
    );
    assert_eq!(
        complete_utf8_prefix_len(&full[..3]),
        3,
        "all 3 bytes: complete"
    );

    let mixed = [b'a', 0xE4, 0xB8, 0xAD, b'b'];
    assert_eq!(
        complete_utf8_prefix_len(&mixed[..3]),
        1,
        "a complete, U+4E2D incomplete"
    );
    assert_eq!(
        complete_utf8_prefix_len(&mixed[..4]),
        4,
        "a + U+4E2D complete"
    );
    assert_eq!(complete_utf8_prefix_len(&mixed[..5]), 5);
    assert_eq!(complete_utf8_prefix_len(b"abc"), 3, "pure ASCII");
    assert_eq!(complete_utf8_prefix_len(&[]), 0);
}

/// Builds a tokenizer whose special-token table mimics a DeepSeek-R1 style
/// model: fullwidth-bar tokens that the GPT-2 pre-tokenizer regex would
/// otherwise split apart (regression for the R1 template, see README).
fn r1_style_tokenizer() -> Tokenizer {
    let mut t = test_tokenizer();
    t.special_tokens.insert("<｜User｜>".into(), 151644);
    t.special_tokens.insert("<｜Assistant｜>".into(), 151645);
    t.special_tokens.insert("<think>".into(), 151648);
    t.special_tokens
        .insert("<｜end▁of▁sentence｜>".into(), 151643);
    rebuild_special_index(&mut t);
    // Populate the vocab so BPE finds "What" and " is" etc. as whole tokens
    // (the pieces the regex would produce must NOT reassemble the specials).
    // Keys must be byte-encoded like bpe_encode expects ("Ġ" = space, "Ċ" = \n).
    // The merge ranks matter: the encoder always starts from single
    // characters (no "whole piece is in the vocab" shortcut), so a synthetic
    // token that is a whole word must be reachable through merges, exactly
    // like a real byte-level BPE vocabulary.
    t.vocab.insert("What".into(), 3838);
    t.vocab.insert(byte_encode(" is", &t.byte_to_unicode), 374);
    t.vocab.insert(byte_encode(" ", &t.byte_to_unicode), 220);
    t.vocab.insert(byte_encode("2", &t.byte_to_unicode), 17);
    t.vocab.insert(byte_encode("+", &t.byte_to_unicode), 10);
    t.vocab.insert(byte_encode("?", &t.byte_to_unicode), 30);
    t.vocab.insert(byte_encode("\n", &t.byte_to_unicode), 198);
    let mut rank = 0usize;
    let mut add_merges = |t: &mut Tokenizer, word: &str| {
        let mut left = String::new();
        for c in word.chars() {
            let right = c.to_string();
            if !left.is_empty() {
                t.merges.insert((left.clone(), right.clone()), rank);
                rank += 1;
            }
            left.push(c);
        }
    };
    for word in ["What".to_string(), byte_encode(" is", &t.byte_to_unicode)] {
        add_merges(&mut t, &word);
    }
    t
}

#[test]
fn special_tokens_match_as_single_ids_before_bpe() {
    let t = r1_style_tokenizer();
    let ids = t.encode("<｜User｜>What is 2+2?<｜Assistant｜><think>\n");
    // llama.cpp reference for the same string (llama-tokenize): 151644 3838 374 220 17 10 17 30 151645 151648 198
    assert_eq!(
        ids,
        vec![151644, 3838, 374, 220, 17, 10, 17, 30, 151645, 151648, 198],
        "special tokens must survive as single IDs, never split by BPE"
    );
}

#[test]
fn special_token_earliest_position_wins() {
    let t = r1_style_tokenizer();
    // text before a special token is BPE-encoded; specials in the middle match
    let ids = t.encode("abc<think>def<｜User｜>ghi");
    let abc: Vec<u32> = t.encode_bpe("abc");
    let def: Vec<u32> = t.encode_bpe("def");
    let ghi: Vec<u32> = t.encode_bpe("ghi");
    let mut expect = abc.clone();
    expect.push(151648);
    expect.extend(def);
    expect.push(151644);
    expect.extend(ghi);
    assert_eq!(ids, expect);
}

#[test]
fn longest_special_token_wins_at_same_position() {
    let mut t = r1_style_tokenizer();
    // Both <think> and <think▁begin｜> start at the same position; the longer
    // one must win even though <think> was inserted first.
    t.special_tokens.insert("<think▁begin｜>".into(), 151649);
    rebuild_special_index(&mut t);
    let ids = t.encode("<think▁begin｜>");
    assert_eq!(ids, vec![151649]);
}

#[test]
fn chatml_specials_still_match_via_fallback() {
    // Even without GGUF type 3/4 info (special_tokens empty), the hardcoded
    // <|im_start|>/<|im_end|> fallback must keep working.
    let mut t = test_tokenizer();
    t.vocab.insert("hi".into(), 42);
    t.merges.insert(("h".into(), "i".into()), 0);
    let ids = t.encode("<|im_start|>hi<|im_end|>");
    // im_start/im_end fall back to 0 (unknown) in this synthetic tokenizer
    assert_eq!(ids, vec![0, 42, 0]);
}

// === F7 (#50) pre-tokenizer rules and byte fallback ======================

/// Reference splits produced by CPython `regex` from the model's own
/// `tokenizer.json` `Split` pattern (see the fixture provenance).
const SPLIT_FIXTURES: &[(&str, &str)] = &[
    (
        "qwen2",
        include_str!("../../tests/fixtures/tokenizer/split_qwen2.json"),
    ),
    (
        "qwen35",
        include_str!("../../tests/fixtures/tokenizer/split_qwen35.json"),
    ),
];

fn pre_of(name: &str) -> PreTokenizer {
    match name {
        "qwen2" => PreTokenizer::Qwen2,
        "qwen35" => PreTokenizer::Qwen35,
        other => panic!("unknown pre-tokenizer {other}"),
    }
}

#[test]
fn pre_tokenizer_split_matches_the_reference() {
    for (name, raw) in SPLIT_FIXTURES {
        let fx: serde_json::Value = serde_json::from_str(raw).expect("split fixture");
        assert_eq!(fx["pre"].as_str().unwrap(), *name);
        assert!(
            fx["provenance"]["reference"]
                .as_str()
                .unwrap()
                .contains("regex"),
            "{name}: the reference engine must be named"
        );
        let pre = pre_of(name);
        let entries = fx["entries"].as_array().unwrap();
        assert!(entries.len() >= 45, "{name}: fixture shrank");
        for e in entries {
            let text = e["text"].as_str().unwrap();
            let want: Vec<&str> = e["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap())
                .collect();
            let got = pre.split(text);
            assert_eq!(
                got, want,
                "{name}: pre-tokenization of {text:?} differs from the reference"
            );
            assert_eq!(
                got.concat(),
                text,
                "{name}: pieces must tile the input for {text:?}"
            );
        }
    }
}

#[test]
fn pre_tokenizer_selection_refuses_unknown_and_missing() {
    assert_eq!(
        PreTokenizer::from_gguf(Some("qwen2")).unwrap(),
        PreTokenizer::Qwen2
    );
    assert_eq!(
        PreTokenizer::from_gguf(Some("deepseek-r1-qwen")).unwrap(),
        PreTokenizer::Qwen2
    );
    assert_eq!(
        PreTokenizer::from_gguf(Some("qwen35")).unwrap(),
        PreTokenizer::Qwen35
    );
    let err = PreTokenizer::from_gguf(Some("llama3")).unwrap_err();
    assert!(
        err.contains("llama3"),
        "the refusal must name the value: {err}"
    );
    let err = PreTokenizer::from_gguf(None).unwrap_err();
    assert!(
        err.contains("missing"),
        "a missing key must be named as such: {err}"
    );
    // Shown by `cargo test -- --nocapture`; this is the user-facing text.
    eprintln!("refusal (unsupported pre-tokenizer):\n{err}");
}

#[test]
fn byte_fallback_emits_one_token_per_byte() {
    let mut t = test_tokenizer();
    // A merge rank exists for "a"+"b" but the merged token does not: the
    // encoder must emit one token per byte. (`Tokenizer::load` guarantees
    // only the 256 byte tokens, not that every merge result is a token.)
    t.vocab.clear();
    fill_byte_tokens(&mut t);
    t.merges.clear();
    t.merges.insert(("a".into(), "b".into()), 0);
    let ids = t.encode_bpe("ab");
    let want: Vec<u32> = "ab".bytes().map(|b| 1000 + b as u32).collect();
    assert_eq!(ids, want);
    assert!(
        !ids.contains(&0),
        "byte fallback must not emit id 0: {ids:?}"
    );
}

// === F7 (#50) token-id equality against the reference tokenizer =========

const ID_FIXTURES: &[(&str, &str, &str)] = &[
    (
        "qwen2.5-0.5b-instruct",
        "~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_qwen2.5-0.5b-instruct.json"),
    ),
    (
        "qwen2.5-7b-instruct",
        "~/.cache/minfer/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_qwen2.5-7b-instruct.json"),
    ),
    (
        "qwen2.5-14b-instruct",
        "~/.cache/minfer/models/hf/Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_qwen2.5-14b-instruct.json"),
    ),
    (
        "qwen3-0.6b",
        "~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_qwen3-0.6b.json"),
    ),
    (
        "qwen3.5-0.8b",
        "~/.cache/minfer/models/hf/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_qwen3.5-0.8b.json"),
    ),
];

fn expand_home(path: &str) -> std::path::PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => {
            let home = std::env::var_os("HOME").expect("HOME");
            std::path::PathBuf::from(home).join(rest)
        }
        None => std::path::PathBuf::from(path),
    }
}

/// The ticket's tokenizer acceptance: byte-for-byte token-id equality with
/// the reference over a corpus that exercises the generalized rules. Ignored
/// because it needs the cached GGUFs (the committed fixtures, not the models,
/// are the reference); run serially:
///   cargo test --release --bin minfer -- --ignored --test-threads=1
///
/// The reference per model is named in the fixture: transformers 5.17.0 for
/// Qwen2.5/Qwen3 (whose GGUF tokenizer and `tokenizer.json` agree), and
/// llama.cpp run on the same GGUF artifact for Qwen3.5 (whose published
/// `tokenizer.json` is a different revision than the GGUF).
#[test]
#[ignore = "requires the cached GGUF models (~/.cache/minfer/models)"]
fn token_ids_match_the_reference() {
    for (slug, path, raw) in ID_FIXTURES {
        let p = expand_home(path);
        if !p.exists() {
            eprintln!("{slug}: {} not cached; skipping", p.display());
            continue;
        }
        let gguf = crate::gguf::load_gguf_model(&p).expect("parse GGUF");
        let tok = Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
        let fx: serde_json::Value = serde_json::from_str(raw).expect("ids fixture");
        assert_eq!(
            fx["pre"].as_str().unwrap(),
            tok.pre.gguf_name(),
            "{slug}: fixture pre-tokenizer does not match the GGUF"
        );
        let entries = fx["entries"].as_array().unwrap();
        assert!(entries.len() >= 45, "{slug}: fixture shrank");
        for e in entries {
            let text = e["text"].as_str().unwrap();
            let want: Vec<u32> = e["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = tok.encode(text);
            assert_eq!(
                got, want,
                "{slug}: token ids differ from transformers for {text:?}"
            );
        }
        // The special/added-token requirement, asserted directly: a special
        // token inside text is one id, never BPE-split.
        let ids = tok.encode("<|im_start|>hi<|im_end|>");
        assert_eq!(
            (ids.first().copied(), ids.last().copied()),
            (Some(tok.im_start), Some(tok.im_end)),
            "{slug}: special tokens must survive as single ids: {ids:?}"
        );
        eprintln!(
            "{slug}: {} corpus entries + special tokens match the reference byte for byte",
            entries.len()
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `tokenizer.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl PreTokenizer {
    /// The `tokenizer.ggml.pre` spelling this rule was selected from.
    ///
    /// Test-only (#239): driven by
    /// `tokenizer::tests::token_ids_match_the_reference`.
    pub fn gguf_name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
        }
    }
}

impl Tokenizer {
    /// Decode token IDs to text.
    ///
    /// Lossy: an incomplete multi-byte sequence at the end becomes U+FFFD.
    /// Streaming paths use [`Tokenizer::decode_bytes`] plus
    /// [`complete_utf8_prefix_len`] holdback instead.
    ///
    /// Test-only (#239): driven by `tokenizer::tests::{decode_bytes_reverses_byte_encoding,
    /// decode_bytes_keeps_multibyte_bytes, decode_out_of_range_id_is_skipped}`.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}
