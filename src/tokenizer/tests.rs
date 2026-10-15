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

// === #132 S1: NFC normalization of the input =============================

/// `Tokenizer::encode` normalizes to NFC, so a decomposed input reaches the
/// same token as its composed form. The composed character is given a real
/// vocabulary entry (a merge plus the two byte-encoded characters) so the
/// assertion is about one *specific* id, and the unnormalized internal path is
/// asserted to disagree — otherwise the test would pass for the wrong reason.
#[test]
fn encode_normalizes_decomposed_input_to_nfc() {
    let mut t = test_tokenizer();
    let encoded = byte_encode("é", &t.byte_to_unicode);
    let chars: Vec<char> = encoded.chars().collect();
    assert_eq!(chars.len(), 2, "é is two UTF-8 bytes");
    t.merges
        .insert((chars[0].to_string(), chars[1].to_string()), 0);
    t.vocab.insert(encoded, 7);
    t.id_to_token.push("é".into());
    t.id_to_score.push(0.0);
    t.id_to_type.push(1);

    assert_eq!(t.encode("é"), vec![7]);
    assert_eq!(
        t.encode("e\u{0301}"),
        vec![7],
        "NFC must compose e + U+0301"
    );
    assert_ne!(
        t.encode_bpe("e\u{0301}"),
        vec![7],
        "the unnormalized path must disagree, or this test proves nothing"
    );
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
        // #132 S1: the declared normalizer, on a real vocabulary. No cached GGUF
        // carries a decomposed corpus entry, so the assertion is the *property*
        // (a decomposed input encodes like its composed form); CPython is the
        // reference for the normalization itself (`nfc::tests`), and the
        // unnormalized path is asserted to disagree — otherwise this would pass
        // for the wrong reason.
        let composed = tok.encode("café");
        let decomposed = tok.encode("cafe\u{0301}");
        let raw_decomposed = tok.encode_bpe("cafe\u{0301}");
        assert_eq!(
            decomposed, composed,
            "{slug}: NFC is not applied to `Tokenizer::encode`"
        );
        assert_eq!(
            tok.encode_raw("cafe\u{0301}"),
            tok.encode_raw("café"),
            "{slug}: NFC is not applied to `Tokenizer::encode_raw`"
        );
        assert_ne!(
            raw_decomposed, composed,
            "{slug}: a real vocabulary must split the decomposed form differently"
        );
        eprintln!(
            "{slug}: {} corpus entries + special tokens match the reference byte for byte; \
             NFC: café -> {composed:?}, decomposed -> {decomposed:?} (raw {raw_decomposed:?})",
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

// === #118: DRY sequence breakers as strings =================================
//
// The breaKer resolver's contract: llama.cpp's `get_overlapping_token_sequences`
// with the two branches (whole token → empty tail; breaker starting inside a
// token and continuing past it), byte-based matching, raw-BPE tails, and the
// refusals minfer adds where llama.cpp truncates or skips.

/// The synthetic vocabulary the breaker tests use: byte-level complete (via
/// [`Tokenizer::byte_level_for_tests`]) plus four entries that let a breaker
/// overlap a token boundary.
fn breaker_tokenizer() -> Tokenizer {
    let mut t = Tokenizer::byte_level_for_tests();
    // A dedicated single-character token (id 300), like a real vocabulary's
    // punctuation token: one breaker string then resolves through two tokens.
    t.vocab.insert(":".to_string(), 300);
    t.id_to_token[300] = ":".to_string();
    // "ab" (id 400): makes "abc" start inside a token and continue past it.
    t.vocab.insert("ab".to_string(), 400);
    t.id_to_token[400] = "ab".to_string();
    t.merges.insert(("a".to_string(), "b".to_string()), 0);
    // "aaa" (id 600): one token in which the breaker "aaaa" starts three times,
    // each leaving a different remainder (the "matching tail" list).
    t.vocab.insert("aaa".to_string(), 600);
    t.id_to_token[600] = "aaa".to_string();
    t.merges.insert(("a".to_string(), "a".to_string()), 1);
    t.merges.insert(("aa".to_string(), "a".to_string()), 2);
    // "ac" (id 800): starts with the breaker's first byte but diverges at the
    // second, so it is *not* an overlap. It is the only token that distinguishes
    // "the match must reach the token's end" from "any position counts", which
    // is why it is here.
    t.vocab.insert("ac".to_string(), 800);
    t.id_to_token[800] = "ac".to_string();
    t.merges.insert(("a".to_string(), "c".to_string()), 4);
    t
}

#[test]
fn dry_breaker_whole_token_resolves_to_a_length_one_sequence() {
    let t = breaker_tokenizer();
    // "\n" is representable only by its byte token, so the resolved set is
    // exactly the token-id spelling — and it must be *length 1*: `apply_dry`'s
    // single-token-breaker exemption reads `b.len() == 1`, and an appended
    // empty tail would silently disable it.
    assert_eq!(
        t.overlapping_token_sequences("\n").unwrap(),
        vec![vec![1010u32]]
    );
    assert_eq!(
        t.resolve_sequence_breakers(&["\n".to_string()]).unwrap(),
        vec![vec![1010u32]]
    );

    // ':' exists both as a dedicated token and as the byte token: llama.cpp
    // reports every overlapping sequence, and each of them is still length 1.
    let seqs = t.overlapping_token_sequences(":").unwrap();
    assert!(seqs.iter().all(|s| s.len() == 1), "{seqs:?}");
    assert!(seqs.contains(&vec![300u32]), "{seqs:?}");
    assert!(seqs.contains(&vec![1058u32]), "{seqs:?}");
}

#[test]
fn dry_breaker_requires_the_match_to_reach_the_token_end() {
    let t = breaker_tokenizer();
    // "ac" (800) starts with the breaker's first byte and diverges at its
    // second, so it is not an overlap: the tail would have to start immediately
    // after a match that never completed. Every other token that starts this
    // breaker does match, and the 'a' byte token's tail is the whole "bX" rest.
    let mut got = t.overlapping_token_sequences("abX").unwrap();
    got.sort();
    let mut want = vec![
        vec![400u32, 1088],     // "ab" holds "ab", tail "X"
        vec![600, 1098, 1088],  // "aaa" ends at the breaker's first byte
        vec![1097, 1098, 1088], // 'a' byte token, tail "bX"
    ];
    want.sort();
    assert_eq!(got, want);
    assert!(
        !got.iter().any(|s| s.first() == Some(&800u32)),
        "a token that diverges mid-breaker must not be a head: {got:?}"
    );
}

#[test]
fn dry_breaker_contained_in_one_token_suppresses_its_overlapping_tails() {
    let mut t = breaker_tokenizer();
    // "aba" (id 700) *contains* "ab", so llama.cpp's first branch applies: the
    // token alone is the whole occurrence. The second position (2) would leave a
    // non-empty tail ("b") — dropping the first branch adds [700, 1098] here,
    // which is exactly the mutation this test pins.
    t.vocab.insert("aba".to_string(), 700);
    t.id_to_token[700] = "aba".to_string();
    t.merges.insert(("ab".to_string(), "a".to_string()), 3);

    let seqs = t.overlapping_token_sequences("ab").unwrap();
    assert!(seqs.contains(&vec![700u32]), "{seqs:?}");
    assert!(!seqs.contains(&vec![700u32, 1098]), "{seqs:?}");
    assert_eq!(
        seqs.iter().filter(|s| s.first() == Some(&700u32)).count(),
        1,
        "{seqs:?}"
    );
}

#[test]
fn dry_breaker_crossing_a_token_boundary_carries_the_remainder_as_its_tail() {
    let t = breaker_tokenizer();
    let mut got = t.overlapping_token_sequences("abc").unwrap();
    got.sort();
    let mut want = vec![
        // "ab" (400) holds "ab", so the tail is the rest of the breaker, "c".
        vec![400u32, 1099],
        // "aaa" (600) ends with the breaker's first byte; same tail as below.
        vec![600, 1098, 1099],
        // The 'a' byte token holds only "a", so the tail is "bc".
        vec![1097, 1098, 1099],
    ];
    want.sort();
    assert_eq!(got, want);
}

#[test]
fn dry_breaker_reports_every_tail_for_one_head_token() {
    let t = breaker_tokenizer();
    // "aaaa" starts at all three positions of the token "aaa" (600), and each
    // position leaves a different remainder: "a", "aa", "aaa". One head, three
    // sequences.
    let mut got = t.overlapping_token_sequences("aaaa").unwrap();
    got.sort();
    let mut want = vec![
        vec![600u32, 600],     // tail "aaa" tokenizes back to 600
        vec![600, 1097],       // tail "a"
        vec![600, 1097, 1097], // tail "aa": no "aa" token, so per byte
        vec![1097, 600],       // head 'a', tail "aaa"
    ];
    want.sort();
    assert_eq!(got, want);
}

#[test]
fn dry_breaker_tail_does_not_absorb_a_special_token() {
    let mut t = breaker_tokenizer();
    t.special_tokens.insert("<|im_end|>".to_string(), 7);
    t.id_to_token[7] = "<|im_end|>".to_string();
    t.im_end = 7;
    rebuild_special_index(&mut t);

    // The special-token pre-scan does collapse the text to the special id ...
    assert_eq!(t.encode("<|im_end|>"), vec![7u32]);
    // ... and the raw path a breaker tail uses does not, so a tail can never
    // name a token the sampler could not have produced as ordinary text.
    let raw = t.encode_raw("<|im_end|>");
    assert_ne!(raw, vec![7u32]);
    assert!(!raw.contains(&7u32), "{raw:?}");
    // ... and the raw path really does cover the whole text, as ordinary
    // byte-level tokens that decode back to it (not an empty vector).
    assert_eq!(t.decode_bytes(&raw), b"<|im_end|>");

    let seqs = t.overlapping_token_sequences("a<|im_end|>").unwrap();
    assert!(seqs.iter().all(|s| !s.contains(&7u32)), "{seqs:?}");
    let head_a = seqs
        .iter()
        .find(|s| s.first() == Some(&1097u32))
        .expect("the 'a' byte token matches");
    assert_eq!(&head_a[1..], &raw[..], "{head_a:?}");
}

#[test]
fn dry_breaker_refuses_what_it_cannot_carry() {
    let t = breaker_tokenizer();

    let err = t.overlapping_token_sequences("").unwrap_err();
    assert!(err.contains("must not be empty"), "{err}");

    // llama.cpp truncates to MAX_CHAR_LEN; minfer names the value and refuses.
    let long = "a".repeat(MAX_BREAKER_CHARS + 1);
    let err = t.overlapping_token_sequences(&long).unwrap_err();
    assert!(err.contains("41") && err.contains("40"), "{err}");

    // Same for a tail over MAX_SEQ_LEN: 'a' + 21 byte tokens needs a 21-token
    // tail after the 'a' byte token.
    let long_tail = format!("a{}", "b".repeat(MAX_BREAKER_TAIL_TOKENS + 1));
    let err = t.overlapping_token_sequences(&long_tail).unwrap_err();
    assert!(
        err.contains(&format!("{MAX_BREAKER_TAIL_TOKENS}-token limit")),
        "{err}"
    );

    // A breaker that matches nothing is refused, never ignored. Every byte token
    // exists in a real vocabulary, so this is a defensive gate — it fires on a
    // vocabulary that is not byte-level complete.
    let err = Tokenizer::empty()
        .overlapping_token_sequences("x")
        .unwrap_err();
    assert!(err.contains("overlaps no token"), "{err}");
}

#[test]
fn dry_breaker_string_form_never_reinterprets_digits_as_token_ids() {
    let t = breaker_tokenizer();
    // "1" is a breaker *string*: it is matched as text, so it resolves to the
    // byte token for '1' (1049) — never to the token whose *id* is 1.
    let seqs = t.resolve_sequence_breakers(&["1".to_string()]).unwrap();
    assert!(seqs.contains(&vec![1049u32]), "{seqs:?}");
    assert!(!seqs.contains(&vec![1u32]), "{seqs:?}");
}

#[test]
fn dry_breaker_strings_deduplicate_and_keep_every_distinct_sequence() {
    let t = breaker_tokenizer();
    let one = t.resolve_sequence_breakers(&["\n".to_string()]).unwrap();
    let twice = t
        .resolve_sequence_breakers(&["\n".to_string(), "\n".to_string()])
        .unwrap();
    assert_eq!(one, twice, "a repeated breaker adds no sequence");

    // Two different breakers that overlap each other keep both of their
    // sequences (this is what "overlapping" means).
    let both = t
        .resolve_sequence_breakers(&["ab".to_string(), ":".to_string()])
        .unwrap();
    assert!(both.contains(&vec![1097u32, 1098]), "{both:?}");
    assert!(both.contains(&vec![300u32]), "{both:?}");
}
