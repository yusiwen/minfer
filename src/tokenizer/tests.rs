//! `#[cfg(test)] mod tests` for `src/tokenizer.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::gguf::GgufKv;

/// Gives a synthetic tokenizer the 256 byte tokens every real byte-level
/// BPE vocabulary has (and which `Tokenizer::load` verifies). Without them
/// the byte fallback has nothing to emit, which is a load-time refusal in
/// production and an invariant violation here.
fn fill_byte_tokens(t: &mut Tokenizer) {
    for (b, c) in t.byte_to_unicode.clone() {
        t.vocab.entry(c.to_string()).or_insert(1000 + b as u32);
    }
}

/// Rebuilds `special_by_first` the way `load()` does: exactly the declared
/// special tokens, nothing inferred from a name (#132 S2).
fn rebuild_special_index(t: &mut Tokenizer) {
    for (pat, id) in t.special_tokens.clone() {
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
        model: TokenizerModel::Gpt2,
        pre: Some(PreTokenizer::Qwen2),
        add_space_prefix: false,
        spm_byte_to_id: HashMap::new(),
        normalizer: NormalizerOptions::default(),
        max_token_len: 0,
        unk_token: 0,
        ugm: UgmData::default(),
        remove_extra_whitespaces: false,
        special_by_first: HashMap::new(),
        bos_token: 0,
        eos_token: 0,
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
fn a_chat_token_the_file_does_not_declare_is_not_invented() {
    // #132 S2: the `<|im_start|>` / `<|im_end|>` hardcode is gone. With an
    // empty declared set those two names are ordinary text — they are never
    // mapped to an id the vocabulary does not have (the old fallback produced
    // id 0 for `<|im_start|>` and the EOS id for `<|im_end|>`).
    let mut t = test_tokenizer();
    t.vocab.insert("hi".into(), 42);
    t.merges.insert(("h".into(), "i".into()), 0);
    let text = "<|im_start|>hi<|im_end|>";
    assert_eq!(t.encode(text), t.encode_bpe(text));
    assert!(
        t.encode(text).contains(&42),
        "the plain text still tokenizes"
    );
    assert!(!t.special_tokens.contains_key("<|im_start|>"));
    assert!(!t.special_tokens.contains_key("<|im_end|>"));
}

// === #132 S4: the rwkv engine ==============================================

/// An RVKV vocabulary: the entries are **escaped** (`\x41` stands for the byte
/// `A`, `\n` for a newline), which is exactly what the trie has to decode before
/// it can match, and `hello`/`he`/`l` make the greedy longest match observable.
fn rwkv_tokenizer() -> Tokenizer {
    let mut t = Tokenizer::empty();
    t.model = TokenizerModel::Rwkv;
    t.pre = None;
    t.unk_token = 0;
    for (text, ty) in [
        ("<unk>", 2),
        ("hello", 1),
        ("he", 1),
        ("l", 1),
        ("\\x41", 1),
        ("\\n", 1),
    ] {
        let id = t.id_to_token.len() as u32;
        t.id_to_token.push(text.to_string());
        t.id_to_type.push(ty);
        t.vocab.insert(text.to_string(), id);
    }
    for (id, text) in t.id_to_token.clone().iter().enumerate() {
        let bytes = unescape_rwkv_token(text);
        t.ugm.trie.insert_bytes(&bytes, id as u32);
    }
    t
}

#[test]
fn rwkv_takes_the_longest_match_and_unescapes_the_vocabulary() {
    let t = rwkv_tokenizer();
    assert_eq!(t.encode("hello"), vec![t.vocab["hello"]]);
    // Not a whole word: `hello` does not match, but `he` then `l` do.
    assert_eq!(t.encode("hel"), vec![t.vocab["he"], t.vocab["l"]]);
    // The escaped entries stand for bytes, so a literal `A` and a newline match.
    assert_eq!(t.encode("A"), vec![t.vocab["\\x41"]]);
    assert_eq!(t.encode("\n"), vec![t.vocab["\\n"]]);
    // Decode is the same unescaping: `\x41` is one byte, not four characters.
    assert_eq!(t.decode_bytes(&[t.vocab["\\x41"]]), b"A");
    assert_eq!(t.decode_bytes(&[t.vocab["hello"]]), b"hello");
}

#[test]
fn rwkv_emits_one_unknown_per_unmatched_byte() {
    let t = rwkv_tokenizer();
    // `z` starts no entry, so it is one unknown token — and the next byte still
    // matches, so a miss is one unknown per byte, never a dropped byte.
    assert_eq!(t.encode("z"), vec![t.unk_token]);
    assert_eq!(t.encode("zA"), vec![t.unk_token, t.vocab["\\x41"]]);
}

// === #132 S4: the gemma4 engine ============================================

/// A gemma4 vocabulary: the declared specials, the `<0xXX>` byte tokens the
/// fallback needs, and the pieces the merge list builds — including `\n\n`, which
/// exists only as a whole (no `\n` entry), so the newline shortcut is observable.
fn gemma4_tokenizer() -> Tokenizer {
    let mut t = Tokenizer::empty();
    t.model = TokenizerModel::Gemma4;
    t.pre = Some(PreTokenizer::Gemma4);

    let mut push = |t: &mut Tokenizer, text: &str, ty: i32| -> u32 {
        let id = t.id_to_token.len() as u32;
        t.id_to_token.push(text.to_string());
        t.id_to_type.push(ty);
        t.vocab.insert(text.to_string(), id);
        id
    };
    push(&mut t, "<pad>", 3);
    push(&mut t, "<eos>", 3);
    for b in 0u8..=255 {
        let id = push(&mut t, &format!("<0x{b:02X}>"), 6);
        t.spm_byte_to_id.insert(b, id);
    }
    for text in ["a", "b", "ab", "\u{2581}a", "\u{2581}", "\n\n"] {
        push(&mut t, text, 1);
    }
    t.merges.insert(("a".to_string(), "b".to_string()), 0);
    t.merges
        .insert(("\u{2581}".to_string(), "a".to_string()), 1);
    t
}

#[test]
fn gemma4_splits_on_newlines_and_does_not_byte_encode() {
    let t = gemma4_tokenizer();
    let (ab, space_a) = (t.vocab["ab"], t.vocab["\u{2581}a"]);
    // A space becomes `▁` *before* BPE, and the merge runs on raw UTF-8: the
    // `▁a` entry is reachable only because no byte-to-unicode step happened.
    assert_eq!(t.encode("ab a"), vec![ab, space_a]);
    // The rule splits on newline runs, so `\n\n` is its own piece — emitted whole
    // because it is a vocabulary entry (the gemma4 shortcut).
    let two_newlines = t.vocab["\n\n"];
    assert_eq!(t.encode("ab\n\na"), vec![ab, two_newlines, t.vocab["a"]]);
}

#[test]
fn gemma4_falls_back_to_byte_tokens() {
    let t = gemma4_tokenizer();
    // `\n` alone is not a vocabulary entry, so the piece falls back byte by byte —
    // never to the unknown token and never dropped.
    assert_eq!(t.encode("\n"), vec![t.spm_byte_to_id[&b'\n']]);
    assert_eq!(t.encode("z"), vec![t.spm_byte_to_id[&b'z']]);
    // Decode is the inverse: `▁` is a space, a byte token is its byte.
    assert_eq!(t.decode_bytes(&[t.vocab["\u{2581}a"]]), b" a");
    assert_eq!(t.decode_bytes(&[t.spm_byte_to_id[&b'z']]), b"z");
}

// === #132 S4: the unigram (`model = t5`) engine ============================

/// A unigram vocabulary with the pieces the Viterbi needs to prefer one
/// segmentation over another: `ab` scores above `a` + `b`, `▁ab` above `▁a` +
/// `b`, and the scores bound the unknown transition (`min_score` is `▁`'s -5, so
/// an unknown costs -15).
///
/// Ids: 0 `<pad>`, 1 `</s>`, 2 `<unk>` (the unknown role), then the pieces, then
/// the two capitals the charsmap test replaces one with.
fn ugm_tokenizer() -> Tokenizer {
    let mut t = Tokenizer::empty();
    t.model = TokenizerModel::Ugm;
    t.pre = None;
    t.unk_token = 2;
    t.add_space_prefix = false;
    t.remove_extra_whitespaces = false;

    let mut push = |t: &mut Tokenizer, text: &str, ty: i32, score: f32| -> u32 {
        let id = t.id_to_token.len() as u32;
        t.id_to_token.push(text.to_string());
        t.id_to_type.push(ty);
        t.id_to_score.push(score);
        t.vocab.insert(text.to_string(), id);
        id
    };
    push(&mut t, "<pad>", 3, 0.0);
    push(&mut t, "</s>", 3, 0.0);
    push(&mut t, "<unk>", 2, 0.0);
    for (text, score) in [
        ("\u{2581}", -5.0),
        ("a", -3.0),
        ("b", -3.0),
        ("ab", -1.0),
        ("\u{2581}a", -2.0),
        ("\u{2581}ab", -1.5),
        ("A", -1.0),
        ("B", -1.0),
    ] {
        push(&mut t, text, 1, score);
    }
    for (id, text) in t.id_to_token.clone().iter().enumerate() {
        let ty = t.id_to_type[id];
        if ty == 1 || ty == 4 || ty == 5 {
            t.ugm.trie.insert(text, id as u32);
        }
    }
    t.ugm.unknown_token_score = -5.0 - 10.0;
    t
}

#[test]
fn ugm_picks_the_highest_scoring_segmentation() {
    let t = ugm_tokenizer();
    // `ab` (-1) beats `a` + `b` (-6).
    assert_eq!(t.encode("ab"), vec![t.vocab["ab"]]);
    // With the prefix space, `▁ab` (-1.5) beats `▁a` + `b` (-5).
    let mut prefixed = ugm_tokenizer();
    prefixed.add_space_prefix = true;
    assert_eq!(prefixed.encode("ab"), vec![prefixed.vocab["\u{2581}ab"]]);
    // A code point nothing matches costs the unknown transition, and a run of
    // them is one unknown token, never one per character.
    assert_eq!(
        t.encode("aXXb"),
        vec![t.vocab["a"], t.unk_token, t.vocab["b"]]
    );
}

#[test]
fn ugm_applies_the_charsmap_prefix_replacement() {
    // A hand-built XCDA table with one mapping: `A` -> `B`. The layout is the one
    // `normalize_prefix` walks: BASE of the root, the child node (LCHECK = the
    // byte, LEAF set), and the node its BASE points at, holding the replacement's
    // offset in its low 31 bits.
    let mut t = ugm_tokenizer();
    assert_eq!(t.encode("A"), vec![t.vocab["A"]]);
    t.ugm.xcda = vec![0x0001_0000, 0x0000_0D41, 0];
    t.ugm.prefix_replacements = b"B\0".to_vec();
    assert_eq!(
        t.encode("A"),
        vec![t.vocab["B"]],
        "the charmap must rewrite `A` to `B` before the Viterbi runs"
    );
}

#[test]
fn ugm_whitespace_normalization_follows_the_file() {
    let mut t = ugm_tokenizer();
    t.add_space_prefix = true;
    // `add_space_prefix` prepends one `▁` to text that does not start with space.
    assert_eq!(t.ugm_normalize("ab"), "\u{2581}ab");
    // `remove_extra_whitespaces` collapses a run of spaces into one, and keeps it
    // attached to the following word.
    t.remove_extra_whitespaces = true;
    assert_eq!(t.ugm_normalize("a  b"), "\u{2581}a\u{2581}b");
    // Without it every space is escaped, so the run survives.
    t.remove_extra_whitespaces = false;
    assert_eq!(t.ugm_normalize("a  b"), "\u{2581}a\u{2581}\u{2581}b");
}

// === #132 S4: the WordPiece (`model = bert`) engine =========================

/// A WordPiece vocabulary shaped like a converted BERT file: the five declared
/// specials (types 2/3/4), `▁`-prefixed word entries (the converter rewrites a
/// BERT vocabulary this way), single letters for the fallback cases, and no
/// merges or scores — the engine reads neither.
///
/// Ids: 0 `[PAD]`, 1 `[UNK]`, 2 `[CLS]`, 3 `[SEP]`, 4 `[MASK]` (all type 3),
/// then the entries. `hex` exists so a word can be composed of two known pieces;
/// `zzz` exists only unprefixed, so `▁zzz` is unknown.
fn wpm_tokenizer(opts: NormalizerOptions) -> Tokenizer {
    let mut t = Tokenizer::empty();
    t.model = TokenizerModel::Bert;
    t.pre = None;
    t.normalizer = opts;
    t.unk_token = 1;

    let mut push = |t: &mut Tokenizer, text: &str, ty: i32| -> u32 {
        let id = t.id_to_token.len() as u32;
        t.id_to_token.push(text.to_string());
        t.id_to_type.push(ty);
        t.vocab.insert(text.to_string(), id);
        id
    };
    for text in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] {
        push(&mut t, text, 3);
    }
    for text in [
        "▁hello", "hello", "##", "▁world", "▁hex", "he", "x", "▁zzz", "zzz", "▁cafe", "▁naive",
        "▁", "!", "▁f", "f", "▁i", "i", "▁S", "▁café",
    ] {
        push(&mut t, text, 1);
    }
    for text in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] {
        let id = t.vocab[text];
        t.special_tokens.insert(text.to_string(), id);
    }
    t.special_by_first = special_index(&t.special_tokens);
    t.max_token_len = t.id_to_token.iter().map(String::len).max().unwrap_or(0);
    t
}

#[test]
fn wpm_segments_by_longest_match_and_unknowns_whole_words() {
    // An explicit no-normalizer configuration, so the words reach the vocabulary
    // exactly as written.
    let t = wpm_tokenizer(NormalizerOptions {
        lowercase: false,
        strip_accents: false,
    });
    let hello = t.vocab["▁hello"];
    let world = t.vocab["▁world"];
    let unk = t.unk_token;
    assert_eq!(t.encode("hello world"), vec![hello, world]);
    // A word with no fully matching segmentation is one `<unk>`, never a partial
    // sequence: `▁h` does not exist, so `hxyz` cannot be built.
    assert_eq!(t.encode("hxyz"), vec![unk]);
    // Punctuation becomes a word of its own. Its phantom-space form `▁!` is not
    // in this vocabulary, so the greedy match emits the bare `▁` entry and then
    // the punctuation entry — the same two pieces llama.cpp's session produces.
    let (space, bang) = (t.vocab["▁"], t.vocab["!"]);
    assert_eq!(t.encode("hello!"), vec![hello, space, bang]);
    // The declared specials still match through the scan before the engine runs.
    let (cls, sep) = (t.vocab["[CLS]"], t.vocab["[SEP]"]);
    assert_eq!(t.encode("[CLS] hello [SEP]"), vec![cls, hello, sep]);
}

#[test]
fn wpm_normalizer_options_follow_the_file() {
    // Lowercase on: the vocabulary's `▁hello` is reached from `Hello`.
    let lowered = wpm_tokenizer(NormalizerOptions {
        lowercase: true,
        strip_accents: false,
    });
    assert_eq!(lowered.encode("Hello"), vec![lowered.vocab["▁hello"]]);
    // Lowercase off (the default a converted BERT file leaves): `Hello` is not a
    // vocabulary entry, so the word is unknown.
    let plain = wpm_tokenizer(NormalizerOptions {
        lowercase: false,
        strip_accents: false,
    });
    assert_eq!(plain.encode("Hello"), vec![plain.unk_token]);
    // Strip accents decomposes with NFD and drops the marks, so `café` written
    // precomposed and decomposed both reach `▁cafe` — and only then does the
    // vocabulary's `▁cafe` entry match.
    let stripped = wpm_tokenizer(NormalizerOptions {
        lowercase: false,
        strip_accents: true,
    });
    let cafe = stripped.vocab["▁cafe"];
    assert_eq!(stripped.encode("cafe\u{0301}"), vec![cafe]);
    assert_eq!(stripped.encode("café"), vec![cafe]);
    // Without it the two forms are different words: the precomposed `café` has a
    // vocabulary entry of its own, and `cafe` + U+0301 does not.
    let plain = wpm_tokenizer(NormalizerOptions {
        lowercase: false,
        strip_accents: false,
    });
    assert_eq!(plain.encode("café"), vec![plain.vocab["▁café"]]);
    assert_eq!(plain.encode("cafe\u{0301}"), vec![plain.unk_token]);
}

#[test]
fn wpm_decode_unescapes_the_phantom_space_and_keeps_specials() {
    let t = wpm_tokenizer(NormalizerOptions::default());
    let hello = t.vocab["▁hello"];
    let sep = t.vocab["[SEP]"];
    assert_eq!(t.decode_bytes(&[hello]), b" hello");
    assert_eq!(t.decode_bytes(&[hello, sep]), b" hello[SEP]");
}

// === #132 S4: the SentencePiece (`model = llama`) engine ====================

/// A SentencePiece vocabulary with the three behaviours the engine has: a
/// score-ordered merge chain, the `▁` space prefix, and a `<0xXX>` byte fallback.
///
/// Ids: 0 `<unk>` (type 2), 1 `<s>`, 2 `</s>` (type 3), 3..=258 the 256 byte
/// tokens (type 6), then the normal entries — `▁` alone, the unprefixed chain
/// (`h`, `he`, `hel`, `hell`, `hello`) and the prefixed one. The prefixed entries
/// carry the higher scores, which is what makes the greedy merge build `▁hello`
/// rather than `hello`.
fn spm_tokenizer(add_space_prefix: bool) -> Tokenizer {
    let mut t = Tokenizer::empty();
    t.model = TokenizerModel::Spm;
    t.pre = None;
    t.add_space_prefix = add_space_prefix;

    let mut push = |t: &mut Tokenizer, text: &str, ty: i32, score: f32| -> u32 {
        let id = t.id_to_token.len() as u32;
        t.id_to_token.push(text.to_string());
        t.id_to_type.push(ty);
        t.id_to_score.push(score);
        t.vocab.insert(text.to_string(), id);
        id
    };
    push(&mut t, "<unk>", 2, 0.0);
    push(&mut t, "<s>", 3, 0.0);
    push(&mut t, "</s>", 3, 0.0);
    for b in 0u8..=255 {
        let id = push(&mut t, &format!("<0x{b:02X}>"), 6, 0.0);
        t.spm_byte_to_id.insert(b, id);
    }
    let space = push(&mut t, "▁", 1, 0.0);
    for (i, part) in ["h", "he", "hel", "hell", "hello"].iter().enumerate() {
        push(&mut t, part, 1, -1.0 - i as f32);
        push(&mut t, &format!("▁{part}"), 1, 1.0 + i as f32);
    }
    // `▁` is emitted on its own when nothing longer matches.
    t.vocab.insert("▁".to_string(), space);
    // `<s>` / `</s>` / `<unk>` are declared specials: the scan index is what
    // `encode` matches through, and the loader builds it the same way.
    for text in ["<unk>", "<s>", "</s>"] {
        let id = t.vocab[text];
        t.special_tokens.insert(text.to_string(), id);
    }
    t.special_by_first = special_index(&t.special_tokens);
    t
}

#[test]
fn the_tokenizer_model_is_a_dispatch() {
    assert_eq!(
        TokenizerModel::from_gguf(None).unwrap(),
        TokenizerModel::Gpt2
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("gpt2")).unwrap(),
        TokenizerModel::Gpt2
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("llama")).unwrap(),
        TokenizerModel::Spm
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("bert")).unwrap(),
        TokenizerModel::Bert
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("t5")).unwrap(),
        TokenizerModel::Ugm
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("gemma4")).unwrap(),
        TokenizerModel::Gemma4
    );
    assert_eq!(
        TokenizerModel::from_gguf(Some("rwkv")).unwrap(),
        TokenizerModel::Rwkv
    );
    // Every other engine is refused *by name* — an approximation with the wrong
    // engine would be worse than a refusal.
    for other in ["plamo2", "whitespace", "hybriddna", "none"] {
        let err = TokenizerModel::from_gguf(Some(other)).unwrap_err();
        assert!(
            err.contains(other),
            "the refusal must name the value: {err}"
        );
    }
}

#[test]
fn spm_merges_by_score_and_falls_back_to_byte_tokens() {
    let t = spm_tokenizer(true);
    // The whole prefixed word is reachable through the scored chain, so the
    // greedy merge builds it and emits one id.
    let hello = *t.vocab.get("▁hello").expect("the fixture has it");
    assert_eq!(t.encode("hello"), vec![hello]);
    // A character the vocabulary has no entry for falls back to its byte token,
    // one per byte — never to id 0 and never dropped.
    let space = *t.vocab.get("▁").expect("the fixture has it");
    let z = *t.spm_byte_to_id.get(&b'z').expect("byte token");
    assert_eq!(t.encode("z"), vec![space, z]);
    // Decode is the inverse for these: `▁` is a space and a byte token is a byte.
    assert_eq!(t.decode_bytes(&[hello]), b" hello");
    assert_eq!(t.decode_bytes(&[space, z]), b" z");
}

#[test]
fn spm_prefixes_the_first_segment_with_a_space() {
    let prefixed = spm_tokenizer(true);
    let plain = spm_tokenizer(false);
    assert_eq!(
        prefixed.encode("hello"),
        vec![*prefixed.vocab.get("▁hello").unwrap()]
    );
    assert_eq!(
        plain.encode("hello"),
        vec![*plain.vocab.get("hello").unwrap()],
        "with `add_space_prefix` off there is no leading space"
    );
    // A special token resets the "previous is special" state, so the segment
    // after it is prefixed again.
    assert_eq!(
        prefixed.encode("hello</s>hello"),
        vec![
            *prefixed.vocab.get("▁hello").unwrap(),
            2,
            *prefixed.vocab.get("▁hello").unwrap()
        ]
    );
}

// === #132 S3 / ADR-0044: the normalizer follows the pre-tokenizer rule =====

#[test]
fn the_normalizer_follows_the_pre_tokenizer_rule() {
    // The declaration is per rule, because that is the granularity the reference
    // `tokenizer.json` files and the GGUF's `pre` key share. `qwen2`/`qwen35`
    // declare NFC; `llama-bpe` declares `"normalizer": null`, and llama.cpp
    // applies none either — so minfer must not normalize there (#564, ADR-0044).
    assert!(PreTokenizer::Qwen2.applies_nfc());
    assert!(PreTokenizer::Qwen35.applies_nfc());
    assert!(!PreTokenizer::LlamaBpe.applies_nfc());
    assert!(!PreTokenizer::Lfm2.applies_nfc());
}

// === #132 S2: the special-token set is what the file declares ==============

/// A GGUF with just enough tokenizer metadata for `Tokenizer::load`: the 256
/// byte tokens (ids 0-255), then `specials` as `(text, token_type)`, a `pre`
/// rule, one merge rank, and `roles` as `(key, text)` role-id declarations.
fn tokenizer_gguf(specials: &[(&str, i32)], roles: &[(&str, &str)], pre: &str) -> GgufContext {
    let byte_to_unicode = build_byte_to_unicode();
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<i32> = Vec::new();
    for b in 0u8..=255 {
        tokens.push(byte_to_unicode[&b].to_string());
        types.push(1);
    }
    let mut id_of = HashMap::new();
    for (text, ty) in specials {
        id_of.insert((*text).to_string(), tokens.len() as u32);
        tokens.push((*text).to_string());
        types.push(*ty);
    }
    let mut kv = vec![
        GgufKv::new_string_array("tokenizer.ggml.tokens".into(), tokens),
        GgufKv::new_array(
            "tokenizer.ggml.token_type".into(),
            GgufType::Int32,
            types.iter().flat_map(|t| t.to_le_bytes()).collect(),
        ),
        GgufKv::new_string("tokenizer.ggml.model".into(), "gpt2".into()),
        GgufKv::new_string("tokenizer.ggml.pre".into(), pre.to_string()),
        GgufKv::new_string_array("tokenizer.ggml.merges".into(), vec!["a b".to_string()]),
    ];
    for (key, text) in roles {
        let id = *id_of
            .get(*text)
            .unwrap_or_else(|| panic!("role {key} names {text}, which is not in `specials`"));
        kv.push(GgufKv::new_u32((*key).to_string(), id));
    }
    GgufContext {
        version: 3,
        kv,
        info: Vec::new(),
        alignment: 32,
        offset: 0,
        size: 0,
    }
}

#[test]
fn special_tokens_are_the_declared_types_and_roles() {
    let gguf = tokenizer_gguf(
        &[
            ("<|ctrl|>", 3),
            ("<|unknown|>", 2),
            ("<|user|>", 4),
            ("<end_of_turn>", 1),
            ("<|ordinary|>", 1),
        ],
        &[("tokenizer.ggml.eot_token_id", "<end_of_turn>")],
        "qwen2",
    );
    let t = Tokenizer::load(&gguf).expect("tokenizer load");

    // A declared type is enough: UNKNOWN, CONTROL and USER_DEFINED all match.
    for text in ["<|ctrl|>", "<|unknown|>", "<|user|>", "<end_of_turn>"] {
        let id = *t
            .special_tokens
            .get(text)
            .unwrap_or_else(|| panic!("{text} is declared special but is missing"));
        assert_eq!(t.encode(text), vec![id], "{text} must be one id");
    }
    // A declared role is a declaration even when the type says NORMAL — and the
    // name here is not Qwen-shaped (`eot_token_id`, `<end_of_turn>`). Ids are
    // the 256 byte tokens followed by `specials` in order.
    let eot = *t.special_tokens.get("<end_of_turn>").expect("declared eot");
    assert_eq!((eot, t.encode("<end_of_turn>")), (259, vec![259]));
    // A token the file declares nowhere stays ordinary text.
    assert!(!t.special_tokens.contains_key("<|ordinary|>"));
    assert_eq!(t.encode("<|ordinary|>"), t.encode_bpe("<|ordinary|>"));
    // Three typed + one by role.
    assert_eq!(t.special_tokens.len(), 4);
}

#[test]
fn the_chatml_names_are_not_fabricated_for_a_vocabulary_without_them() {
    // Transcribed from `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` (Q4_K_M,
    // probed 2026-10-15 with the first 7 MB of the file): `pre =
    // deepseek-r1-qwen`, the fullwidth-bar control tokens are type 3 / 4, and
    // the vocabulary contains **neither** `<|im_start|>` nor `<|im_end|>` — the
    // two names the removed hardcode used to map to id 0 and to EOS.
    let gguf = tokenizer_gguf(
        &[
            ("<｜end▁of▁sentence｜>", 3),
            ("<｜User｜>", 3),
            ("<｜Assistant｜>", 3),
            ("<｜begin▁of▁sentence｜>", 3),
            ("<|EOT|>", 3),
            ("<think>", 4),
        ],
        &[
            ("tokenizer.ggml.bos_token_id", "<｜begin▁of▁sentence｜>"),
            ("tokenizer.ggml.eos_token_id", "<｜end▁of▁sentence｜>"),
            ("tokenizer.ggml.padding_token_id", "<｜end▁of▁sentence｜>"),
        ],
        "deepseek-r1-qwen",
    );
    let t = Tokenizer::load(&gguf).expect("tokenizer load");

    // The R1 template's tokens still match as single ids.
    for text in [
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "<｜end▁of▁sentence｜>",
    ] {
        let id = *t.special_tokens.get(text).expect("declared special");
        assert_eq!(t.encode(text), vec![id], "{text}");
    }
    // The two ChatML names are absent, so nothing may be invented for them.
    for text in ["<|im_start|>", "<|im_end|>"] {
        assert!(
            !t.special_tokens.contains_key(text),
            "{text} is not in this vocabulary and must not become special"
        );
        assert_eq!(
            t.encode(text),
            t.encode_bpe(text),
            "{text} must tokenize as the ordinary text it is"
        );
        assert!(!t.encode(text).contains(&t.eos_token));
    }
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

/// The counterpart of the test above, for a rule that declares **no**
/// normalizer: `llama-bpe`'s `tokenizer.json` says `"normalizer": null`, so a
/// decomposed input must reach BPE exactly as it arrived (#564, ADR-0044).
/// The composed character has a real vocabulary entry again, so a rule that
/// (wrongly) normalized would collapse `e` + U+0301 onto it.
#[test]
fn encode_leaves_decomposed_input_alone_when_the_rule_declares_no_normalizer() {
    let mut t = test_tokenizer();
    t.pre = Some(PreTokenizer::LlamaBpe);
    let encoded = byte_encode("é", &t.byte_to_unicode);
    let chars: Vec<char> = encoded.chars().collect();
    t.merges
        .insert((chars[0].to_string(), chars[1].to_string()), 0);
    t.vocab.insert(encoded, 7);
    t.id_to_token.push("é".into());
    t.id_to_score.push(0.0);
    t.id_to_type.push(1);

    assert_eq!(t.encode("é"), vec![7]);
    assert_ne!(
        t.encode("e\u{0301}"),
        vec![7],
        "a rule that declares no normalizer must not compose e + U+0301"
    );
    assert_eq!(
        t.encode("e\u{0301}"),
        t.encode_bpe("e\u{0301}"),
        "the normalized path must be the raw path when nothing is declared"
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
    (
        "llama-bpe",
        include_str!("../../tests/fixtures/tokenizer/split_llama-bpe.json"),
    ),
    (
        "lfm2",
        include_str!("../../tests/fixtures/tokenizer/split_lfm2.json"),
    ),
];

fn pre_of(name: &str) -> PreTokenizer {
    match name {
        "qwen2" => PreTokenizer::Qwen2,
        "qwen35" => PreTokenizer::Qwen35,
        "llama-bpe" => PreTokenizer::LlamaBpe,
        "lfm2" => PreTokenizer::Lfm2,
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
    assert_eq!(
        PreTokenizer::from_gguf(Some("llama-bpe")).unwrap(),
        PreTokenizer::LlamaBpe
    );
    assert_eq!(
        PreTokenizer::from_gguf(Some("llama3")).unwrap(),
        PreTokenizer::LlamaBpe
    );
    assert_eq!(
        PreTokenizer::from_gguf(Some("llama-v3")).unwrap(),
        PreTokenizer::LlamaBpe
    );
    assert_eq!(
        PreTokenizer::from_gguf(Some("lfm2")).unwrap(),
        PreTokenizer::Lfm2
    );
    let err = PreTokenizer::from_gguf(Some("starcoder")).unwrap_err();
    assert!(
        err.contains("starcoder"),
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
    (
        "llama-3.2-1b-instruct",
        "~/.cache/minfer/models/hf/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_llama-3.2-1b-instruct.json"),
    ),
    (
        "lfm2-1.2b",
        "~/.cache/minfer/models/hf/LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_lfm2-1.2b.json"),
    ),
    (
        "tinyllama-1.1b",
        "~/.cache/minfer/models/hf/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_tinyllama-1.1b.json"),
    ),
    (
        "bge-small-en-v1.5",
        "~/.cache/minfer/models/hf/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/bge-small-en-v1.5-q8_0.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_bge-small-en-v1.5.json"),
    ),
    (
        "flan-t5-small",
        "~/.cache/minfer/models/hf/Felladrin/gguf-flan-t5-small/flan-t5-small.Q8_0.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_flan-t5-small.json"),
    ),
    (
        "gemma-4-e2b",
        "~/.cache/minfer/models/hf/unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_gemma-4-e2b.json"),
    ),
    (
        "rwkv-6-world-1.6b",
        "~/.cache/minfer/models/hf/latestissue/rwkv-6-world-1b6-gguf/rwkv-6-world-1.6b-Q4_0.gguf",
        include_str!("../../tests/fixtures/tokenizer/ids_rwkv-6-world-1.6b.json"),
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
        // The engine: `pre` names the BPE rule (for `gemma4`, the rule the model
        // value derives — the artifact itself carries no key), and its absence
        // means SentencePiece or unigram (#132 S4). A fixture may also name the family
        // explicitly, which the artifact must agree with.
        if let Some(family) = fx.get("tokenizer_model").and_then(|v| v.as_str()) {
            assert_eq!(
                family,
                tok.model.gguf_name(),
                "{slug}: fixture tokenizer model does not match the GGUF"
            );
        }
        match (tok.pre, fx.get("pre").and_then(|v| v.as_str())) {
            (Some(pre), Some(want)) => assert_eq!(
                want,
                pre.gguf_name(),
                "{slug}: fixture pre-tokenizer does not match the GGUF"
            ),
            (None, None) => {} // SentencePiece: no pre-tokenization rule
            (Some(pre), None) => panic!(
                "{slug}: the artifact carries pre = {} but the fixture names none",
                pre.gguf_name()
            ),
            (None, Some(want)) => {
                panic!("{slug}: the fixture names pre = {want} but the artifact is SentencePiece")
            }
        }
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
        // token inside text is one id, never BPE-split. The pair is taken from
        // whatever this vocabulary declares (Qwen's ChatML pair, Llama-3's own
        // header/eot tokens, …), smallest id first — `#132` S2 added the check
        // and the model list is no longer Qwen-only.
        let mut declared: Vec<(&String, u32)> =
            tok.special_tokens.iter().map(|(t, &id)| (t, id)).collect();
        declared.sort_by_key(|&(_, id)| id);
        assert!(
            declared.len() >= 2,
            "{slug}: needs at least two declared special tokens"
        );
        let (open_text, open_id) = (declared[0].0.clone(), declared[0].1);
        let (close_text, close_id) = (declared[1].0.clone(), declared[1].1);
        let ids = tok.encode(&format!("{open_text}hi{close_text}"));
        assert_eq!(
            (ids.first().copied(), ids.last().copied()),
            (Some(open_id), Some(close_id)),
            "{slug}: special tokens {open_text:?}/{close_text:?} must survive as single ids: {ids:?}"
        );
        // #132 S1/S3, ADR-0044: the normalizer follows the rule's declaration.
        // The corpus above already carries decomposed entries for the family that
        // declares none (their reference ids are the decomposed ones), so this
        // asserts the *property* per family, and the unnormalized path is the
        // control: with NFC the decomposed input must differ from the composed
        // one, without it they must coincide.
        let composed = tok.encode("café");
        let decomposed = tok.encode("cafe\u{0301}");
        if tok.model == TokenizerModel::Ugm && !tok.ugm.xcda.is_empty() {
            // The precompiled charsmap normalizes both spellings to one sequence —
            // that is what this fixture's decomposed entries assert — so the
            // property is the fold, not the distinction the BPE/SPM families make.
            assert_eq!(
                decomposed, composed,
                "{slug}: the charsmap must fold both spellings onto one sequence"
            );
            assert_eq!(
                decomposed,
                tok.encode_raw("cafe\u{0301}"),
                "{slug}: `Tokenizer::encode_raw` must use the same normalization"
            );
        } else if tok.model == TokenizerModel::Bert {
            // The BertNormalizer is NFD + dropping `Mn`, so with the options this
            // artifact's file leaves at their defaults the composed and decomposed
            // spellings of an accented word are the *same* word — the opposite of
            // the property the BPE/SPM families assert, and what BERT declares.
            assert_eq!(
                tok.normalizer,
                NormalizerOptions {
                    lowercase: true,
                    strip_accents: true,
                },
                "{slug}: this fixture's file is expected to leave the BertNormalizer \
                 options at their defaults"
            );
            assert_eq!(
                decomposed, composed,
                "{slug}: the BertNormalizer must fold both spellings onto one word"
            );
            assert_eq!(
                decomposed,
                tok.encode_raw("cafe\u{0301}"),
                "{slug}: `Tokenizer::encode_raw` must use the same normalizer"
            );
        } else if tok.applies_nfc() {
            // The control is the BPE engine's *unnormalized* path (a normalizing
            // rule is always a BPE one; SentencePiece never normalizes).
            let unnormalized = tok.encode_bpe("cafe\u{0301}");
            assert_eq!(
                decomposed, composed,
                "{slug}: NFC must be applied to `Tokenizer::encode`"
            );
            assert_eq!(
                tok.encode_raw("cafe\u{0301}"),
                tok.encode_raw("café"),
                "{slug}: NFC must be applied to `Tokenizer::encode_raw`"
            );
            assert_ne!(
                unnormalized, composed,
                "{slug}: a real vocabulary must split the decomposed form differently"
            );
        } else {
            assert_ne!(
                decomposed, composed,
                "{slug}: this tokenizer declares no normalizer, so the decomposed input must stay decomposed"
            );
            assert_eq!(
                decomposed,
                tok.encode_raw("cafe\u{0301}"),
                "{slug}: `Tokenizer::encode` must not normalize what the file does not declare"
            );
        }
        eprintln!(
            "{slug}: {} corpus entries + special tokens match the reference byte for byte; \
             {}: café -> {composed:?}, decomposed -> {decomposed:?} (encode_raw {raw:?})",
            entries.len(),
            if tok.applies_nfc() {
                "NFC declared"
            } else if tok.model == TokenizerModel::Bert {
                "BertNormalizer declared"
            } else if tok.model == TokenizerModel::Ugm {
                "charsmap normalizer declared"
            } else {
                "no normalizer"
            },
            raw = tok.encode_raw("cafe\u{0301}")
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `tokenizer.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl TokenizerModel {
    /// The `tokenizer.ggml.model` spelling this engine was selected from —
    /// test-only (#132 S4): the id gate compares a fixture's `model` with the
    /// artifact's.
    pub fn gguf_name(self) -> &'static str {
        match self {
            Self::Gpt2 => "gpt2",
            Self::Spm => "llama",
            Self::Bert => "bert",
            Self::Ugm => "t5",
            Self::Gemma4 => "gemma4",
            Self::Rwkv => "rwkv",
        }
    }
}

impl PreTokenizer {
    /// The `tokenizer.ggml.pre` spelling this rule was selected from.
    ///
    /// Test-only (#239): driven by
    /// `tokenizer::tests::token_ids_match_the_reference`.
    pub fn gguf_name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
            Self::LlamaBpe => "llama-bpe",
            Self::Lfm2 => "lfm2",
            Self::Gemma4 => "gemma4",
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
