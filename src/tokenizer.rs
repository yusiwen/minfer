// BPE Tokenizer (self-contained, no external deps)
// Loads tokens, scores, types, and BPE merges directly from GGUF metadata

use crate::gguf::{GgufContext, GgufType};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Pre-tokenization (F7, #50)
//
// The rule set is selected by `tokenizer.ggml.pre` and each supported rule is a
// hand-written splitter, a port of llama.cpp's `unicode_regex_split_custom_*`.
// Hand-written rather than a `regex` crate pattern because the Rust `regex`
// crate has no lookahead and `\s+(?!\S)` is load-bearing for whitespace runs.
// The rules tile the input exactly (every byte belongs to one piece); the
// committed fixtures under tests/fixtures/tokenizer/split_*.json hold the
// reference splits produced by CPython `regex` from the model's own
// `tokenizer.json` pattern.
// ---------------------------------------------------------------------------

/// `^\p{L}$` etc. on one codepoint: the `regex` crate implements UTS#18 general
/// categories, the same sets the HF `tokenizers` `Split` patterns use.
fn unicode_class_is(cache: &'static OnceLock<Regex>, pattern: &str, c: char) -> bool {
    let re = cache.get_or_init(|| Regex::new(pattern).expect("static Unicode class pattern"));
    let mut buf = [0u8; 4];
    re.is_match(c.encode_utf8(&mut buf))
}

fn is_letter(c: char) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    unicode_class_is(&R, r"^\p{L}$", c)
}

fn is_number(c: char) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    unicode_class_is(&R, r"^\p{N}$", c)
}

fn is_accent_mark(c: char) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    unicode_class_is(&R, r"^\p{M}$", c)
}

/// A byte-level BPE pre-tokenizer rule, selected by `tokenizer.ggml.pre`.
///
/// An unsupported value (including a missing key) is a refused load, never a
/// guessed split — see [`PreTokenizer::from_gguf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// `qwen2` (aliases: `deepseek-r1-qwen`): Qwen2 / Qwen2.5 / Qwen3.
    Qwen2,
    /// `qwen35`: Qwen3.5 — letter runs also consume Unicode combining marks.
    Qwen35,
    /// `llama-bpe` (aliases: `llama3`, `llama-v3`): Llama 3.x — the letter and
    /// symbol arms are the Qwen2 ones, but digits group **up to three at a
    /// time** (`\p{N}{1,3}`).
    LlamaBpe,
    /// `gemma4`: Gemma-4 — the file's `model = gemma4` **forces** this rule
    /// (llama.cpp sets the pre-type from the model value, and a gemma4 GGUF
    /// carries no `tokenizer.ggml.pre`). Spaces are already `▁` when the split
    /// runs, so the rule is only **newline runs** (`[^\n]+|[\n]+`): a merge
    /// lookup never spans a newline, so a piece has to end at one.
    Gemma4,
    /// `lfm2`: LiquidAI LFM2 — a separate value whose reference pattern is
    /// **byte-identical** to `llama-bpe`'s (verified 2026-10-15 on
    /// `LiquidAI/LFM2-1.2B@40f3da0d`), and whose `tokenizer.json` declares no
    /// normalizer either. It is its own variant so the fixture and the refusal
    /// name the value the file actually carries.
    Lfm2,
    /// `jina-v1-en`: the **plain GPT-2** rule —
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`.
    /// A piece may start with **one space** (the other rules allow a preceding
    /// non-letter instead), digits group without limit, the symbol arm has no
    /// trailing-newline tail, and there is no `\s*[\r\n]+` arm.
    JinaV1En,
    /// `whitespace`: Jina's rule (llama.cpp #18756) — the text is cut on Unicode
    /// whitespace, the whitespace itself is **dropped**, and each remaining run is
    /// BPE'd on its own. With this rule the engine also turns lower-casing off
    /// (`llama-vocab.cpp:2231-2233`).
    Whitespace,
}

impl PreTokenizer {
    /// Resolve `tokenizer.ggml.pre`. Unknown and missing values are refused.
    pub fn from_gguf(pre: Option<&str>) -> Result<Self, String> {
        match pre {
            Some("qwen2") | Some("deepseek-r1-qwen") => Ok(Self::Qwen2),
            Some("qwen35") => Ok(Self::Qwen35),
            Some("llama-bpe") | Some("llama3") | Some("llama-v3") => Ok(Self::LlamaBpe),
            // The rest of llama.cpp's `LLAMA_VOCAB_PRE_TYPE_LLAMA3` group
            // (`llama-vocab.cpp:2158-2169`): same regex, plus `ignore_merges`.
            Some("falcon3") | Some("falcon-h1") | Some("pixtral") | Some("midm-2.0")
            | Some("jina-v5-nano") => Ok(Self::LlamaBpe),
            Some("lfm2") => Ok(Self::Lfm2),
            Some("whitespace") => Ok(Self::Whitespace),
            Some("jina-v1-en") => Ok(Self::JinaV1En),
            Some("gemma4") => Ok(Self::Gemma4),
            Some(other) => Err(format!(
                "tokenizer.ggml.pre = \"{other}\" is not supported by minfer's byte-level BPE \
                 pre-tokenizer (supported: \"qwen2\", alias \"deepseek-r1-qwen\"; \"qwen35\"; \
                 \"llama-bpe\", aliases \"llama3\", \"llama-v3\"; \"lfm2\"; \"gemma4\"; \
                 \"whitespace\"). \
                 Refusing to tokenize rather than split with the wrong rule."
            )),
            None => Err(
                "tokenizer.ggml.pre is missing, so the pre-tokenization rule is unknown; minfer \
                 refuses to guess (supported: \"qwen2\", alias \"deepseek-r1-qwen\"; \"qwen35\"; \
                 \"llama-bpe\", aliases \"llama3\", \"llama-v3\"; \"lfm2\"; \"gemma4\")"
                    .to_string(),
            ),
        }
    }

    /// Whether llama.cpp sets `ignore_merges` for this rule: a pre-split piece that is
    /// **itself** a vocabulary entry is emitted whole, before the merge ranks get a
    /// say (`llama-vocab.cpp:626-630`). True for the whole `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
    /// group (`:2155-2169`), which is where `llama-bpe` and `lfm2` sit too.
    fn ignores_merges(self) -> bool {
        matches!(self, Self::LlamaBpe | Self::Lfm2)
    }

    fn include_marks(self) -> bool {
        matches!(self, Self::Qwen35)
    }

    /// How many digits one piece may hold: Qwen's pattern is `\p{N}` (one per
    /// piece), Llama 3.x's is `\p{N}{1,3}` (up to three).
    fn number_run(self) -> usize {
        match self {
            Self::LlamaBpe | Self::Lfm2 => 3,
            // `gemma4` and `whitespace` never reach the letter/number arms (their own
            // splits run), so this is the unused arm of an exhaustive match.
            Self::Qwen2 | Self::Qwen35 | Self::Gemma4 | Self::Whitespace => 1,
            // The plain GPT-2 rule groups digits without limit.
            Self::JinaV1En => usize::MAX,
        }
    }

    /// Whether the input is normalized to NFC before pre-tokenization.
    ///
    /// The answer is the one the reference `tokenizer.json` declares for this
    /// rule's models — the GGUF carries no normalizer field, so the `pre` value is
    /// the only signal the file gives:
    ///
    /// * `qwen2`/`qwen35`: `"normalizer": {"type": "NFC"}` (verified 2026-10-15 on
    ///   `Qwen/Qwen3-0.6B@c1899de2`, `Qwen/Qwen2.5-0.5B-Instruct@main` and
    ///   `unsloth/Qwen3.5-0.8B@23c69c53`);
    /// * `llama-bpe`: `"normalizer": null` (verified on
    ///   `unsloth/Llama-3.2-1B-Instruct@5a8abab4`, and the family's other
    ///   revisions), so minfer must **not** normalize — llama.cpp does not either.
    ///
    /// A model whose `tokenizer.json` disagrees with its rule's answer is out of
    /// reach until the GGUF carries the flag; that limit is recorded in
    /// [#564](https://github.com/yusiwen/minfer/issues/564) and ADR-0044.
    pub fn applies_nfc(self) -> bool {
        match self {
            Self::Qwen2 | Self::Qwen35 => true,
            Self::LlamaBpe | Self::Lfm2 | Self::Gemma4 | Self::Whitespace | Self::JinaV1En => false,
        }
    }

    /// Split `text` into pre-tokenization pieces. The returned slices are in
    /// order and tile `text` exactly (concatenating them reproduces the input).
    ///
    /// The patterns differ in exactly two places: Qwen3.5 replaces `\p{L}` with
    /// `[\p{L}\p{M}]` in the letter arm, and Llama 3.x replaces the Qwen digit
    /// arm `\p{N}` with `\p{N}{1,3}`. Everything else is shared:
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|<digits>| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
    /// implemented as a scan rather than a `regex` pattern because the crate has
    /// no lookahead.
    pub fn split<'a>(self, text: &'a str) -> Vec<&'a str> {
        if self == Self::JinaV1En {
            // `(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
            // — see the variant's doc comment for how it differs from the others.
            let chars: Vec<(usize, char)> = text.char_indices().collect();
            let n = chars.len();
            let byte_of = |k: usize| chars.get(k).map(|&(b, _)| b).unwrap_or(text.len());
            let symbol = |c: char| !c.is_whitespace() && !is_letter(c) && !is_number(c);
            let mut out: Vec<&str> = Vec::new();
            let mut i = 0usize;
            while i < n {
                let start = i;
                let c = chars[i].1;
                let next = chars.get(i + 1).map(|&(_, x)| x);
                if c == '\'' {
                    let l1 = next.map(|x| x.to_ascii_lowercase());
                    let two = matches!(l1, Some('s' | 't' | 'm' | 'd'));
                    let three = matches!(
                        (l1, chars.get(i + 2).map(|&(_, x)| x.to_ascii_lowercase())),
                        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l'))
                    );
                    if two {
                        i += 2;
                    } else if three {
                        i += 3;
                    } else {
                        i += 1;
                        while i < n && symbol(chars[i].1) {
                            i += 1;
                        }
                    }
                } else if c == ' ' && next.map(is_letter).unwrap_or(false) {
                    i += 1;
                    while i < n && is_letter(chars[i].1) {
                        i += 1;
                    }
                } else if is_letter(c) {
                    i += 1;
                    while i < n && is_letter(chars[i].1) {
                        i += 1;
                    }
                } else if c == ' ' && next.map(is_number).unwrap_or(false) {
                    i += 1;
                    while i < n && is_number(chars[i].1) {
                        i += 1;
                    }
                } else if is_number(c) {
                    i += 1;
                    while i < n && is_number(chars[i].1) {
                        i += 1;
                    }
                } else if c == ' ' && next.map(symbol).unwrap_or(false) {
                    i += 1;
                    while i < n && symbol(chars[i].1) {
                        i += 1;
                    }
                } else if c != ' ' && symbol(c) {
                    i += 1;
                    while i < n && symbol(chars[i].1) {
                        i += 1;
                    }
                } else if c.is_whitespace() {
                    let mut j = i;
                    while j < n && chars[j].1.is_whitespace() {
                        j += 1;
                    }
                    // `\s+(?!\S)`: when a non-space follows a run of **two or more**,
                    // the last space stays for the next piece (` ?`); a single space is
                    // its own piece (the rule's final `\s+` alternative); at end of
                    // input the whole run goes.
                    i = if j < n && j - i > 1 { j - 1 } else { j };
                } else {
                    i += 1;
                }
                if byte_of(i) > byte_of(start) {
                    out.push(&text[byte_of(start)..byte_of(i)]);
                }
            }
            return out;
        }
        if self == Self::Whitespace {
            // `\s+` dropped, everything between kept: the segmenting the whitespace
            // session performs before it hands a run to the BPE tables.
            return text.split_whitespace().collect();
        }
        if self == Self::Gemma4 {
            // `[^\n]+|[\n]+` — maximal runs of newlines and of everything else.
            let mut out = Vec::new();
            let bytes = text.as_bytes();
            let mut start = 0usize;
            for (i, &b) in bytes.iter().enumerate() {
                if i > start && (b == b'\n') != (bytes[start] == b'\n') {
                    out.push(&text[start..i]);
                    start = i;
                }
            }
            if start < text.len() {
                out.push(&text[start..]);
            }
            return out;
        }
        let marks = self.include_marks();
        let word = |c: char| is_letter(c) || (marks && is_accent_mark(c));
        let symbol = |c: char| {
            !c.is_whitespace() && !is_letter(c) && !is_number(c) && !(marks && is_accent_mark(c))
        };

        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let n = chars.len();
        let cpt = |i: usize| -> Option<char> { chars.get(i).map(|&(_, c)| c) };
        let byte_of = |i: usize| -> usize { chars.get(i).map(|&(b, _)| b).unwrap_or(text.len()) };

        let mut out: Vec<&'a str> = Vec::new();
        let mut prev = 0usize; // byte offset where the pending piece starts
        let mut pos = 0usize; // char index of the scan head
        macro_rules! emit {
            ($end:expr) => {{
                let e = byte_of($end);
                if e > prev {
                    out.push(&text[prev..e]);
                }
                prev = e;
            }};
        }

        while pos < n {
            let c = chars[pos].1;

            // (?i:'s|'t|'re|'ve|'m|'ll|'d) — case-insensitive contraction.
            if c == '\'' {
                if let Some(c1) = cpt(pos + 1) {
                    let l1 = c1.to_ascii_lowercase();
                    if matches!(l1, 's' | 't' | 'm' | 'd') {
                        pos += 2;
                        emit!(pos);
                        continue;
                    }
                    if let Some(c2) = cpt(pos + 2) {
                        let l2 = c2.to_ascii_lowercase();
                        if (l1 == 'r' && l2 == 'e')
                            || (l1 == 'v' && l2 == 'e')
                            || (l1 == 'l' && l2 == 'l')
                        {
                            pos += 3;
                            emit!(pos);
                            continue;
                        }
                    }
                }
            }

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if c != '\r' && c != '\n' && !is_number(c) {
                let next_is_word = cpt(pos + 1).map(word).unwrap_or(false);
                if word(c) || next_is_word {
                    pos += 1;
                    while pos < n && word(chars[pos].1) {
                        pos += 1;
                    }
                    emit!(pos);
                    continue;
                }
            }

            // \p{N}{1,3} (llama-bpe) or \p{N} (Qwen, one digit per piece)
            if is_number(c) {
                let cap = self.number_run();
                let mut taken = 0usize;
                while taken < cap && pos < n && is_number(chars[pos].1) {
                    pos += 1;
                    taken += 1;
                }
                emit!(pos);
                continue;
            }

            // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let next = if c == ' ' { cpt(pos + 1) } else { Some(c) };
            if next.map(symbol).unwrap_or(false) {
                if c == ' ' {
                    pos += 1;
                }
                while pos < n && symbol(chars[pos].1) {
                    pos += 1;
                }
                while pos < n && (chars[pos].1 == '\r' || chars[pos].1 == '\n') {
                    pos += 1;
                }
                emit!(pos);
                continue;
            }

            // \s*[\r\n]+
            let mut num_ws = 0usize;
            let mut last_end_r_or_n = 0usize;
            while let Some(wc) = cpt(pos + num_ws) {
                if !wc.is_whitespace() {
                    break;
                }
                if wc == '\r' || wc == '\n' {
                    last_end_r_or_n = pos + num_ws + 1;
                }
                num_ws += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                emit!(pos);
                continue;
            }

            // \s+(?!\S)
            if num_ws > 1 && cpt(pos + num_ws).is_some() {
                pos += num_ws - 1;
                emit!(pos);
                continue;
            }

            // \s+
            if num_ws > 0 {
                pos += num_ws;
                emit!(pos);
                continue;
            }

            // No alternative matched: emit the single codepoint.
            pos += 1;
            emit!(pos);
        }
        if prev < text.len() {
            out.push(&text[prev..]);
        }
        out
    }
}

/// The tokenizer family, from `tokenizer.ggml.model` (#132 S4).
///
/// This is a **dispatch**, not a hint: the value selects the engine, and every
/// other engine llama.cpp knows is refused by name rather than approximated with
/// the wrong one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// `gpt2` (or a missing key): byte-level BPE — merge ranks plus the 256
    /// byte-to-unicode byte tokens, pre-tokenized by [`PreTokenizer`].
    Gpt2,
    /// `llama`: SentencePiece — score-driven merges over a `▁`-escaped text with
    /// `<0xXX>` byte fallback. The `tokenizer.ggml.pre` key does **not**
    /// participate: llama.cpp fixes the SPM flags from the model type
    /// (`add_space_prefix = true`, `clean_spaces = false`,
    /// `src/llama-vocab.cpp:2409`) and never pre-splits an SPM text.
    Spm,
    /// `t5`: unigram — llama.cpp's UGM. The text is rewritten by the file's
    /// **precompiled charsmap** (an XCDA-compressed prefix replacement table) and
    /// by the whitespace rules, then segmented by a Viterbi walk over the
    /// vocabulary trie weighted by `tokenizer.ggml.scores`, with one unknown token
    /// per run of unmatched input. `pre`, merges and byte tokens do not
    /// participate.
    Ugm,
    /// `rwkv`: RWKV — a greedy longest match over a trie built from the
    /// vocabulary's **unescaped** bytes (`\xNN`, `\t`, `\n`, `\r` escapes are
    /// decoded first, because the tokens stand for arbitrary bytes). No merges, no
    /// scores, no `pre` rule and no byte-level encoding: a byte that starts no
    /// entry emits the unknown token and advances one byte.
    Rwkv,
    /// `whitespace`: Jina's tokenizer — the `gpt2` engine behind a pre rule that cuts
    /// on whitespace and drops it. llama.cpp maps the model value to the BPE type and
    /// only swaps the session (`llama-vocab.cpp:1986`), so the engine is the byte-level
    /// BPE; the distinguishing behaviour is the `whitespace` pre rule.
    Whitespace,
    /// `none` / `no_vocab`: the file declares **no vocabulary** — llama.cpp reads
    /// `tokenizer.ggml.vocab_size`, creates that many placeholder tokens and aborts
    /// in `tokenize` (`llama-vocab.cpp:1942`/`:3587`), because the ids come from the
    /// caller (vision/audio/diffusion pipelines). minfer accepts the file so it can
    /// be inspected, and refuses text tokenization through the checked entry points.
    None,
    /// `hybriddna`: Carbon-3B's HybridDNA tokenizer — byte-level BPE plus a
    /// `<dna>`…`</dna>` block whose content is upper-cased and cut into 6-mers that
    /// look up tokens carrying a reserved U+E000 suffix (`<oov>` when a k-mer is
    /// missing, the tail padded with `A`). Outside the tags it is the `gpt2` engine,
    /// `pre` rule included.
    HybridDna,
    /// `plamo2`: PLaMo-2 — a backward DP over codepoints whose objective is the
    /// vocabulary's own `tokenizer.ggml.scores`, with one `<0xXX>` byte token per
    /// byte as the fallback (never `<unk>`). The `pre` key does **not** participate
    /// (there is no regex rule), and neither do merges.
    Plamo2,
    /// `gemma4`: Gemma-4 — byte-level BPE's merge loop over **raw UTF-8** (no
    /// byte-to-unicode step), a normalizer that replaces spaces with `▁`, the
    /// gemma4 newline rule taken from the model value (the file carries no `pre`,
    /// llama.cpp sets it at load) and a `<0xXX>` byte fallback.
    Gemma4,
    /// `bert`: WordPiece — llama.cpp's WPM. Each word is prefixed with the
    /// phantom `▁` and segmented by **greedy longest match** against the
    /// vocabulary, with one `<unk>` per word that does not fully match. Neither
    /// `pre` (there is no regex pre-tokenizer) nor merges nor scores participate;
    /// the BertNormalizer options come from `tokenizer.ggml.normalizer.lowercase`
    /// and `.strip_accents` (both default `false`).
    Bert,
}

impl TokenizerModel {
    /// Resolve `tokenizer.ggml.model`. Only the two engines this crate
    /// implements are accepted; every other value is a refusal naming it.
    pub fn from_gguf(model: Option<&str>) -> Result<Self, String> {
        match model {
            // llama.cpp reads this key as **required** (`ml.get_key(LLM_KV_TOKENIZER_MODEL,
            // tokenizer_model)`, no optional flag, `llama-vocab.cpp:1935`), so a file
            // without it is refused rather than assumed to be `gpt2` ([#585]).
            //
            // [#585]: https://github.com/yusiwen/minfer/issues/585
            None => Err(
                "tokenizer.ggml.model is missing, so the tokenizer engine is unknown; minfer \
                 refuses the file rather than assuming `gpt2` (llama.cpp reads this key as \
                 required too)"
                    .to_string(),
            ),
            Some("gpt2") => Ok(Self::Gpt2),
            Some("none") | Some("no_vocab") => Ok(Self::None),
            Some("whitespace") => Ok(Self::Whitespace),
            Some("llama") => Ok(Self::Spm),
            Some("bert") => Ok(Self::Bert),
            Some("gemma4") => Ok(Self::Gemma4),
            Some("rwkv") => Ok(Self::Rwkv),
            Some("plamo2") => Ok(Self::Plamo2),
            Some("hybriddna") => Ok(Self::HybridDna),
            Some("t5") => Ok(Self::Ugm),
            Some(other) => Err(format!(
                "tokenizer.ggml.model = \"{other}\" is not a tokenizer minfer implements \
                 (supported: \"gpt2\" — byte-level BPE — \"llama\" — SentencePiece — \
                 \"bert\" — WordPiece — \"t5\" — unigram — \"gemma4\" — Gemma-4's \
                 raw-UTF-8 BPE — and \"rwkv\" — RVKV's trie match). Refusing to run a \
                 different engine rather than tokenize with the wrong one."
            )),
        }
    }
}

/// The text SentencePiece and WordPiece use for a space (`U+2581`).
const SPM_SPACE: char = '\u{2581}';

/// The BertNormalizer options the WordPiece engine reads
/// (`tokenizer.ggml.normalizer.*`).
///
/// **Both default to `true`** — llama.cpp's `normalizer_options` member
/// initialisers (`llama-vocab.h:83`), which is why a converted BERT file that
/// carries neither key still lowercases its input; `strip_accents` follows
/// `lowercase` unless the file says otherwise. The one pre-type that turns
/// lowercasing off is `whitespace`, which this engine does not implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizerOptions {
    /// Lowercase every code point (the simple 1:1 mapping).
    pub lowercase: bool,
    /// Decompose with NFD and drop the combining marks.
    pub strip_accents: bool,
}

impl Default for NormalizerOptions {
    fn default() -> Self {
        Self {
            lowercase: true,
            strip_accents: true,
        }
    }
}

/// BERT's BasicTokenizer preprocessing, as the WordPiece engine needs it
/// (`llm_tokenizer_wpm_session::preprocess`, `llama-vocab.cpp:827`).
///
/// Whitespace ends a word; a control character is dropped; with `strip_accents`
/// the text is NFD-decomposed first and its marks are dropped; with `lowercase`
/// every code point is lowercased; and punctuation, an ASCII symbol or a Chinese
/// character becomes a word of its own.
fn wpm_preprocess(text: &str, opts: NormalizerOptions) -> Vec<String> {
    let decomposed = if opts.strip_accents {
        nfc::nfd(text).into_owned()
    } else {
        text.to_string()
    };

    let mut words: Vec<String> = vec![String::new()];
    for c in decomposed.chars() {
        if c.is_whitespace() {
            if !words.last().is_some_and(String::is_empty) {
                words.push(String::new());
            }
            continue;
        }
        if c == '\0' || c == '\u{FFFD}' || is_control_char(c) {
            continue;
        }
        if opts.strip_accents && is_accent_mark_char(c) {
            continue;
        }
        let piece: String = if opts.lowercase {
            // The *simple* 1:1 mapping llama.cpp's `unicode_tolower` table
            // implements; `char::to_lowercase` is the full mapping, so a code
            // point whose full mapping is multi-character (U+0130 is the classic)
            // is kept as the simple first character instead of expanding.
            let mut lowered = c.to_lowercase();
            match (lowered.next(), lowered.next()) {
                (Some(first), None) => first.to_string(),
                _ => c.to_string(),
            }
        } else {
            c.to_string()
        };

        if is_punctuation_char(c)
            || (c < '\u{7F}' && !c.is_alphanumeric() && c != '_')
            || is_chinese_char(c as u32)
        {
            if !words.last().is_some_and(String::is_empty) {
                words.push(String::new());
            }
            *words.last_mut().expect("never empty") = piece;
            words.push(String::new());
        } else {
            words
                .last_mut()
                .expect("never empty")
                .push_str(piece.as_str());
        }
    }
    if words.last().is_some_and(String::is_empty) {
        words.pop();
    }
    words
}

/// A `regex` Unicode class asked about a single character, without allocating.
fn char_in_class(re: &regex::Regex, c: char) -> bool {
    let mut buf = [0u8; 4];
    re.is_match(c.encode_utf8(&mut buf))
}

/// Unicode general category `C` — the WordPiece normalizer drops these.
fn is_control_char(c: char) -> bool {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    char_in_class(
        RE.get_or_init(|| regex::Regex::new(r"\p{C}").expect("a static pattern")),
        c,
    )
}

/// Unicode general category `P` — WordPiece splits on these.
fn is_punctuation_char(c: char) -> bool {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    char_in_class(
        RE.get_or_init(|| regex::Regex::new(r"\p{P}").expect("a static pattern")),
        c,
    )
}

/// Unicode general category `Mn` — the marks `strip_accents` drops.
///
/// **Non-spacing marks only.** BERT's `_run_strip_accents` drops a character when
/// `unicodedata.category(char) == "Mn"`, so a *spacing* mark (`Mc`, e.g. the
/// Devanagari vowel sign U+093F) survives; llama.cpp's table flags all of `\p{M}`
/// and drops those too, which is a recorded divergence (ADR-0046).
fn is_accent_mark_char(c: char) -> bool {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    char_in_class(
        RE.get_or_init(|| regex::Regex::new(r"\p{Mn}").expect("a static pattern")),
        c,
    )
}

/// BERT's `_is_chinese_char`, ranges and typo included (llama.cpp notes that the
/// HF Rust code spells `0x2B920` where the intent is `0x2B820`).
fn is_chinese_char(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B920..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Group declared special tokens by first char, longest-first inside each group
/// (the earliest-position, longest-match scan in [`Tokenizer::encode`] needs
/// both). Built by the loader and by the tests' synthetic vocabularies.
fn special_index(special_tokens: &HashMap<String, u32>) -> HashMap<char, Vec<(String, u32)>> {
    let mut index: HashMap<char, Vec<(String, u32)>> = HashMap::new();
    for (pat, id) in special_tokens.iter() {
        let first = pat.chars().next().unwrap_or('\0');
        index.entry(first).or_default().push((pat.clone(), *id));
    }
    for group in index.values_mut() {
        group.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }
    index
}

/// The reserved suffix (U+E000) that keeps a DNA k-mer distinct from an identical
/// base-vocabulary BPE token of the same spelling; erased from `id_to_token` at load
/// and kept in the token-to-id map (`llama-vocab.cpp:2524`).
const DNA_KMER_MARKER: &str = "\u{e000}";

/// Decode an RVKV vocabulary entry: `\xNN` is the byte 0xNN, `\t`/`\n`/`\r` are
/// themselves, any other `\c` is `c`, and a trailing lone `\` is dropped
/// (`llama_unescape_rwkv_token`, `llama-vocab.cpp:1247`).
fn unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining = 0u8;
    let mut hex_acc = 0u8;
    for &b in escaped.as_bytes() {
        if hex_remaining != 0 {
            let value = if b >= b'a' { b - b'a' + 10 } else { b - b'0' };
            hex_acc = (hex_acc << 4) + value;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }
        if escaping {
            match b {
                b't' => out.push(b'\t'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b'x' => hex_remaining = 2,
                other => out.push(other),
            }
            escaping = false;
            continue;
        }
        if b == b'\\' {
            escaping = true;
            continue;
        }
        out.push(b);
    }
    out
}

/// Parse a SentencePiece byte token (`<0xXX>`) into the byte it stands for.
/// Anything else (including a malformed spelling) is not a byte token.
fn parse_spm_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// A byte trie over vocabulary entries — the unigram engine walks it once per
/// input code point to find every token that starts there.
#[derive(Default, Clone)]
struct VocabTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Default, Clone)]
struct TrieNode {
    children: HashMap<u8, u32>,
    token: Option<u32>,
}

impl VocabTrie {
    fn insert(&mut self, text: &str, id: u32) {
        self.insert_bytes(text.as_bytes(), id);
    }

    /// Insert a token whose bytes are **not** necessarily valid UTF-8 (the RWKV
    /// vocabulary escapes arbitrary bytes as `\xNN`, and the trie has to hold the
    /// decoded bytes, not the escape text).
    fn insert_bytes(&mut self, bytes: &[u8], id: u32) {
        if self.nodes.is_empty() {
            self.nodes.push(TrieNode::default());
        }
        let mut node = 0u32;
        for &b in bytes {
            let next = match self.nodes[node as usize].children.get(&b) {
                Some(&next) => next,
                None => {
                    self.nodes.push(TrieNode::default());
                    let next = (self.nodes.len() - 1) as u32;
                    self.nodes[node as usize].children.insert(b, next);
                    next
                }
            };
            node = next;
        }
        self.nodes[node as usize].token = Some(id);
    }

    /// Every `(id, end)` whose entry is a prefix of `bytes[start..]`, in
    /// increasing `end` order.
    fn matches(&self, bytes: &[u8], start: usize) -> Vec<(u32, usize)> {
        let mut out = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        let mut node = 0u32;
        for (i, &b) in bytes[start..].iter().enumerate() {
            match self.nodes[node as usize].children.get(&b) {
                Some(&next) => node = next,
                None => break,
            }
            if let Some(id) = self.nodes[node as usize].token {
                out.push((id, start + i + 1));
            }
        }
        out
    }

    /// The child of `node` for the byte `b`, if any.
    fn step(&self, node: u32, b: u8) -> Option<u32> {
        self.nodes
            .get(node as usize)
            .and_then(|n| n.children.get(&b).copied())
    }

    /// The entry that ends at `node`, if any.
    fn terminal(&self, node: u32) -> Option<u32> {
        self.nodes.get(node as usize).and_then(|n| n.token)
    }

    /// Whether any entry starts with the byte `b` (a root child).
    fn starts_with_byte(&self, b: u8) -> bool {
        !self.nodes.is_empty() && self.nodes[0].children.contains_key(&b)
    }

    /// The longest entry that is a prefix of `text`, with its byte length.
    fn longest_prefix(&self, text: &str) -> Option<(u32, usize)> {
        self.longest_prefix_bytes(text.as_bytes())
    }

    /// The longest entry that is a prefix of `bytes`, with its byte length.
    fn longest_prefix_bytes(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = None;
        let mut node = 0u32;
        for (i, &b) in bytes.iter().enumerate() {
            match self.nodes[node as usize].children.get(&b) {
                Some(&next) => node = next,
                None => break,
            }
            if let Some(id) = self.nodes[node as usize].token {
                best = Some((id, i + 1));
            }
        }
        best
    }
}

/// `hybriddna`'s DNA block: the three tag ids and the marked 6-mers, keyed by the
/// **bare** k-mer (the marker is erased from `id_to_token` at load, as llama.cpp
/// does, so detokenization yields the bare sequence).
#[derive(Default, Clone)]
struct DnaData {
    begin: u32,
    end: u32,
    oov: u32,
    kmers: HashMap<String, u32>,
}

/// The unigram engine's loaded state: the two tries, the score bounds and the
/// decoded precompiled charsmap (empty when the file carries none).
#[derive(Default, Clone)]
struct UgmData {
    /// Every NORMAL / USER_DEFINED / UNUSED entry, for the Viterbi walk.
    trie: VocabTrie,
    /// The USER_DEFINED entries alone: a prefix matching one of these is emitted
    /// whole, before the charsmap is consulted.
    user_defined: VocabTrie,
    /// The lowest normal-token score; the unknown transition costs 10 below it.
    unknown_token_score: f32,
    /// The XCDA array and the replacement strings of
    /// `tokenizer.ggml.precompiled_charsmap`.
    xcda: Vec<u32>,
    prefix_replacements: Vec<u8>,
}

/// A candidate merge in the SPM session: two adjacent symbols whose
/// concatenation is a vocabulary entry, ordered by that entry's score.
#[derive(PartialEq)]
struct SpmBigram {
    /// The score of the merged vocabulary entry — the higher, the earlier.
    score: f32,
    left: usize,
    right: usize,
    /// The byte length of the concatenation when the bigram was pushed; a
    /// mismatch means one side grew since, and the bigram is stale.
    size: usize,
}

impl Eq for SpmBigram {}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Highest score first; ties by the smaller left index, exactly as
        // llama.cpp's `llm_bigram_spm::comparator`.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// One symbol in the SPM session's doubly-linked list.
struct SpmSymbol {
    text: String,
    /// Byte length of `text`; 0 marks a symbol merged into its left neighbour.
    n: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Build byte-to-unicode mapping (GPT-2 style).
fn build_byte_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u32> = Vec::new();
    // Printable ASCII: ! to ~
    for b in 0x21..=0x7e {
        bs.push(b);
    }
    // Latin-1 supplement: ¡ to ¬
    for b in 0xa1..=0xac {
        bs.push(b);
    }
    // Latin-1 supplement: ® to ÿ
    for b in 0xae..=0xff {
        bs.push(b);
    }

    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b as u8, char::from_u32(c).unwrap()))
        .collect()
}

/// Byte-encode text using GPT-2 unicode mapping.
fn byte_encode(text: &str, byte_to_unicode: &HashMap<u8, char>) -> String {
    let mut result = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        result.push(byte_to_unicode[&b]);
    }
    result
}

/// llama.cpp's `MAX_CHAR_LEN` for a DRY sequence breaker (`llama-sampler.cpp`).
///
/// llama.cpp *truncates* a longer breaker with a warning; minfer refuses it by
/// name instead (a refusal is the house rule for an input this engine cannot
/// carry faithfully — ADR-0009). The limit is counted in Unicode scalar values
/// here, which agrees with llama.cpp's byte count for the ASCII breakers this
/// feature exists for (`\n`, `:`, `"`, `*`).
pub const MAX_BREAKER_CHARS: usize = 40;

/// llama.cpp's `MAX_SEQ_LEN` for the tail of an overlapping token sequence.
///
/// Same story as [`MAX_BREAKER_CHARS`]: llama.cpp truncates the tail
/// tokenization to this many tokens, minfer refuses it.
pub const MAX_BREAKER_TAIL_TOKENS: usize = 20;

#[derive(Clone)]
pub struct Tokenizer {
    pub id_to_token: Vec<String>,
    /// The GGUF `tokenizer.ggml.scores` array.
    ///
    /// Read by the plamo2 engine ([#132] S4): its backward DP maximizes the sum of
    /// these scores, so `plamo2_score_int` is the field's first production reader
    /// (the allowance [#244] carried was removed with it). The other engines still
    /// ignore the array.
    ///
    /// [#132]: https://github.com/yusiwen/minfer/issues/132
    /// [#244]: https://github.com/yusiwen/minfer/issues/244
    pub id_to_score: Vec<f32>,
    /// The GGUF `tokenizer.ggml.token_type` array.
    ///
    /// Retained with an allowance ([#244]'s verdict: keep, deferred): the *field*
    /// is dead in every build, but the loader's local copy of the same array is
    /// what derives the special-token table (`special_tokens` /
    /// `special_by_first`, types 3/4), so the parse is not dead — only this copy
    /// of it. What would read the field: a diagnostic or writer that re-emits the
    /// metadata, or a tokenizer test that inspects the loaded types.
    #[allow(dead_code)]
    pub id_to_type: Vec<i32>,
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), usize>,
    byte_to_unicode: HashMap<u8, char>,
    /// Reverse mapping for decode
    unicode_to_byte: HashMap<char, u8>,
    /// The special tokens the file declares: `tokenizer.ggml.token_type` 2/3/4,
    /// plus every id a `tokenizer.ggml.*_token_id` role names (#132 S2).
    ///
    /// Read only by `tokenizer::tests`; production matches through
    /// `special_by_first`, which the loader builds from this map.
    #[cfg_attr(not(test), allow(dead_code))]
    pub special_tokens: HashMap<String, u32>,
    /// The engine selected from `tokenizer.ggml.model` (#132 S4).
    pub model: TokenizerModel,
    /// The pre-tokenization rule selected from `tokenizer.ggml.pre` (F7) — the
    /// **BPE** engine's step. `None` for SentencePiece, which has no
    /// pre-tokenization rule: llama.cpp fixes the SPM flags from the model type
    /// and never pre-splits an SPM text.
    pub pre: Option<PreTokenizer>,
    /// SentencePiece: whether a raw-text segment is prefixed with a space
    /// (`tokenizer.ggml.add_space_prefix`, default `true` for SPM).
    add_space_prefix: bool,
    /// SentencePiece: the `<0xXX>` byte token ids, one per byte value. Empty for
    /// the BPE engine, whose byte fallback is the 256 byte-level characters.
    spm_byte_to_id: HashMap<u8, u32>,
    /// WordPiece: the BertNormalizer options from `tokenizer.ggml.normalizer.*`.
    normalizer: NormalizerOptions,
    /// WordPiece: the longest vocabulary entry in bytes — the greedy match's
    /// upper bound (`llama_vocab::max_token_len`).
    max_token_len: usize,
    /// WordPiece: the `<unk>` id, the fallback for a word that cannot be fully
    /// segmented. `load` refuses a vocabulary that does not declare one. The
    /// unigram engine reads it too (one unknown per unmatched run).
    unk_token: u32,
    /// Unigram (`t5`): the trie, the score bound and the charsmap.
    ugm: UgmData,
    /// Whether a pre-split piece that is a whole vocabulary entry is emitted before
    /// the merge ranks (llama.cpp's `ignore_merges`; set from the pre rule).
    ignore_merges: bool,
    /// `hybriddna` only; `None` for every other engine.
    dna: Option<DnaData>,
    /// Unigram (`t5`): `tokenizer.ggml.remove_extra_whitespaces` — collapse a run
    /// of whitespace into one `▁`.
    remove_extra_whitespaces: bool,
    /// Special tokens grouped by first char, longest-first within a group,
    /// built from `special_tokens` — the file's declared types and roles
    /// (#132 S2). Nothing here is inferred from a token's name.
    special_by_first: HashMap<char, Vec<(String, u32)>>,
    pub bos_token: u32,
    /// The EOS id; production reads it (`spec::SpecEngine` and the special-token
    /// table).
    pub eos_token: u32,
}

/// The `tokenizer.ggml.*_token_id` keys that name a **single vocabulary entry**,
/// transcribed from llama.cpp's role table (`src/llama-arch.cpp:382-406`, with
/// the deprecated `prefix`/`suffix`/`middle` aliases it still reads at
/// `src/llama-vocab.cpp:2584-2586`).
///
/// A declared role is a declaration, not a shape: the file says "this id is the
/// EOS / eot / fim_pad", so that token matches as one id even when
/// `tokenizer.ggml.token_type` says NORMAL. llama.cpp forces its *name* list
/// upward only when the role is **not** declared; minfer never guesses a name at
/// all, so a type-1 token is special here exactly when the file names it
/// ([#132](https://github.com/yusiwen/minfer/issues/132) S2, ADR-0043).
///
/// The key spellings are llama.cpp's, typo included (`seperator_token_id`).
const SPECIAL_ROLE_KEYS: &[&str] = &[
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.seperator_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.fim_pre_token_id",
    "tokenizer.ggml.fim_suf_token_id",
    "tokenizer.ggml.fim_mid_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
    "tokenizer.ggml.prefix_token_id",
    "tokenizer.ggml.suffix_token_id",
    "tokenizer.ggml.middle_token_id",
];

impl Tokenizer {
    /// A vocabulary with no tokens.
    ///
    /// Test-only: F8's server tests need an `AppState` (to build the real router
    /// and drive it over HTTP) without a model on disk. Nothing tokenizes on the
    /// paths they assert — the drain check runs before tokenization, and
    /// `/health`, `/v1/models` and `/metrics` never touch the vocabulary — so an
    /// empty one is honest rather than a stub that could hide a real call.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            id_to_token: Vec::new(),
            id_to_score: Vec::new(),
            id_to_type: Vec::new(),
            vocab: HashMap::new(),
            merges: HashMap::new(),
            byte_to_unicode: HashMap::new(),
            unicode_to_byte: HashMap::new(),
            special_tokens: HashMap::new(),
            model: TokenizerModel::Gpt2,
            pre: Some(PreTokenizer::Qwen2),
            add_space_prefix: false,
            spm_byte_to_id: HashMap::new(),
            normalizer: NormalizerOptions::default(),
            max_token_len: 0,
            unk_token: 0,
            ugm: UgmData::default(),
            ignore_merges: false,
            dna: None,
            remove_extra_whitespaces: false,
            special_by_first: HashMap::new(),
            bos_token: 0,
            eos_token: 0,
        }
    }

    /// A byte-level-complete vocabulary and nothing else: the 256 byte tokens
    /// at id `1000 + byte`, the `qwen2` pre-tokenizer, no merges.
    ///
    /// Test-only, alongside [`Tokenizer::empty`]. The DRY breaker resolver needs
    /// a vocabulary whose tokenization is fully known and model-independent, and
    /// the server's request tests need one to resolve a breaker string without a
    /// GGUF on disk. Ids below 1000 are empty tokens, exactly like the pad /
    /// unused slots a real vocabulary has.
    #[cfg(test)]
    pub(crate) fn byte_level_for_tests() -> Self {
        const BASE: u32 = 1000;
        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte = byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();
        let mut vocab = HashMap::new();
        let mut id_to_token = vec![String::new(); (BASE + 256) as usize];
        for (&b, &c) in &byte_to_unicode {
            vocab.insert(c.to_string(), BASE + b as u32);
            id_to_token[(BASE + b as u32) as usize] = c.to_string();
        }
        let n = id_to_token.len();
        Self {
            id_to_token,
            id_to_score: vec![0.0; n],
            id_to_type: vec![1; n],
            vocab,
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
            ignore_merges: false,
            dna: None,
            remove_extra_whitespaces: false,
            special_by_first: HashMap::new(),
            bos_token: 0,
            eos_token: 0,
        }
    }

    /// Load tokenizer from a GgufContext (re-parses metadata only, no tensor data).
    ///
    /// Refuses loudly (returns `Err`) when the metadata describes a tokenizer
    /// this engine cannot reproduce byte for byte:
    ///   * `tokenizer.ggml.model` naming an engine other than `gpt2`
    ///     (byte-level BPE) or `llama` (SentencePiece) — see
    ///     [`TokenizerModel::from_gguf`];
    ///   * for `gpt2`: `tokenizer.ggml.pre` missing or unsupported (see
    ///     [`PreTokenizer`]), no merges (a byte-level BPE without ranks would
    ///     silently degrade to per-byte splits), or a vocabulary that is not
    ///     byte-level complete (the byte fallback in [`Tokenizer::bpe_encode`]
    ///     would otherwise drop bytes silently);
    ///   * for `llama`: an empty `tokenizer.ggml.scores` (the SPM merges are
    ///     score-ordered) or a vocabulary missing any `<0xXX>` byte token (the
    ///     SPM byte fallback would otherwise drop bytes).
    pub fn load(gguf: &GgufContext) -> Result<Self, String> {
        // Load token strings
        let mut id_to_token: Vec<String> = Vec::new();
        let mut id_to_score: Vec<f32> = Vec::new();
        let mut id_to_type: Vec<i32> = Vec::new();

        for kv in &gguf.kv {
            if kv.key == "tokenizer.ggml.tokens" && kv.is_array {
                for i in 0..kv.get_ne() {
                    id_to_token.push(kv.get_val_str(i).to_string());
                }
            }
            if kv.key == "tokenizer.ggml.scores" && kv.is_array {
                for i in 0..kv.get_ne() {
                    id_to_score.push(kv.get_val_f32(i));
                }
            }
            if kv.key == "tokenizer.ggml.token_type" && kv.is_array {
                for i in 0..kv.get_ne() {
                    id_to_type.push(kv.get_val_i32(i));
                }
            }
        }

        // Default scores/types if missing
        if id_to_score.is_empty() {
            id_to_score = vec![0.0f32; id_to_token.len()];
        }
        if id_to_type.is_empty() {
            id_to_type = vec![1i32; id_to_token.len()];
        }

        // Build vocab: token string → id
        let mut vocab = HashMap::new();
        for (id, token) in id_to_token.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }

        // Load BPE merge ranks
        let mut merges = HashMap::new();
        for kv in &gguf.kv {
            if kv.key == "tokenizer.ggml.merges" && kv.is_array {
                for i in 0..kv.get_ne() {
                    let s = kv.get_val_str(i);
                    if let Some(pos) = s.find(' ') {
                        let first = s[..pos].to_string();
                        let second = s[pos + 1..].to_string();
                        merges.insert((first, second), i);
                    }
                }
            }
        }

        // Special tokens: what the file declares, never a name shape (#132 S2).
        //
        //  * `tokenizer.ggml.token_type` 2 (UNKNOWN), 3 (CONTROL) or 4
        //    (USER_DEFINED) — llama.cpp's special-token cache predicate
        //    (`src/llama-vocab.cpp:3012`, `CONTROL | USER_DEFINED | UNKNOWN`).
        //  * every id a declared role names ([`SPECIAL_ROLE_KEYS`]): the file
        //    says "this is the EOS / eot / …", so that token matches as one id
        //    even when its type says NORMAL.
        //
        // The `<|im_start|>` / `<|im_end|>` / EOS-shaped fallbacks this used to
        // add are gone: they fabricated tokens the vocabulary does not contain.
        // Measured on `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` (Q4_K_M,
        // 2026-10-15), whose vocabulary has neither name: `<|im_start|>` was
        // mapped to id 0 and `<|im_end|>` to the EOS token 151643, so two
        // literal strings tokenized as tokens the model never had. llama.cpp
        // matches only declared control tokens either; a converter that leaves a
        // chat token type NORMAL gets it BPE-split there as well.
        let mut special_tokens = HashMap::new();
        for (id, token) in id_to_token.iter().enumerate() {
            if matches!(id_to_type.get(id).copied().unwrap_or(1), 2 | 3 | 4) {
                special_tokens.insert(token.clone(), id as u32);
            }
        }
        for key in SPECIAL_ROLE_KEYS {
            if let Some(id) = Self::get_gguf_u32(gguf, key) {
                if let Some(text) = id_to_token.get(id as usize) {
                    if !text.is_empty() {
                        special_tokens.insert(text.clone(), id);
                    }
                }
            }
        }

        // Token IDs from GGUF metadata
        let bos_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.eos_token_id").unwrap_or(0);

        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte: HashMap<char, u8> =
            byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();

        let special_by_first = special_index(&special_tokens);

        // #132 S4: the engine comes from `tokenizer.ggml.model`, and each engine
        // validates its own prerequisites.
        let model =
            TokenizerModel::from_gguf(Self::get_gguf_str(gguf, "tokenizer.ggml.model").as_deref())?;
        let mut add_space_prefix = false;
        let mut spm_byte_to_id: HashMap<u8, u32> = HashMap::new();
        let mut dna: Option<DnaData> = None;
        let mut ignore_merges = false;
        let mut normalizer = NormalizerOptions::default();
        let mut max_token_len = 0;
        let mut unk_token = 0;
        let mut ugm = UgmData::default();
        let mut remove_extra_whitespaces = false;
        let pre = match model {
            TokenizerModel::None => {
                // The placeholder vocabulary: `vocab_size` entries with no text, no
                // scores, no types and no special roles — the shape llama.cpp builds
                // with `id_to_token.resize(n_tokens)`.
                let n = Self::get_gguf_u32(gguf, "tokenizer.ggml.vocab_size").unwrap_or(0) as usize;
                id_to_token = vec![String::new(); n];
                id_to_score = vec![0.0; n];
                id_to_type = vec![0; n];
                None
            }
            TokenizerModel::Gpt2 | TokenizerModel::HybridDna | TokenizerModel::Whitespace => {
                // F7: the pre-tokenization rule must be one this engine implements.
                let pre = PreTokenizer::from_gguf(
                    Self::get_gguf_str(gguf, "tokenizer.ggml.pre").as_deref(),
                )?;
                if merges.is_empty() {
                    return Err(
                        "tokenizer.ggml.merges is empty; minfer refuses to run a byte-level BPE \
                         with no merge ranks (every piece would silently degrade to single bytes)"
                            .to_string(),
                    );
                }
                // The byte fallback is exact only when every byte has a token.
                // Qwen's vocabularies have all 256; a vocabulary that does not is
                // a load error rather than a lossy fallback.
                // A byte that has no token is acceptable only when it cannot reach the
                // fallback: it never occurs in the vocabulary at all (an invalid UTF-8
                // lead byte — there is nothing to fall back to), or it **starts** an
                // existing token, so a piece beginning with it is matched by that token
                // first. Anything else is still a load error rather than a lossy
                // fallback (ADR-0056; measured on Falcon3-1B, which omits 18 bytes).
                let missing: Vec<u8> = (0u8..=255)
                    .filter(|b| !vocab.contains_key(&byte_to_unicode[b].to_string()))
                    .collect();
                let mut reachable: Vec<u8> = Vec::new();
                for &b in &missing {
                    let mut occurs = false;
                    let mut starts_token = false;
                    for text in id_to_token.iter() {
                        let raw = text.as_bytes();
                        if raw.first() == Some(&b) {
                            starts_token = true;
                        }
                        if raw.contains(&b) {
                            occurs = true;
                        }
                    }
                    if occurs && !starts_token {
                        reachable.push(b);
                    }
                }
                if let Some(first) = reachable.first() {
                    return Err(format!(
                        "vocabulary is not byte-level complete: {} of 256 byte tokens are missing and \
                         cannot be reached by any token (first: <0x{first:02X}>); minfer refuses a \
                         lossy byte fallback",
                        reachable.len()
                    ));
                }
                if matches!(model, TokenizerModel::HybridDna) {
                    // The three tag ids must all be declared, and the k-mer table is
                    // read **before** the marker is erased: the marked spelling is the
                    // lookup key, the bare one is what detokenization emits.
                    let dna_tag = |tag: &str| vocab.get(tag).copied();
                    match (dna_tag("<dna>"), dna_tag("</dna>"), dna_tag("<oov>")) {
                        (Some(begin), Some(end), Some(oov)) => {
                            let mut kmers = HashMap::new();
                            for (id, text) in id_to_token.iter_mut().enumerate() {
                                let Some(bare) = text.strip_suffix(DNA_KMER_MARKER) else {
                                    continue;
                                };
                                kmers.insert(bare.to_string(), id as u32);
                                *text = bare.to_string();
                            }
                            dna = Some(DnaData {
                                begin,
                                end,
                                oov,
                                kmers,
                            });
                        }
                        // llama.cpp falls back to plain BPE when a tag is absent.
                        _ => {}
                    }
                }
                ignore_merges = pre.ignores_merges();
                Some(pre)
            }
            TokenizerModel::Spm => {
                // The SPM merge order is the score array; without it every merge
                // would tie and the split would be arbitrary.
                if id_to_score.iter().all(|&s| s == 0.0) {
                    return Err(
                        "tokenizer.ggml.scores is empty (or all zero) for a SentencePiece \
                         vocabulary; minfer refuses to order the merges arbitrarily"
                            .to_string(),
                    );
                }
                // `<0xXX>` byte tokens: built from the declared BYTE type (6), and
                // required to be complete — the SPM fallback emits one per byte.
                for (id, token) in id_to_token.iter().enumerate() {
                    if id_to_type.get(id).copied().unwrap_or(1) != 6 {
                        continue;
                    }
                    if let Some(b) = parse_spm_byte_token(token) {
                        spm_byte_to_id.insert(b, id as u32);
                    }
                }
                let missing: Vec<u8> = (0u8..=255)
                    .filter(|b| !spm_byte_to_id.contains_key(b))
                    .collect();
                if let Some(first) = missing.first() {
                    return Err(format!(
                        "vocabulary is not byte-level complete: {} of 256 SentencePiece byte \
                         tokens are missing (first: <0x{first:02X}>); minfer refuses a lossy \
                         byte fallback",
                        missing.len()
                    ));
                }
                // llama.cpp's SPM default; the key overrides it.
                add_space_prefix =
                    Self::get_gguf_bool(gguf, "tokenizer.ggml.add_space_prefix").unwrap_or(true);
                None
            }
            TokenizerModel::Rwkv => {
                // RVKV has no `pre` rule, no merges and no scores: the vocabulary
                // *is* the tokenizer, and the trie holds the decoded bytes.
                for (id, text) in id_to_token.iter().enumerate() {
                    let bytes = unescape_rwkv_token(text);
                    ugm.trie.insert_bytes(&bytes, id as u32);
                }
                // The engine answers a byte no entry starts with by emitting the
                // unknown token (llama.cpp's session). Real RVKV vocabularies carry
                // all 256 bytes as single-byte entries, so that path is unreachable
                // — and the artifact does not always declare the role. Require the
                // role only when a byte really has no entry, and refuse that case by
                // naming the byte instead of emitting a made-up id.
                let missing: Vec<u8> = (0u8..=255)
                    .filter(|&b| !ugm.trie.starts_with_byte(b))
                    .collect();
                match Self::get_gguf_u32(gguf, "tokenizer.ggml.unknown_token_id") {
                    Some(id) => {
                        if id as usize >= id_to_token.len() {
                            return Err(format!(
                                "tokenizer.ggml.unknown_token_id = {id} is out of range (the \
                                 vocabulary has {} entries)",
                                id_to_token.len()
                            ));
                        }
                        unk_token = id;
                    }
                    None if missing.is_empty() => {
                        // Unreachable by construction; `rwkv_encode` never emits it.
                        unk_token = 0;
                    }
                    None => {
                        return Err(format!(
                            "this RVKV vocabulary has no token starting with <0x{:02X}> ({} byte(s) \
                             uncovered) and declares no tokenizer.ggml.unknown_token_id, so a byte \
                             of that value has no representation; minfer refuses it rather than \
                             inventing an id",
                            missing[0],
                            missing.len()
                        ));
                    }
                }
                None
            }
            TokenizerModel::Plamo2 => {
                // `model = plamo2` fixes the rule itself: llama.cpp sets the
                // pre-type from the model value, and this artifact declares
                // `pre = "default"`, which is not one of the four rules minfer
                // accepts — so the key is deliberately not read.
                //
                // The DP's objective is the file's own scores and its fallback is
                // the `<0xXX>` byte table; both are mandatory here.
                if id_to_score.len() != id_to_token.len() || id_to_score.is_empty() {
                    return Err(format!(
                        "tokenizer.ggml.scores has {} entries for {} tokens; the plamo2 engine's \
                         objective is the file's own scores, so a mismatched table cannot be run",
                        id_to_score.len(),
                        id_to_token.len()
                    ));
                }
                for (id, token) in id_to_token.iter().enumerate() {
                    if id_to_type.get(id).copied().unwrap_or(1) == 6 {
                        if let Some(b) = parse_spm_byte_token(token) {
                            spm_byte_to_id.insert(b, id as u32);
                        }
                        continue;
                    }
                    // Byte tokens are the fallback, not candidates.
                    ugm.trie.insert_bytes(token.as_bytes(), id as u32);
                }
                let missing: Vec<u8> = (0u8..=255)
                    .filter(|b| !spm_byte_to_id.contains_key(b))
                    .collect();
                if let Some(first) = missing.first() {
                    return Err(format!(
                        "this plamo2 vocabulary declares no <0x{first:02X}> byte token ({} byte(s) \
                         uncovered); the engine's fallback emits one byte token per byte, so it \
                         refuses the file rather than dropping input",
                        missing.len()
                    ));
                }
                None
            }
            TokenizerModel::Gemma4 => {
                // `model = gemma4` **forces** the rule: llama.cpp sets
                // `tokenizer_pre = "gemma4"` from the model value, and the file
                // carries no `pre` key — so the key is not read.
                if merges.is_empty() {
                    return Err(
                        "tokenizer.ggml.merges is empty; minfer refuses to run the gemma4 BPE \
                         with no merge ranks (every piece would silently degrade to single bytes)"
                            .to_string(),
                    );
                }
                // The byte fallback is the `<0xXX>` tokens the vocabulary declares,
                // exactly as for SentencePiece: all 256 are required.
                for (id, token) in id_to_token.iter().enumerate() {
                    if id_to_type.get(id).copied().unwrap_or(1) != 6 {
                        continue;
                    }
                    if let Some(b) = parse_spm_byte_token(token) {
                        spm_byte_to_id.insert(b, id as u32);
                    }
                }
                let missing: Vec<u8> = (0u8..=255)
                    .filter(|b| !spm_byte_to_id.contains_key(b))
                    .collect();
                if let Some(first) = missing.first() {
                    return Err(format!(
                        "vocabulary is not byte-level complete: {} of 256 gemma4 byte tokens are \
                         missing (first: <0x{first:02X}>); minfer refuses a lossy byte fallback",
                        missing.len()
                    ));
                }
                Some(PreTokenizer::Gemma4)
            }
            TokenizerModel::Ugm => {
                // The Viterbi is ordered by the score array; without it every
                // segmentation would tie.
                if id_to_score.iter().all(|&s| s == 0.0) {
                    return Err(
                        "tokenizer.ggml.scores is empty (or all zero) for a unigram vocabulary; \
                         minfer refuses to order the Viterbi arbitrarily"
                            .to_string(),
                    );
                }
                unk_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.unknown_token_id").ok_or(
                    "tokenizer.ggml.unknown_token_id is missing; minfer's unigram engine needs \
                     the unknown id for input it cannot segment"
                        .to_string(),
                )?;
                if unk_token as usize >= id_to_token.len() {
                    return Err(format!(
                        "tokenizer.ggml.unknown_token_id = {unk_token} is out of range (the \
                         vocabulary has {} entries)",
                        id_to_token.len()
                    ));
                }
                // llama.cpp inserts NORMAL, USER_DEFINED and UNUSED entries.
                let mut min_score = f32::MAX;
                for (id, text) in id_to_token.iter().enumerate() {
                    let ty = id_to_type.get(id).copied().unwrap_or(1);
                    if ty == 1 || ty == 4 || ty == 5 {
                        ugm.trie.insert(text, id as u32);
                    }
                    if ty == 4 {
                        ugm.user_defined.insert(text, id as u32);
                    }
                    if ty == 1 {
                        min_score = min_score.min(id_to_score.get(id).copied().unwrap_or(0.0));
                    }
                }
                let min_score = if min_score == f32::MAX {
                    0.0
                } else {
                    min_score
                };
                // llama.cpp's penalty: the unknown transition scores ten below the
                // worst normal entry, so it is chosen only when nothing matches.
                ugm.unknown_token_score = min_score - 10.0;
                // The precompiled charsmap is a byte array: a 4-byte little-endian
                // XCDA blob size, the blob, then NUL-terminated replacements.
                if let Some(bytes) =
                    Self::get_gguf_bytes(gguf, "tokenizer.ggml.precompiled_charsmap")
                {
                    if bytes.len() < 4 {
                        return Err(
                            "tokenizer.ggml.precompiled_charsmap is shorter than its 4-byte \
                             header; minfer refuses to normalise with a truncated table"
                                .to_string(),
                        );
                    }
                    let blob =
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    if 4 + blob > bytes.len() || blob % 4 != 0 {
                        return Err(format!(
                            "tokenizer.ggml.precompiled_charsmap declares a {blob}-byte XCDA blob \
                             that does not fit its {} bytes; refusing a truncated table",
                            bytes.len()
                        ));
                    }
                    ugm.xcda = bytes[4..4 + blob]
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    ugm.prefix_replacements = bytes[4 + blob..].to_vec();
                }
                // llama.cpp: add_space_prefix defaults to false, and the key
                // overrides it; treat_whitespace_as_suffix is always false in the
                // pinned revision, so the prefix form is the only one implemented.
                add_space_prefix =
                    Self::get_gguf_bool(gguf, "tokenizer.ggml.add_space_prefix").unwrap_or(false);
                remove_extra_whitespaces =
                    Self::get_gguf_bool(gguf, "tokenizer.ggml.remove_extra_whitespaces")
                        .unwrap_or(false);
                None
            }
            TokenizerModel::Bert => {
                // BertNormalizer options: both default to `true`
                // (`llama-vocab.h:83`), and `strip_accents` follows `lowercase`
                // unless the file says otherwise (llama.cpp, `llama-vocab.cpp:2635`).
                let lowercase = Self::get_gguf_bool(gguf, "tokenizer.ggml.normalizer.lowercase")
                    .unwrap_or(true);
                let strip_accents =
                    Self::get_gguf_bool(gguf, "tokenizer.ggml.normalizer.strip_accents")
                        .unwrap_or(lowercase);
                normalizer = NormalizerOptions {
                    lowercase,
                    strip_accents,
                };
                // The `<unk>` fallback is the engine's only answer for a word it
                // cannot segment, so the role must be declared and in range.
                unk_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.unknown_token_id").ok_or(
                    "tokenizer.ggml.unknown_token_id is missing; minfer's WordPiece engine needs \
                     the [UNK] id for a word it cannot segment"
                        .to_string(),
                )?;
                if unk_token as usize >= id_to_token.len() {
                    return Err(format!(
                        "tokenizer.ggml.unknown_token_id = {unk_token} is out of range (the \
                         vocabulary has {} entries)",
                        id_to_token.len()
                    ));
                }
                max_token_len = id_to_token.iter().map(String::len).max().unwrap_or(0);
                None
            }
        };

        Ok(Tokenizer {
            id_to_token,
            id_to_score,
            id_to_type,
            vocab,
            merges,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
            model,
            pre,
            add_space_prefix,
            spm_byte_to_id,
            dna,
            normalizer,
            max_token_len,
            unk_token,
            ugm,
            ignore_merges,
            remove_extra_whitespaces,
            special_by_first,
            bos_token,
            eos_token,
        })
    }

    /// A GGUF metadata string value (F7: `tokenizer.ggml.model` / `.pre`).
    fn get_gguf_str(gguf: &GgufContext, key: &str) -> Option<String> {
        gguf.kv
            .iter()
            .find(|kv| kv.key == key && kv.type_ == GgufType::String)
            .map(|kv| kv.get_val_str(0).to_string())
    }

    /// A GGUF metadata byte array (int8 or uint8), if the key is present with
    /// that type and is an array.
    fn get_gguf_bytes(gguf: &GgufContext, key: &str) -> Option<Vec<u8>> {
        let kv = gguf.kv.iter().find(|kv| {
            kv.key == key && kv.is_array && matches!(kv.type_, GgufType::Int8 | GgufType::Uint8)
        })?;
        Some(
            (0..kv.get_ne())
                .map(|i| {
                    if kv.type_ == GgufType::Int8 {
                        kv.get_val_i8(i) as u8
                    } else {
                        kv.get_val_u8(i)
                    }
                })
                .collect(),
        )
    }

    /// A GGUF metadata boolean, if the key is present with that type.
    fn get_gguf_bool(gguf: &GgufContext, key: &str) -> Option<bool> {
        gguf.kv
            .iter()
            .find(|kv| kv.key == key && kv.type_ == GgufType::Bool && !kv.is_array)
            .map(|kv| kv.get_val_bool(0))
    }

    fn get_gguf_u32(gguf: &GgufContext, key: &str) -> Option<u32> {
        for kv in &gguf.kv {
            if kv.key == key && kv.type_ == GgufType::Uint32 {
                return Some(kv.get_val_u32(0));
            }
            if kv.key == key && kv.type_ == GgufType::Int32 {
                return Some(kv.get_val_i32(0) as u32);
            }
            if kv.key == key && kv.type_ == GgufType::Int64 {
                return Some(kv.get_val_i64(0) as u32);
            }
            if kv.key == key && kv.type_ == GgufType::Uint64 {
                return Some(kv.get_val_u64(0) as u32);
            }
        }
        None
    }

    /// BPE encode a single pre-token (already byte-encoded).
    ///
    /// The merge loop always starts from single characters. There is deliberately
    /// **no** "whole piece is in the vocabulary" shortcut: a vocabulary entry is
    /// not necessarily reachable through merges, and returning it directly gave a
    /// different — wrong — split (Qwen3.5's `क्` + `ष` merge into a vocab entry
    /// that has no rank, and transformers splits it). llama.cpp's shortcut is
    /// gated on `tokenizer_ignore_merges`, which none of the Qwen rules set.
    fn bpe_encode(&self, token: &str) -> Vec<u32> {
        // Split into characters
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        loop {
            // Find the best merge (lowest rank)
            let mut best_rank: Option<usize> = None;
            let mut best_idx: Option<usize> = None;

            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = Some(i);
                    }
                }
            }

            if best_idx.is_none() {
                break;
            }

            // Merge at best_idx
            let idx = best_idx.unwrap();
            let merged = format!("{}{}", word[idx], word[idx + 1]);
            word.splice(idx..=idx + 1, std::iter::once(merged));
        }

        // Look up each merged piece; a piece with no token is emitted byte by
        // byte (byte-level BPE maps every byte to a vocabulary entry — `load`
        // refuses a vocabulary where that is not true, so this never drops
        // bytes silently, unlike the old `unwrap_or(0)`).
        let mut out = Vec::with_capacity(word.len());
        for piece in &word {
            match self.vocab.get(piece) {
                Some(&id) => out.push(id),
                None => out.extend(self.byte_fallback(piece)),
            }
        }
        out
    }

    /// One token per byte of a byte-encoded piece.
    ///
    /// `Tokenizer::load` verified that every one of the 256 byte tokens exists,
    /// so a miss here is an invariant violation and aborts with the value —
    /// the engine never substitutes a wrong id.
    fn byte_fallback(&self, piece: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(piece.len());
        for c in piece.chars() {
            let single = c.to_string();
            match self.vocab.get(&single) {
                Some(&id) => out.push(id),
                None => panic!(
                    "tokenizer invariant violated: no token for byte-encoded character {c:?} \
                     (U+{:04X}); Tokenizer::load verifies all 256 byte tokens",
                    c as u32
                ),
            }
        }
        out
    }

    /// Pre-tokenize with the rule selected from `tokenizer.ggml.pre`, then
    /// byte-encode each piece and run BPE on it.
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        let pre = self
            .pre
            .expect("a gpt2 tokenizer always has a pre-tokenization rule");
        let mut result = Vec::new();
        for piece in pre.split(text) {
            let encoded = byte_encode(piece, &self.byte_to_unicode);
            // `ignore_merges`: a piece that is itself a vocabulary entry is emitted
            // whole, before the merge ranks (`llama-vocab.cpp:626-630`).
            if self.ignore_merges {
                if let Some(&id) = self.vocab.get(&encoded) {
                    result.push(id);
                    continue;
                }
            }
            result.extend(self.bpe_encode(&encoded));
        }
        result
    }

    /// Tokenize text into token IDs.
    ///
    /// The text is first normalized to NFC when the rule's reference
    /// `tokenizer.json` declares one ([`PreTokenizer::applies_nfc`]), so for
    /// those families a decomposed input tokenizes exactly like its composed
    /// form ([#132](https://github.com/yusiwen/minfer/issues/132) S1, refined by
    /// ADR-0044) — and for a family that declares none (`llama-bpe`) it is left
    /// exactly as it arrived.
    ///
    /// Special tokens are matched as single ids *before* BPE — earliest position
    /// wins, longest text wins at the same position. The set is what the file
    /// declares ([`Tokenizer::load`], #132 S2): `tokenizer.ggml.token_type`
    /// 2/3/4 plus every id a `tokenizer.ggml.*_token_id` role names. This is what
    /// makes special-token templates work (e.g. DeepSeek-R1's `<｜User｜>` /
    /// `<think>`), matching llama.cpp — whose converter marks those type 3.
    /// The **checked** text→id entry point ([#586]). Every engine that has a
    /// vocabulary delegates to [`Tokenizer::encode`]; `model = none` is a named
    /// refusal, because a vocabulary-less file has nothing to tokenize text with
    /// (llama.cpp aborts there — `llama-vocab.cpp:3587` — and an empty vector would
    /// be a silent lie). Production callers that turn user text into tokens use this.
    ///
    /// [#586]: https://github.com/yusiwen/minfer/issues/586
    pub fn encode_checked(&self, text: &str) -> Result<Vec<u32>, String> {
        if self.model == TokenizerModel::None {
            return Err(format!(
                "this file declares `tokenizer.ggml.model = {}` (a vocabulary-less model with {} \
                 placeholder token(s)): it has no vocabulary, so no text can be turned into \
                 tokens — supply token ids directly instead",
                "none/no_vocab",
                self.id_to_token.len()
            ));
        }
        Ok(self.encode(text))
    }

    /// The **checked** id→text entry point ([#586]): `model = none` has no pieces to
    /// emit, and an empty result would be indistinguishable from "the ids decoded to
    /// nothing".
    ///
    /// [#586]: https://github.com/yusiwen/minfer/issues/586
    pub fn decode_checked(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        if self.model == TokenizerModel::None {
            return Err(format!(
                "this file declares `tokenizer.ggml.model = {}` (a vocabulary-less model): its ids \
                 have no text representation to decode",
                "none/no_vocab"
            ));
        }
        Ok(self.decode_bytes(ids))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalized(text);
        let text: &str = normalized.as_ref();
        let mut result = Vec::new();
        let mut remaining = text;
        // SentencePiece prefixes the first raw-text segment (and every segment
        // after a special token) with a space — llama.cpp's `is_prev_special`,
        // which starts `true` (`src/llama-vocab.cpp:3413`).
        let mut prev_special = true;

        loop {
            // Find the earliest position where any special token starts.
            let mut earliest: Option<(usize, u32, usize)> = None; // (byte_pos, id, byte_len)
            'scan: for (ci, ch) in remaining.char_indices() {
                if let Some(group) = self.special_by_first.get(&ch) {
                    let rest = &remaining[ci..];
                    for (pat, id) in group {
                        if rest.starts_with(pat.as_str()) {
                            earliest = Some((ci, *id, pat.len()));
                            break 'scan; // group is longest-first; earliest char wins
                        }
                    }
                }
            }

            if let Some((pos, id, len)) = earliest {
                // Encode text before the special token
                if pos > 0 {
                    result.extend(self.encode_text(&remaining[..pos], prev_special));
                }
                result.push(id);
                remaining = &remaining[pos + len..];
                prev_special = true;
            } else {
                // No more special tokens, encode the rest
                result.extend(self.encode_text(remaining, prev_special));
                break;
            }
        }

        result
    }

    /// Tokenize text as pure byte-level BPE, *without* the special-token
    /// pre-scan [`Tokenizer::encode`] performs.
    ///
    /// This is llama.cpp's `tokenize(text, /*add_special*/ false,
    /// /*parse_special*/ false)`: a special token's **text** is split by the
    /// pre-tokenizer and BPE like any other text, and never collapses to the
    /// special token's id. DRY breaker tails need this — a tail is a fragment
    /// of a breaker string, not a prompt, so absorbing a special token there
    /// would produce a sequence the engine could never actually emit.
    ///
    /// NFC normalization applies here too: it belongs to the tokenizer, not to
    /// the special-token pre-scan.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalized(text);
        // No special-token scan, but the SPM space prefix still applies: it is
        // part of the tokenizer, not of the scan (llama.cpp applies it on the
        // `parse_special = false` path too).
        self.encode_text(normalized.as_ref(), true)
    }

    /// Encode one raw-text segment — the piece between two special tokens — with
    /// the engine this file selected.
    ///
    /// `prev_special` is `true` for the first segment and after every special
    /// token; SentencePiece uses it to decide the leading-space prefix.
    fn encode_text(&self, text: &str, prev_special: bool) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        match self.model {
            // `model = none` has no vocabulary; `encode_checked` refuses it before
            // this point, and a direct `encode` call gets an empty list.
            TokenizerModel::None => Vec::new(),
            TokenizerModel::Gpt2 | TokenizerModel::Whitespace => self.encode_bpe(text),
            TokenizerModel::HybridDna => self.hybriddna_encode(text),
            TokenizerModel::Bert => self.wpm_encode(text),
            TokenizerModel::Ugm => self.ugm_encode(text),
            TokenizerModel::Gemma4 => self.gemma4_encode(text),
            TokenizerModel::Rwkv => self.rwkv_encode(text),
            TokenizerModel::Plamo2 => self.plamo2_encode(text),
            TokenizerModel::Spm => {
                let mut escaped = String::with_capacity(text.len() + SPM_SPACE.len_utf8());
                if self.add_space_prefix && prev_special {
                    escaped.push(SPM_SPACE);
                }
                for c in text.chars() {
                    escaped.push(if c == ' ' { SPM_SPACE } else { c });
                }
                self.spm_encode(&escaped)
            }
        }
    }

    /// SentencePiece encode a `▁`-escaped segment: llama.cpp's
    /// `llm_tokenizer_spm_session` (`src/llama-vocab.cpp:114`).
    ///
    /// Symbols start as single characters. A bigram whose concatenation is a
    /// vocabulary entry is queued with that entry's **score**; the highest score
    /// is merged first, and every merge re-queues the new neighbours. A bigram
    /// whose sides changed since it was queued is stale and skipped. The final
    /// symbol walk emits a vocabulary entry where it can, re-splits through the
    /// recorded merge pair where it cannot, and otherwise falls back to one
    /// `<0xXX>` byte token per byte.
    fn spm_encode(&self, text: &str) -> Vec<u32> {
        use std::collections::BinaryHeap;

        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut symbols: Vec<SpmSymbol> = Vec::with_capacity(n);
        for (i, c) in chars.iter().enumerate() {
            let text = c.to_string();
            symbols.push(SpmSymbol {
                n: text.len(),
                text,
                prev: if i == 0 { None } else { Some(i - 1) },
                next: if i + 1 == n { None } else { Some(i + 1) },
            });
        }

        // The concatenation → the pair of symbol texts it was built from, for the
        // re-split fallback (`rev_merge` in llama.cpp's session).
        let mut rev_merge: HashMap<String, (String, String)> = HashMap::new();
        let mut queue: BinaryHeap<SpmBigram> = BinaryHeap::new();

        let try_add = |left: usize,
                       right: usize,
                       symbols: &Vec<SpmSymbol>,
                       vocab: &HashMap<String, u32>,
                       scores: &[f32],
                       rev_merge: &mut HashMap<String, (String, String)>,
                       queue: &mut BinaryHeap<SpmBigram>| {
            let mut joined = String::with_capacity(symbols[left].n + symbols[right].n);
            joined.push_str(&symbols[left].text);
            joined.push_str(&symbols[right].text);
            if let Some(&id) = vocab.get(&joined) {
                if (id as usize) < scores.len() {
                    queue.push(SpmBigram {
                        score: scores[id as usize],
                        left,
                        right,
                        size: joined.len(),
                    });
                    rev_merge.insert(
                        joined,
                        (symbols[left].text.clone(), symbols[right].text.clone()),
                    );
                }
            }
        };

        for i in 1..n {
            try_add(
                i - 1,
                i,
                &symbols,
                &self.vocab,
                &self.id_to_score,
                &mut rev_merge,
                &mut queue,
            );
        }

        while let Some(bigram) = queue.pop() {
            let (left, right) = (bigram.left, bigram.right);
            if symbols[left].n == 0
                || symbols[right].n == 0
                || symbols[left].n + symbols[right].n != bigram.size
            {
                continue; // one side grew (or died) since this bigram was queued
            }
            let right_text = std::mem::take(&mut symbols[right].text);
            symbols[left].n += symbols[right].n;
            symbols[left].text.push_str(&right_text);
            symbols[right].n = 0;
            symbols[right].text.clear();

            let next = symbols[right].next;
            symbols[left].next = next;
            if let Some(nx) = next {
                symbols[nx].prev = Some(left);
            }
            if let Some(prev) = symbols[left].prev {
                try_add(
                    prev,
                    left,
                    &symbols,
                    &self.vocab,
                    &self.id_to_score,
                    &mut rev_merge,
                    &mut queue,
                );
            }
            if let Some(nx) = symbols[left].next {
                try_add(
                    left,
                    nx,
                    &symbols,
                    &self.vocab,
                    &self.id_to_score,
                    &mut rev_merge,
                    &mut queue,
                );
            }
        }

        let mut out = Vec::new();
        let mut i = Some(0usize);
        while let Some(idx) = i {
            self.resegment(&symbols[idx].text, &rev_merge, &mut out);
            i = symbols[idx].next;
        }
        out
    }

    /// Emit `text` as tokens: a vocabulary entry, else its recorded merge pair,
    /// else one `<0xXX>` byte token per byte.
    fn resegment(
        &self,
        text: &str,
        rev_merge: &HashMap<String, (String, String)>,
        out: &mut Vec<u32>,
    ) {
        if let Some(&id) = self.vocab.get(text) {
            out.push(id);
            return;
        }
        if text.is_empty() {
            return;
        }
        if let Some((left, right)) = rev_merge.get(text) {
            // A symbol that is not itself a vocabulary entry but was built from a
            // pair we recorded: re-split it (both halves are strictly shorter).
            self.resegment(left, rev_merge, out);
            self.resegment(right, rev_merge, out);
            return;
        }
        for b in text.bytes() {
            match self.spm_byte_to_id.get(&b) {
                Some(&id) => out.push(id),
                None => panic!(
                    "tokenizer invariant violated: no SentencePiece byte token for <0x{b:02X}>;                      Tokenizer::load verifies all 256"
                ),
            }
        }
    }

    /// HybridDNA encode (`llm_tokenizer_hybriddna_session`, `llama-vocab.cpp:1640`):
    /// the `<dna>`…`</dna>` blocks are cut into 6-mers, everything else is the plain
    /// `gpt2` engine. Without the three tag ids the engine is plain BPE.
    fn hybriddna_encode(&self, text: &str) -> Vec<u32> {
        const OPEN: &str = "<dna>";
        const CLOSE: &str = "</dna>";
        let Some(dna) = self.dna.as_ref() else {
            return self.encode_bpe(text);
        };
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < text.len() {
            let Some(start) = text[pos..].find(OPEN).map(|i| i + pos) else {
                out.extend(self.encode_bpe(&text[pos..]));
                break;
            };
            if start > pos {
                out.extend(self.encode_bpe(&text[pos..start]));
            }
            out.push(dna.begin);
            let content_start = start + OPEN.len();
            let close = text[content_start..].find(CLOSE).map(|i| i + content_start);
            let content_end = close.unwrap_or(text.len());
            out.extend(self.hybriddna_kmers(&text[content_start..content_end], dna));
            match close {
                Some(end) => {
                    out.push(dna.end);
                    pos = end + CLOSE.len();
                }
                // No closing tag: the rest of the input is the block, as in llama.cpp.
                None => break,
            }
        }
        out
    }

    /// `emit_dna_kmers`: upper-case the ASCII letters, cut into 6-byte k-mers, look
    /// each up, and pad the tail with `A` to six bytes.
    fn hybriddna_kmers(&self, raw: &str, dna: &DnaData) -> Vec<u32> {
        const K: usize = 6;
        let seq: Vec<u8> = raw
            .bytes()
            .map(|b| if b.is_ascii_lowercase() { b - 32 } else { b })
            .collect();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + K <= seq.len() {
            let kmer = String::from_utf8_lossy(&seq[i..i + K]).to_string();
            out.push(*dna.kmers.get(&kmer).unwrap_or(&dna.oov));
            i += K;
        }
        if i < seq.len() {
            let mut kmer = String::from_utf8_lossy(&seq[i..]).to_string();
            while kmer.len() < K {
                kmer.push('A');
            }
            out.push(*dna.kmers.get(&kmer).unwrap_or(&dna.oov));
        }
        out
    }

    /// PLaMo-2 encode: a backward DP over codepoints from the end of the text
    /// (`llm_tokenizer_plamo2`, `llama-vocab.cpp:1360-1592`).
    ///
    /// The candidates at `i` are every non-byte piece that is a prefix of
    /// `text[i..]`, longest first, plus the one-codepoint fallback row:
    /// `s = scores[i + piece_length] - round(score × 1e4)`, and the **first**
    /// candidate to lower `scores[i]` wins (a strict `<`, so ties keep the longest).
    /// A chosen fallback emits one `<0xXX>` byte token per UTF-8 byte, never `<unk>`,
    /// and a leading BOM is stripped. llama.cpp indexes the candidate set with a
    /// suffix automaton; this direct walk is measured equivalent to it (ADR-0050).
    fn plamo2_encode(&self, text: &str) -> Vec<u32> {
        const INVALID_SCORE: i64 = -20_000_000;
        const UNKNOWN_SCORE: i64 = -10_000_000;
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let cps: Vec<char> = text.chars().collect();
        let n = cps.len();
        if n == 0 {
            return Vec::new();
        }
        let mut scores = vec![1i64 << 60; n + 1];
        scores[n] = 0;
        let mut take: Vec<(usize, Option<u32>)> = vec![(1, None); n];
        for i in (0..n).rev() {
            let start = text
                .char_indices()
                .nth(i)
                .map(|(b, _)| b)
                .unwrap_or(text.len());
            let bytes = text.as_bytes();
            let mut candidates: Vec<(usize, u32)> = Vec::new();
            if !self.ugm.trie.nodes.is_empty() {
                let mut node = 0u32;
                let mut k = start;
                while k < bytes.len() {
                    match self.ugm.trie.step(node, bytes[k]) {
                        Some(next) => {
                            node = next;
                            k += 1;
                        }
                        None => break,
                    }
                    if let Some(id) = self.ugm.trie.terminal(node) {
                        candidates.push((text[start..k].chars().count(), id));
                    }
                }
            }
            for &(piece_len, id) in candidates.iter().rev() {
                let score = self.plamo2_score_int(id);
                if score <= INVALID_SCORE {
                    continue;
                }
                let s = scores[i + piece_len] - score;
                if s < scores[i] {
                    scores[i] = s;
                    take[i] = (piece_len, Some(id));
                }
            }
            let s = scores[i + 1] - UNKNOWN_SCORE;
            if s < scores[i] {
                scores[i] = s;
                take[i] = (1, None);
            }
        }
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < n {
            let (piece_len, id) = take[pos];
            match id {
                Some(id) => out.push(id),
                None => {
                    let mut buf = [0u8; 4];
                    for &b in cps[pos].encode_utf8(&mut buf).as_bytes() {
                        if let Some(&byte_id) = self.spm_byte_to_id.get(&b) {
                            out.push(byte_id);
                        }
                    }
                }
            }
            pos += piece_len;
        }
        out
    }

    /// `round(score × 1e4)` as `llama.cpp` quantizes it, or `INVALID_SCORE` for a
    /// non-finite entry (which the DP skips).
    fn plamo2_score_int(&self, id: u32) -> i64 {
        match self.id_to_score.get(id as usize) {
            Some(score) if score.is_finite() => (score * 1e4).round() as i64,
            _ => -20_000_000,
        }
    }

    /// RVKV encode: a greedy longest match over the vocabulary trie, byte by byte
    /// (`llm_tokenizer_rwkv_session`, `llama-vocab.cpp:1285`). A byte that starts
    /// no entry emits the unknown token and advances one byte.
    fn rwkv_encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut position = 0usize;
        while position < bytes.len() {
            match self.ugm.trie.longest_prefix_bytes(&bytes[position..]) {
                Some((id, len)) => {
                    out.push(id);
                    position += len;
                }
                None => {
                    out.push(self.unk_token);
                    position += 1;
                }
            }
        }
        out
    }

    /// Gemma-4 encode: the normalizer replaces spaces with `▁`, the rule splits on
    /// newline runs, and BPE runs on the raw UTF-8 pieces — **no** byte-to-unicode
    /// step, which is what `model = gemma4` selects (`byte_encode = false`).
    fn gemma4_encode(&self, text: &str) -> Vec<u32> {
        let escaped: String = text
            .chars()
            .map(|c| if c == ' ' { SPM_SPACE } else { c })
            .collect();
        let pre = self
            .pre
            .expect("gemma4 always has its rule (it comes from the model value)");
        let mut out = Vec::new();
        for piece in pre.split(&escaped) {
            // A piece that is nothing but newlines is emitted whole when the
            // vocabulary has it (llama.cpp's gemma4 fix, PR 21343).
            if piece.chars().all(|c| c == '\n') {
                if let Some(&id) = self.vocab.get(piece) {
                    out.push(id);
                    continue;
                }
            }
            out.extend(self.bpe_encode_raw(piece));
        }
        out
    }

    /// BPE over a raw UTF-8 piece: the merge loop on characters, a vocabulary
    /// lookup per merged piece, and one `<0xXX>` byte token per byte otherwise.
    fn bpe_encode_raw(&self, piece: &str) -> Vec<u32> {
        let mut word: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: Option<usize> = None;
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else { break };
            let merged = format!("{}{}", word[idx], word[idx + 1]);
            word.splice(idx..=idx + 1, std::iter::once(merged));
        }

        let mut out = Vec::with_capacity(word.len());
        for w in &word {
            match self.vocab.get(w) {
                Some(&id) => out.push(id),
                None => {
                    for b in w.as_bytes() {
                        match self.spm_byte_to_id.get(b) {
                            Some(&id) => out.push(id),
                            None => panic!(
                                "tokenizer invariant violated: no byte token for <0x{b:02X}>; \
                                 Tokenizer::load verifies all 256"
                            ),
                        }
                    }
                }
            }
        }
        out
    }

    /// The byte length of the UTF-8 sequence that starts with `b` (llama.cpp's
    /// `unicode_len_utf8`; a continuation byte counts as one).
    fn utf8_len(b: u8) -> usize {
        if b < 0x80 {
            1
        } else if b >> 5 == 0b110 {
            2
        } else if b >> 4 == 0b1110 {
            3
        } else if b >> 3 == 0b11110 {
            4
        } else {
            1
        }
    }

    /// Unigram (`t5`) encode a raw segment: llama.cpp's `llm_tokenizer_ugm_session`
    /// (`llama-vocab.cpp:963`).
    ///
    /// The text is normalized first (charsmap replacements + whitespace rules),
    /// then a Viterbi walk over the vocabulary trie finds the highest-scoring
    /// segmentation: at every code-point boundary, every token that starts there
    /// is a transition, and when no token covers the whole code point the unknown
    /// token is one too, at `min_score - 10`. The walk back merges a run of
    /// consecutive unknowns into one token.
    fn ugm_encode(&self, text: &str) -> Vec<u32> {
        let normalized = self.ugm_normalize(text);
        let bytes = normalized.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }

        // `best[i]` = the token ending at `i`, where it started, and the best score
        // sum reaching `i`. Seeded the way llama.cpp seeds it: everything unknown
        // at `f64::MIN`, the start at 0 — so an offset nothing reaches behaves
        // exactly like the C++ default-initialized entry.
        let mut best: Vec<(u32, usize, f64)> = vec![(self.unk_token, 0, f64::MIN); n + 1];
        best[0] = (self.unk_token, 0, 0.0);

        let mut offset = 0usize;
        while offset < n {
            let cp_len = Self::utf8_len(bytes[offset]).min(n - offset);
            let current = best[offset].2;
            let mut single_codepoint = false;
            for (id, end) in self.ugm.trie.matches(bytes, offset) {
                if end - offset == cp_len {
                    single_codepoint = true;
                }
                let score = if self.id_to_type.get(id as usize).copied() == Some(4) {
                    // A user-defined token scores 0, so it is preferred over the
                    // negative-scored normal entries (llama.cpp).
                    0.0
                } else {
                    self.id_to_score.get(id as usize).copied().unwrap_or(0.0) as f64
                };
                let challenger = current + score;
                if challenger > best[end].2 {
                    best[end] = (id, offset, challenger);
                }
            }
            if !single_codepoint {
                let end = offset + cp_len;
                let challenger = current + self.ugm.unknown_token_score as f64;
                if challenger > best[end].2 {
                    best[end] = (self.unk_token, offset, challenger);
                }
            }
            offset += cp_len;
        }

        let mut out = Vec::new();
        let mut at = n;
        let mut prev_unknown = false;
        loop {
            let (id, prev, _) = best[at];
            let unknown = id == self.unk_token;
            if !(prev_unknown && unknown) {
                out.push(id);
            }
            if prev == 0 {
                break;
            }
            prev_unknown = unknown;
            at = prev;
        }
        out.reverse();
        out
    }

    /// The unigram whitespace and charsmap normalization
    /// (`llm_tokenizer_ugm_session::normalize`, `llama-vocab.cpp:1071`).
    ///
    /// A space becomes `▁`; `add_space_prefix` prepends one when the text starts
    /// with a non-space, and `remove_extra_whitespaces` collapses a run of spaces
    /// into one (it also makes them part of the following word).
    fn ugm_normalize(&self, input: &str) -> String {
        let shall_prepend_space = self.add_space_prefix;
        let shall_merge_spaces = self.remove_extra_whitespaces;
        let mut out = String::with_capacity(input.len() * 3);
        let mut is_space_prepended = false;
        let mut processing_non_ws = false;
        let mut offset = 0usize;
        while offset < input.len() {
            let (replacement, consumed) = self.ugm_normalize_prefix(input, offset);
            if consumed == 0 {
                break;
            }
            for c in replacement.chars() {
                if c != ' ' {
                    if !processing_non_ws {
                        processing_non_ws = true;
                        if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                            out.push(SPM_SPACE);
                            is_space_prepended = true;
                        }
                    }
                    out.push(c);
                } else {
                    processing_non_ws = false;
                    if !shall_merge_spaces {
                        out.push(SPM_SPACE);
                    }
                }
            }
            offset += consumed;
        }
        out
    }

    /// The longest normalization of `input[offset..]` and how much input it
    /// consumed: a USER_DEFINED token, else a charmap prefix replacement, else one
    /// code point unchanged (`llm_tokenizer_ugm_session::normalize_prefix`).
    fn ugm_normalize_prefix(&self, input: &str, offset: usize) -> (String, usize) {
        // A declared USER_DEFINED token is kept whole.
        if let Some((_, len)) = self.ugm.user_defined.longest_prefix(&input[offset..]) {
            return (input[offset..offset + len].to_string(), len);
        }

        if !self.ugm.xcda.is_empty() {
            let bytes = input.as_bytes();
            // BASE of the root node, then `BASE[s] ^ c` per byte: a node is a child
            // only when its LCHECK equals the byte, and its LEAF bit marks the
            // nodes whose BASE holds the replacement's offset.
            let base = |packed: u32| (packed >> 10) << ((packed & (1 << 9)) >> 6);
            let mut node = base(self.ugm.xcda[0]);
            let mut longest_len = 0usize;
            let mut longest_off = 0usize;
            for (i, &c) in bytes[offset..].iter().enumerate() {
                if c == 0 {
                    break;
                }
                node ^= c as u32;
                let Some(&packed) = self.ugm.xcda.get(node as usize) else {
                    // A truncated array stops the match rather than panicking; the
                    // loader has already checked the header and the blob size.
                    break;
                };
                if packed & ((1 << 31) | 0xff) != c as u32 {
                    break;
                }
                let is_leaf = (packed >> 8) & 1 == 1;
                node ^= base(packed);
                if is_leaf {
                    longest_len = i + 1;
                    let value = self.ugm.xcda.get(node as usize).copied().unwrap_or(0);
                    longest_off = (value & ((1 << 31) - 1)) as usize;
                }
            }
            if longest_len > 0 {
                if let Some(rest) = self.ugm.prefix_replacements.get(longest_off..) {
                    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                    return (
                        String::from_utf8_lossy(&rest[..end]).into_owned(),
                        longest_len,
                    );
                }
                // A replacement offset outside the table: leave the input alone.
                return (input[offset..offset + longest_len].to_string(), longest_len);
            }
        }

        // No replacement: one code point, unchanged.
        match input[offset..].chars().next() {
            Some(c) => (c.to_string(), c.len_utf8()),
            None => (String::new(), 0),
        }
    }

    /// WordPiece encode a raw segment: llama.cpp's `llm_tokenizer_wpm_session`
    /// (`llama-vocab.cpp:777`).
    ///
    /// Each preprocessed word gets the phantom `▁` prefix and is segmented by
    /// **greedy longest match**; a word with no match at all — or one that stops
    /// matching part way — emits a single `<unk>` and nothing else. `max_token_len`
    /// bounds the match length so the scan is not quadratic in the vocabulary.
    fn wpm_encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in wpm_preprocess(text, self.normalizer) {
            if word.is_empty() {
                continue;
            }
            // The phantom space is part of the vocabulary's shape: the converter
            // rewrites a BERT vocabulary into `▁`-prefixed entries, so every word
            // starts with one.
            let word1 = format!("{SPM_SPACE}{word}");
            let n = word1.len();
            let start = out.len();
            let mut i = 0usize;
            while i < n {
                let mut matched = false;
                let mut j = (i + self.max_token_len + 1).min(n);
                while j > i {
                    if word1.is_char_boundary(i) && word1.is_char_boundary(j) {
                        if let Some(&id) = self.vocab.get(&word1[i..j]) {
                            out.push(id);
                            matched = true;
                            break;
                        }
                    }
                    j -= 1;
                }
                if !matched {
                    // Drop what this word produced and report it as unknown —
                    // llama.cpp's `output.resize(current_tokens)` + break.
                    out.truncate(start);
                    break;
                }
                i = j;
            }
            if out.len() == start {
                out.push(self.unk_token);
            }
        }
        out
    }

    /// Whether the input is normalized to NFC: true when the file's
    /// pre-tokenizer rule declares one (ADR-0044). SentencePiece has no rule, and
    /// the SPM families this engine accepts declare none.
    pub fn applies_nfc(&self) -> bool {
        self.pre.map_or(false, |pre| pre.applies_nfc())
    }

    /// The text this tokenizer actually tokenizes: NFC when the rule declares it,
    /// the input itself otherwise.
    fn normalized<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.applies_nfc() {
            nfc::nfc(text)
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// Every token sequence that *overlaps* `breaker` — llama.cpp's
    /// `get_overlapping_token_sequences` (`src/llama-sampler.cpp`).
    ///
    /// For each token whose text overlaps the breaker, the result holds
    /// `[head, tail...]`: `head` is the token in whose text the breaker
    /// *starts*, and the tail is the tokenization of the breaker's remaining
    /// text (which continues in the tokens that follow `head`). Two branches,
    /// both from llama.cpp:
    ///
    /// 1. the breaker occurs **anywhere** inside the token's text — the token
    ///    alone is the whole occurrence, so the entry is `[head]` (an empty
    ///    tail). This is the case `apply_dry`'s single-token-breaker exemption
    ///    reads, which is why the entry must stay length 1 and must not gain a
    ///    trailing empty element.
    /// 2. otherwise, for **every** byte position where the breaker's first byte
    ///    occurs in the token's text, match the breaker from that position —
    ///    the match is allowed to run *past the end* of the token — and
    ///    tokenize the rest of the breaker string as the tail. A position whose
    ///    match does not reach the token's end is not a match.
    ///
    /// Matching is on **bytes** (a token's detokenized form is bytes), and the
    /// tails are tokenized with [`Tokenizer::encode_raw`]. Matching is *not*
    /// normalized — the vocabulary's own text is the reference, exactly as in
    /// llama.cpp — so a non-NFC breaker matches nothing and is refused by name
    /// rather than silently matched on normalized text; the tails do go through
    /// NFC, which is the tokenizer's normalizer for every text it tokenizes.
    ///
    /// Refusals, each naming the value (llama.cpp warns and skips or truncates
    /// instead — minfer never silently drops or cuts a requested breaker):
    /// an empty breaker, one longer than [`MAX_BREAKER_CHARS`], a tail longer
    /// than [`MAX_BREAKER_TAIL_TOKENS`], a breaker that matches no token at
    /// all, and a match that would have to split a UTF-8 character to form its
    /// tail (llama.cpp slices the bytes and tokenizes the invalid result).
    pub fn overlapping_token_sequences(&self, breaker: &str) -> Result<Vec<Vec<u32>>, String> {
        if breaker.is_empty() {
            return Err(
                "a DRY sequence breaker must not be empty (an empty breaker cannot be matched)"
                    .to_string(),
            );
        }
        let n_chars = breaker.chars().count();
        if n_chars > MAX_BREAKER_CHARS {
            return Err(format!(
                "the DRY sequence breaker {breaker:?} is {n_chars} characters long, over the \
                 {MAX_BREAKER_CHARS}-character limit (llama.cpp truncates it silently; minfer \
                 refuses rather than sampling with a breaker you did not ask for)"
            ));
        }

        let needle = breaker.as_bytes();
        let mut out: Vec<Vec<u32>> = Vec::new();
        for id in 0..self.id_to_token.len() as u32 {
            let word = self.decode_bytes(&[id]);
            if word.is_empty() {
                continue; // a token with no text cannot overlap anything
            }
            if word.windows(needle.len()).any(|w| w == needle) {
                // Branch 1: the whole breaker sits inside this token.
                out.push(vec![id]);
                continue;
            }
            // Branch 2: the breaker may *start* inside this token and continue
            // in the tokens that follow it. Only a match that reaches the end
            // of this token's text can continue into the next token.
            let mut from = 0usize;
            while let Some(pos) = word[from..]
                .iter()
                .position(|&b| b == needle[0])
                .map(|p| p + from)
            {
                from = pos + 1;
                let mut i = 1usize;
                while i < needle.len() && pos + i < word.len() && word[pos + i] == needle[i] {
                    i += 1;
                }
                if i != needle.len() && pos + i != word.len() {
                    continue; // stopped on a mismatch, not at the token's end
                }
                let tail = std::str::from_utf8(&needle[i..]).map_err(|_| {
                    format!(
                        "the DRY sequence breaker {breaker:?} can only overlap the vocabulary by \
                         splitting a UTF-8 character (llama.cpp would slice the raw bytes here); \
                         refusing to resolve it"
                    )
                })?;
                let tail_ids = self.encode_raw(tail);
                if tail_ids.len() > MAX_BREAKER_TAIL_TOKENS {
                    return Err(format!(
                        "the DRY sequence breaker {breaker:?} needs a {} token tail after token \
                         {id}, over the {MAX_BREAKER_TAIL_TOKENS}-token limit (llama.cpp truncates \
                         the tail silently; minfer refuses)",
                        tail_ids.len()
                    ));
                }
                let mut seq = Vec::with_capacity(tail_ids.len() + 1);
                seq.push(id);
                seq.extend(tail_ids);
                if !out.contains(&seq) {
                    out.push(seq);
                }
            }
        }

        if out.is_empty() {
            return Err(format!(
                "the DRY sequence breaker {breaker:?} overlaps no token in this vocabulary, so it \
                 could never be matched; refusing it rather than ignoring it"
            ));
        }
        Ok(out)
    }

    /// Resolve a list of DRY breaker **strings** into the token sequences
    /// `SamplerConfig::dry_breakers` takes, preserving first-seen order and
    /// dropping duplicates (two breakers can resolve to the same sequence).
    ///
    /// Every refusal in [`Tokenizer::overlapping_token_sequences`] propagates,
    /// so a breaker that matches nothing is an error at the boundary rather
    /// than an ignored option.
    pub fn resolve_sequence_breakers(&self, breakers: &[String]) -> Result<Vec<Vec<u32>>, String> {
        let mut out: Vec<Vec<u32>> = Vec::new();
        for breaker in breakers {
            for seq in self.overlapping_token_sequences(breaker)? {
                if !out.contains(&seq) {
                    out.push(seq);
                }
            }
        }
        Ok(out)
    }

    /// Decode token IDs to raw bytes (reverse byte-level encoding).
    ///
    /// Unlike `decode`, this never performs lossy UTF-8 conversion: a token that
    /// is an incomplete multi-byte sequence keeps its raw bytes. Callers that
    /// render text incrementally (streaming) must buffer incomplete sequences —
    /// see [`complete_utf8_prefix_len`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        if self.model == TokenizerModel::Bert {
            // WordPiece: a declared special is emitted verbatim, every other
            // token's `▁` is a space (llama.cpp's `token_to_piece` for WPM).
            let mut out = Vec::new();
            for &id in ids {
                let Some(token) = self.id_to_token.get(id as usize) else {
                    continue;
                };
                let ty = self.id_to_type.get(id as usize).copied().unwrap_or(1);
                if ty == 2 || ty == 3 || ty == 4 {
                    out.extend_from_slice(token.as_bytes());
                    continue;
                }
                for c in token.chars() {
                    if c == SPM_SPACE {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            return out;
        }
        if self.model == TokenizerModel::Rwkv {
            // RVKV: every token is its unescaped bytes (`llama-vocab.cpp:3674`).
            let mut out = Vec::new();
            for &id in ids {
                if let Some(token) = self.id_to_token.get(id as usize) {
                    out.extend_from_slice(&unescape_rwkv_token(token));
                }
            }
            return out;
        }
        if matches!(
            self.model,
            TokenizerModel::Spm
                | TokenizerModel::Ugm
                | TokenizerModel::Gemma4
                | TokenizerModel::Plamo2
        ) {
            // SentencePiece: a BYTE token is its byte, every other token's `▁`
            // is a space (llama.cpp's `token_to_piece` for SPM).
            let mut out = Vec::new();
            for &id in ids {
                let Some(token) = self.id_to_token.get(id as usize) else {
                    continue;
                };
                if self.id_to_type.get(id as usize).copied().unwrap_or(1) == 6 {
                    if let Some(b) = parse_spm_byte_token(token) {
                        out.push(b);
                        continue;
                    }
                }
                for c in token.chars() {
                    if c == SPM_SPACE {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            return out;
        }
        let mut encoded = String::new();
        for &id in ids {
            if (id as usize) < self.id_to_token.len() {
                let token = &self.id_to_token[id as usize];
                encoded.push_str(token);
            }
        }

        // Reverse byte-level encoding
        let mut result = Vec::new();
        for c in encoded.chars() {
            if let Some(&b) = self.unicode_to_byte.get(&c) {
                result.push(b);
            } else {
                // Fallback: encode the char as UTF-8
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                result.extend_from_slice(s.as_bytes());
            }
        }

        result
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Text of the BOS token ("" when unknown) — chat templates render it.
    pub fn bos_text(&self) -> String {
        self.id_to_token
            .get(self.bos_token as usize)
            .cloned()
            .unwrap_or_default()
    }
}

/// Length of the longest prefix of `bytes` that ends on a complete UTF-8
/// character boundary.
///
/// Mirrors llama.cpp's `format_incomplete_utf8` holdback: when a multi-byte
/// character is split across two tokens, the trailing bytes are kept until the
/// character completes, so streamed output never contains U+FFFD.
pub fn complete_utf8_prefix_len(bytes: &[u8]) -> usize {
    let mut end = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let len = if b < 0x80 {
            1
        } else if b & 0xE0 == 0xC0 {
            2
        } else if b & 0xF0 == 0xE0 {
            3
        } else if b & 0xF8 == 0xF0 {
            4
        } else {
            1 // stray continuation byte: consume it as one unit
        };
        if i + len > bytes.len() {
            break; // incomplete trailing character
        }
        i += len;
        end = i;
    }
    end
}

// ---------------------------------------------------------------------------
// Input normalization (#132 S1)
//
// Every reference `tokenizer.json` this engine gates against declares an NFC
// normalizer, so the reference token ids already contain it. minfer normalizes
// at `encode` / `encode_raw` — the two text→ids entry points — with generated
// tables and no new dependency; llama.cpp applies no normalizer, which is a
// recorded divergence (docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md §3.6).
// ---------------------------------------------------------------------------

mod nfc;

#[cfg(test)]
mod tests;
