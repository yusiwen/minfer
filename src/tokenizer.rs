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
}

impl PreTokenizer {
    /// Resolve `tokenizer.ggml.pre`. Unknown and missing values are refused.
    pub fn from_gguf(pre: Option<&str>) -> Result<Self, String> {
        match pre {
            Some("qwen2") | Some("deepseek-r1-qwen") => Ok(Self::Qwen2),
            Some("qwen35") => Ok(Self::Qwen35),
            Some(other) => Err(format!(
                "tokenizer.ggml.pre = \"{other}\" is not supported by minfer's byte-level BPE \
                 pre-tokenizer (supported: \"qwen2\", alias \"deepseek-r1-qwen\"; \"qwen35\"). \
                 Refusing to tokenize rather than split with the wrong rule."
            )),
            None => Err(
                "tokenizer.ggml.pre is missing, so the pre-tokenization rule is unknown; minfer \
                 refuses to guess (supported: \"qwen2\", alias \"deepseek-r1-qwen\"; \"qwen35\")"
                    .to_string(),
            ),
        }
    }

    /// The `tokenizer.ggml.pre` spelling this rule was selected from.
    pub fn gguf_name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
        }
    }

    fn include_marks(self) -> bool {
        matches!(self, Self::Qwen35)
    }

    /// Split `text` into pre-tokenization pieces. The returned slices are in
    /// order and tile `text` exactly (concatenating them reproduces the input).
    ///
    /// This is the Qwen2 pattern
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    /// (Qwen3.5 replaces `\p{L}` with `[\p{L}\p{M}]`), implemented as a scan
    /// rather than a `regex` pattern because the crate has no lookahead.
    pub fn split<'a>(self, text: &'a str) -> Vec<&'a str> {
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

            // \p{N}
            if is_number(c) {
                pos += 1;
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

#[derive(Clone)]
pub struct Tokenizer {
    pub id_to_token: Vec<String>,
    /// Score / type / special-token maps are loaded from GGUF metadata for
    /// llama.cpp parity; the engine reads `id_to_token` + `vocab` + `merges`.
    #[allow(dead_code)]
    pub id_to_score: Vec<f32>,
    #[allow(dead_code)]
    pub id_to_type: Vec<i32>,
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), usize>,
    byte_to_unicode: HashMap<u8, char>,
    /// Reverse mapping for decode
    unicode_to_byte: HashMap<char, u8>,
    #[allow(dead_code)]
    pub special_tokens: HashMap<String, u32>,
    /// The pre-tokenization rule selected from `tokenizer.ggml.pre` (F7).
    pub pre: PreTokenizer,
    /// Special tokens grouped by first char, longest-first within a group.
    /// Built from `special_tokens` (GGUF type 3/4) plus the hardcoded
    /// `<|im_start|>` / `<|im_end|>` / EOS fallbacks, so models whose
    /// converters mark special tokens as type 1 still match.
    special_by_first: HashMap<char, Vec<(String, u32)>>,
    pub bos_token: u32,
    // Reserved special-token ids (special-token handling flows through
    // `special_tokens()` / SpecialTokens; these are kept for API completeness).
    #[allow(dead_code)]
    pub eos_token: u32,
    #[allow(dead_code)]
    pub im_start: u32,
    #[allow(dead_code)]
    pub im_end: u32,
}

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
            pre: PreTokenizer::Qwen2,
            special_by_first: HashMap::new(),
            bos_token: 0,
            eos_token: 0,
            im_start: 0,
            im_end: 0,
        }
    }

    /// Load tokenizer from a GgufContext (re-parses metadata only, no tensor data).
    ///
    /// Refuses loudly (returns `Err`) when the metadata describes a tokenizer
    /// this engine cannot reproduce byte for byte:
    ///   * `tokenizer.ggml.model` present and not `gpt2` (only byte-level BPE);
    ///   * `tokenizer.ggml.pre` missing or unsupported (see [`PreTokenizer`]);
    ///   * no merges (a byte-level BPE without ranks would silently degrade to
    ///     per-byte splits);
    ///   * a vocabulary that is not byte-level complete (the byte fallback in
    ///     [`Tokenizer::bpe_encode`] would otherwise drop bytes silently).
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

        // Special tokens: type 3 (CONTROL) or 4 (USER_DEFINED)
        let mut special_tokens = HashMap::new();
        for (id, token) in id_to_token.iter().enumerate() {
            if id_to_type.get(id).copied().unwrap_or(1) == 3
                || id_to_type.get(id).copied().unwrap_or(1) == 4
            {
                special_tokens.insert(token.clone(), id as u32);
            }
        }

        // Token IDs from GGUF metadata
        let bos_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_token = Self::get_gguf_u32(gguf, "tokenizer.ggml.eos_token_id").unwrap_or(0);
        let im_start = vocab.get("<|im_start|>").copied().unwrap_or(0);
        let im_end = vocab.get("<|im_end|>").copied().unwrap_or(eos_token);

        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte: HashMap<char, u8> =
            byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();

        // Merge GGUF special tokens (type 3/4) with hardcoded fallbacks, then
        // group by first char with longest-first ordering inside each group
        // (an earliest-position, longest-match scan needs both).
        let mut merged: HashMap<String, u32> = special_tokens.clone();
        if !merged.contains_key("<|im_start|>") {
            merged.insert("<|im_start|>".to_string(), im_start);
        }
        if !merged.contains_key("<|im_end|>") {
            merged.insert("<|im_end|>".to_string(), im_end);
        }
        if eos_token != 0 {
            if let Some(eos_text) = id_to_token.get(eos_token as usize) {
                if eos_text.starts_with('<') && !merged.contains_key(eos_text) {
                    merged.insert(eos_text.clone(), eos_token);
                }
            }
        }
        let mut special_by_first: HashMap<char, Vec<(String, u32)>> = HashMap::new();
        for (pat, id) in merged {
            let first = pat.chars().next().unwrap_or('\0');
            special_by_first.entry(first).or_default().push((pat, id));
        }
        for group in special_by_first.values_mut() {
            group.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        }

        // F7: the pre-tokenization rule must be one this engine implements.
        let model = Self::get_gguf_str(gguf, "tokenizer.ggml.model");
        if let Some(model) = model.as_deref() {
            if model != "gpt2" {
                return Err(format!(
                    "tokenizer.ggml.model = \"{model}\" is not a byte-level BPE; minfer's \
                     tokenizer implements byte-level BPE only"
                ));
            }
        }
        let pre =
            PreTokenizer::from_gguf(Self::get_gguf_str(gguf, "tokenizer.ggml.pre").as_deref())?;
        if merges.is_empty() {
            return Err(
                "tokenizer.ggml.merges is empty; minfer refuses to run a byte-level BPE with no \
                 merge ranks (every piece would silently degrade to single bytes)"
                    .to_string(),
            );
        }
        // The byte fallback is exact only when every byte has a token. Qwen's
        // vocabularies have all 256; a vocabulary that does not is a load error
        // rather than a lossy fallback.
        let missing: Vec<u8> = (0u8..=255)
            .filter(|b| !vocab.contains_key(&byte_to_unicode[b].to_string()))
            .collect();
        if let Some(first) = missing.first() {
            return Err(format!(
                "vocabulary is not byte-level complete: {} of 256 byte tokens are missing \
                 (first: <0x{first:02X}>); minfer refuses a lossy byte fallback",
                missing.len()
            ));
        }

        Ok(Tokenizer {
            id_to_token,
            id_to_score,
            id_to_type,
            vocab,
            merges,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
            pre,
            special_by_first,
            bos_token,
            eos_token,
            im_start,
            im_end,
        })
    }

    /// A GGUF metadata string value (F7: `tokenizer.ggml.model` / `.pre`).
    fn get_gguf_str(gguf: &GgufContext, key: &str) -> Option<String> {
        gguf.kv
            .iter()
            .find(|kv| kv.key == key && kv.type_ == GgufType::String)
            .map(|kv| kv.get_val_str(0).to_string())
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
        let mut result = Vec::new();
        for piece in self.pre.split(text) {
            let encoded = byte_encode(piece, &self.byte_to_unicode);
            result.extend(self.bpe_encode(&encoded));
        }
        result
    }

    /// Tokenize text into token IDs.
    ///
    /// Special tokens (from the GGUF `tokenizer.ggml.token_type` 3/4 table,
    /// plus `<|im_start|>` / `<|im_end|>` / EOS fallbacks) are matched as
    /// single tokens *before* BPE — earliest position wins, longest text wins
    /// at the same position. This is what makes special-token templates work
    /// (e.g. DeepSeek-R1's `<｜User｜>` / `<think>`), matching llama.cpp.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        let mut remaining = text;

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
                    result.extend(self.encode_bpe(&remaining[..pos]));
                }
                result.push(id);
                remaining = &remaining[pos + len..];
            } else {
                // No more special tokens, encode the rest
                result.extend(self.encode_bpe(remaining));
                break;
            }
        }

        result
    }

    /// Decode token IDs to raw bytes (reverse byte-level encoding).
    ///
    /// Unlike `decode`, this never performs lossy UTF-8 conversion: a token that
    /// is an incomplete multi-byte sequence keeps its raw bytes. Callers that
    /// render text incrementally (streaming) must buffer incomplete sequences —
    /// see [`complete_utf8_prefix_len`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
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

    /// Decode token IDs to text.
    ///
    /// Lossy: an incomplete multi-byte sequence at the end becomes U+FFFD.
    /// Streaming paths should use [`Tokenizer::decode_bytes`] plus
    /// [`complete_utf8_prefix_len`] holdback instead. (Tests use this wrapper;
    /// the CLI streams via `decode_bytes`.)
    #[allow(dead_code)]
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
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

#[cfg(test)]
mod tests {
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
            include_str!("../tests/fixtures/tokenizer/split_qwen2.json"),
        ),
        (
            "qwen35",
            include_str!("../tests/fixtures/tokenizer/split_qwen35.json"),
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
            include_str!("../tests/fixtures/tokenizer/ids_qwen2.5-0.5b-instruct.json"),
        ),
        (
            "qwen2.5-7b-instruct",
            "~/.cache/minfer/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            include_str!("../tests/fixtures/tokenizer/ids_qwen2.5-7b-instruct.json"),
        ),
        (
            "qwen2.5-14b-instruct",
            "~/.cache/minfer/models/hf/Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
            include_str!("../tests/fixtures/tokenizer/ids_qwen2.5-14b-instruct.json"),
        ),
        (
            "qwen3-0.6b",
            "~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            include_str!("../tests/fixtures/tokenizer/ids_qwen3-0.6b.json"),
        ),
        (
            "qwen3.5-0.8b",
            "~/.cache/minfer/models/hf/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
            include_str!("../tests/fixtures/tokenizer/ids_qwen3.5-0.8b.json"),
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
}
