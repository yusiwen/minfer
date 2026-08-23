// BPE Tokenizer (self-contained, no external deps)
// Loads tokens, scores, types, and BPE merges directly from GGUF metadata

use std::collections::HashMap;
use crate::gguf::{GgufContext, GgufType};

/// Build byte-to-unicode mapping (GPT-2 style).
fn build_byte_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u32> = Vec::new();
    // Printable ASCII: ! to ~
    for b in 0x21..=0x7e { bs.push(b); }
    // Latin-1 supplement: ¡ to ¬
    for b in 0xa1..=0xac { bs.push(b); }
    // Latin-1 supplement: ® to ÿ
    for b in 0xae..=0xff { bs.push(b); }

    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.iter().zip(cs.iter()).map(|(&b, &c)| (b as u8, char::from_u32(c).unwrap())).collect()
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
    /// Special tokens grouped by first char, longest-first within a group.
    /// Built from `special_tokens` (GGUF type 3/4) plus the hardcoded
    /// `<|im_start|>` / `<|im_end|>` / EOS fallbacks, so models whose
    /// converters mark special tokens as type 1 still match.
    special_by_first: HashMap<char, Vec<(String, u32)>>,
    pub bos_token: u32,
    pub eos_token: u32,
    pub im_start: u32,
    pub im_end: u32,
}

impl Tokenizer {
    /// Load tokenizer from a GgufContext (re-parses metadata only, no tensor data).
    pub fn load(gguf: &GgufContext) -> Self {
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
        let unicode_to_byte: HashMap<char, u8> = byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();

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

        Tokenizer {
            id_to_token,
            id_to_score,
            id_to_type,
            vocab,
            merges,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
            special_by_first,
            bos_token,
            eos_token,
            im_start,
            im_end,
        }
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
    fn bpe_encode(&self, token: &str) -> Vec<u32> {
        // If the whole token is in vocab, return it directly
        if let Some(&id) = self.vocab.get(token) {
            return vec![id];
        }

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

        // Look up each token in vocab
        word.iter()
            .map(|w| self.vocab.get(w).copied().unwrap_or(0))
            .collect()
    }

    /// GPT-2 regex pre-tokenization → byte-encode → BPE
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // GPT-2 pre-tokenization regex (from llama-vocab.cpp / gpt2_tokenizer.py)
        let re = regex::Regex::new(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
        ).expect("Invalid GPT-2 regex");

        let mut result = Vec::new();
        for mat in re.find_iter(text) {
            let pre_token = mat.as_str();
            let encoded = byte_encode(pre_token, &self.byte_to_unicode);
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
            special_by_first: HashMap::new(),
            bos_token: 0,
            eos_token: 0,
            im_start: 0,
            im_end: 0,
        };
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
        assert_eq!(complete_utf8_prefix_len(&full[..1]), 0, "1 of 3 bytes: incomplete");
        assert_eq!(complete_utf8_prefix_len(&full[..2]), 0, "2 of 3 bytes: incomplete");
        assert_eq!(complete_utf8_prefix_len(&full[..3]), 3, "all 3 bytes: complete");

        let mixed = [b'a', 0xE4, 0xB8, 0xAD, b'b'];
        assert_eq!(complete_utf8_prefix_len(&mixed[..3]), 1, "a complete, U+4E2D incomplete");
        assert_eq!(complete_utf8_prefix_len(&mixed[..4]), 4, "a + U+4E2D complete");
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
        t.special_tokens.insert("<｜end▁of▁sentence｜>".into(), 151643);
        rebuild_special_index(&mut t);
        // Populate the vocab so BPE finds "What" and " is" etc. as whole tokens
        // (the pieces the regex would produce must NOT reassemble the specials).
        // Keys must be byte-encoded like bpe_encode expects ("Ġ" = space, "Ċ" = \n).
        t.vocab.insert("What".into(), 3838);
        t.vocab.insert(byte_encode(" is", &t.byte_to_unicode), 374);
        t.vocab.insert(byte_encode(" ", &t.byte_to_unicode), 220);
        t.vocab.insert(byte_encode("2", &t.byte_to_unicode), 17);
        t.vocab.insert(byte_encode("+", &t.byte_to_unicode), 10);
        t.vocab.insert(byte_encode("?", &t.byte_to_unicode), 30);
        t.vocab.insert(byte_encode("\n", &t.byte_to_unicode), 198);
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
        let ids = t.encode("<|im_start|>hi<|im_end|>");
        // im_start/im_end fall back to 0 (unknown) in this synthetic tokenizer
        assert_eq!(ids, vec![0, 42, 0]);
    }
}
