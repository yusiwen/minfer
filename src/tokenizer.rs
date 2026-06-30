// BPE Tokenizer (self-contained, no external deps)
// Translated from: llama.cpp/src/llama-vocab.cpp + minfer2/src/tokenizer.rs
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
    pub id_to_score: Vec<f32>,
    pub id_to_type: Vec<i32>,
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), usize>,
    byte_to_unicode: HashMap<u8, char>,
    /// Reverse mapping for decode
    unicode_to_byte: HashMap<char, u8>,
    pub special_tokens: HashMap<String, u32>,
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

        Tokenizer {
            id_to_token,
            id_to_score,
            id_to_type,
            vocab,
            merges,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
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
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        let mut remaining = text;

        // Special token patterns (must match early)
        let mut special_patterns: Vec<(&str, u32)> =
            vec![("<|im_start|>", self.im_start), ("<|im_end|>", self.im_end)];
        // Also add EOS if it's a known special token
        if self.eos_token != 0 {
            // Only add EOS as special if its text is known
            if let Some(eos_text) = self.id_to_token.get(self.eos_token as usize) {
                if eos_text.starts_with('<') {
                    special_patterns.push((eos_text.as_str(), self.eos_token));
                }
            }
        }

        loop {
            // Find the earliest special token
            let mut earliest: Option<(usize, &str, u32)> = None;
            for &(pat, id) in &special_patterns {
                if let Some(pos) = remaining.find(pat) {
                    if earliest.is_none() || pos < earliest.unwrap().0 {
                        earliest = Some((pos, pat, id));
                    }
                }
            }

            if let Some((pos, pat, id)) = earliest {
                // Encode text before the special token
                if pos > 0 {
                    let before = &remaining[..pos];
                    result.extend(self.encode_bpe(before));
                }
                result.push(id);
                remaining = &remaining[pos + pat.len()..];
            } else {
                // No more special tokens, encode the rest
                result.extend(self.encode_bpe(remaining));
                break;
            }
        }

        result
    }

    /// Decode token IDs to text (reverse byte encoding).
    pub fn decode(&self, ids: &[u32]) -> String {
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

        String::from_utf8_lossy(&result).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}
