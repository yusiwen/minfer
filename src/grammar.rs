//! Grammar- and JSON-schema-constrained decoding (F2, [#47]).
//!
//! A GBNF-style grammar is parsed into a flat program per rule, compiled into a
//! pushdown automaton (a *set* of `{rule, pc}` call stacks) that consumes Unicode
//! codepoints, and turned into a per-state token mask applied inside the one
//! sampler pipeline (`crate::sampler`). A JSON-Schema front end emits GBNF text
//! and goes through exactly the same parser/automaton, so a schema and a
//! hand-written grammar share one engine, one refuser and one set of gates.
//!
//! The design record — the exact accepted subset, every loud refusal, the mask's
//! position in the pipeline — is `docs/GRAMMAR-DESIGN.md`. This module is the
//! implementation of that contract; a construct it does not implement must be a
//! named error here, never a silent guess.
//!
//! Codepoint-level matching, byte-level correctness: token pieces are raw bytes
//! (`Tokenizer::decode_bytes`), so a piece may be one byte of a multi-byte
//! character or span several characters. A trailing incomplete sequence is
//! carried in the state (`partial`) until the next token completes it; an
//! invalid byte or a codepoint the automaton cannot consume rejects the token as
//! a whole. `\xNN` in a GBNF literal/class means the codepoint U+00NN (there is
//! no byte-level alphabet in this engine).
//!
//! [#47]: https://github.com/yusiwen/minfer/issues/47

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::models::SpecialTokens;
use crate::tokenizer::Tokenizer;

/// Bound on the automaton's call-stack depth. A stack that grows past it means a
/// rule that recurses without consuming input (left recursion), which a pushdown
/// parser cannot simulate — reported as such rather than looping.
const MAX_STACK_DEPTH: usize = 256;
/// Bound on the number of live nondeterministic stacks. Every in-scope grammar
/// stays far below it; exceeding it is an error, never a silent truncation.
const MAX_STACKS: usize = 64;
/// Bound on the number of visited stacks in one epsilon closure (a safety net
/// against an epsilon-heavy grammar, independent of `MAX_STACK_DEPTH`).
const MAX_CLOSURE_STEPS: usize = 65_536;
/// The per-run mask cache holds this many states, then clears. JSON revisits a
/// handful of states per member, so the hit rate matters; the bound keeps a long
/// generation from growing without limit.
const MAX_MASK_CACHE: usize = 64;
/// Upper bound on an explicit GBNF repetition (`{m,n}`) and on array-length
/// repetitions. An explicit cap keeps the compiled program finite and the
/// refusal names the limit.
const MAX_REPEAT: usize = 1024;

// ===========================================================================
// Instructions and the compiled grammar
// ===========================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
enum Inst {
    /// Consume exactly this codepoint (a literal character).
    Cp(u32),
    /// Consume a codepoint in the class (or, when `negated`, outside it).
    Class {
        negated: bool,
        ranges: Vec<(u32, u32)>,
    },
    /// Consume any codepoint (`.`).
    Any,
    /// Nondeterministic branch (alternation, `?`, `*`).
    Split(usize, usize),
    /// Epsilon jump (a loop back-edge).
    Jump(usize),
    /// Enter a rule: the current frame advances past this instruction and a new
    /// frame `{rule, 0}` is pushed.
    Call(u32),
    /// Return. At stack depth 1 the `root` rule has finished: that is the
    /// accepting state (no separate `Match` instruction is needed, and `root`
    /// can be referenced recursively like any other rule).
    Ret,
}

/// One automaton state: a set of call stacks plus the incomplete UTF-8 tail.
///
/// Canonicalised (stacks sorted, `partial` carried) so equal states compare and
/// hash equal — which is what makes the mask cache key correct.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateKey {
    /// Each stack is a chain of `[rule, pc]` frames; the last frame is current.
    stacks: Vec<Vec<[u32; 2]>>,
    /// Bytes of an incomplete UTF-8 character, 0..3 long.
    partial: Vec<u8>,
    /// An end-of-generation token was accepted: nothing further is allowed.
    done: bool,
}

/// The per-run mutable automaton state (one per CLI run / conversation session /
/// request / batch slot, exactly like `MirostatState`). It owns the per-state
/// mask cache, so a state visited twice pays for the vocabulary once.
#[derive(Clone)]
pub struct GrammarState {
    inner: StateKey,
    accepting: bool,
    cache: HashMap<StateKey, Arc<[u64]>>,
}

impl GrammarState {
    /// The state is a complete, acceptable string (end-of-generation is legal).
    pub fn is_accepting(&self) -> bool {
        self.accepting
    }

    /// Bytes of an incomplete UTF-8 character carried across the last token.
    pub fn pending_bytes(&self) -> usize {
        self.inner.partial.len()
    }

    /// Number of states whose mask is cached (bounded by `MAX_MASK_CACHE`).
    pub fn cached_states(&self) -> usize {
        self.cache.len()
    }
}

/// A compiled grammar: rule programs, the token pieces and end-of-generation
/// flags of one vocabulary, and the printable GBNF that produced it.
pub struct Grammar {
    /// Program per rule id.
    rules: Vec<Vec<Inst>>,
    /// Rule name per id (diagnostics).
    names: Vec<String>,
    root: u32,
    n_vocab: usize,
    /// Raw byte piece per token id; `None` for a token with an empty piece.
    pieces: Vec<Option<Box<[u8]>>>,
    /// End-of-generation token ids.
    eog: Vec<bool>,
    /// The GBNF text this grammar was compiled from (generated for a schema).
    source: String,
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grammar")
            .field("rules", &self.rules.len())
            .field("root", &self.names[self.root as usize])
            .field("n_vocab", &self.n_vocab)
            .field("eog", &self.eog.iter().filter(|b| **b).count())
            .field("source_len", &self.source.len())
            .finish()
    }
}

/// What a request asks for: raw GBNF, a JSON Schema, or "any JSON value"
/// (`response_format: {"type":"json_object"}`). The raw form travels from the
/// CLI/server boundary until the tokenizer is available, then
/// [`compile_source`] builds the [`Grammar`].
#[derive(Clone, Debug)]
pub enum GrammarSource {
    /// A GBNF grammar text.
    Gbnf(String),
    /// A JSON Schema document.
    Json(Value),
    /// Any single JSON value (`root ::= j-value`).
    AnyJson,
}

/// Compile a request's grammar against a vocabulary: build the per-token byte
/// pieces and the EOG set, then parse/compile. This is the one place the two
/// front ends meet the tokenizer, so the CLI and the server cannot disagree
/// about what "the vocabulary" means.
pub fn compile_source(
    source: &GrammarSource,
    tokenizer: &Tokenizer,
    special: &SpecialTokens,
) -> Result<Grammar, String> {
    let n_vocab = tokenizer.vocab_size();
    let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
        .map(|id| {
            let bytes = tokenizer.decode_bytes(&[id as u32]);
            if bytes.is_empty() {
                None
            } else {
                Some(bytes.into_boxed_slice())
            }
        })
        .collect();
    let mut eog = vec![false; n_vocab];
    for id in std::iter::once(special.eos).chain(special.im_end) {
        if (id as usize) < n_vocab {
            eog[id as usize] = true;
        }
    }
    match source {
        GrammarSource::Gbnf(text) => Grammar::from_gbnf(text, pieces, eog),
        GrammarSource::Json(schema) => Grammar::from_json_schema(schema, pieces, eog),
        GrammarSource::AnyJson => {
            Grammar::from_json_schema(&Value::Object(Default::default()), pieces, eog)
        }
    }
}

impl Grammar {
    /// The GBNF text this grammar was compiled from (for `--json-schema` this is
    /// the generated grammar, so what was enforced can be printed).
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Vocabulary size the mask is sized for.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// A fresh initial state for one run.
    pub fn state(&self) -> GrammarState {
        let stacks = self
            .close(vec![vec![[self.root, 0]]])
            .expect("a compiled grammar's initial state is always closable");
        let inner = StateKey {
            stacks,
            partial: Vec::new(),
            done: false,
        };
        let accepting = self.state_key_accepting(&inner);
        GrammarState {
            inner,
            accepting,
            cache: HashMap::new(),
        }
    }

    /// Compile a GBNF grammar. Every refusal names the construct (see
    /// `docs/GRAMMAR-DESIGN.md` §2).
    pub fn from_gbnf(
        source: &str,
        pieces: Vec<Option<Box<[u8]>>>,
        eog: Vec<bool>,
    ) -> Result<Grammar, String> {
        let parsed = GbnfParser::parse(source)?;
        let names: Vec<String> = parsed.iter().map(|(n, _)| n.clone()).collect();
        let ids: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u32))
            .collect();
        let root = *ids.get("root").ok_or_else(|| {
            "grammar: no rule named `root` (the start rule is required)".to_string()
        })?;
        let mut rules: Vec<Vec<Inst>> = Vec::with_capacity(parsed.len());
        for (name, node) in &parsed {
            let mut prog: Vec<Inst> = Vec::new();
            emit_node(node, &ids, &mut prog)?;
            prog.push(Inst::Ret);
            assert!(
                prog.len() < u32::MAX as usize,
                "grammar: rule `{name}` compiles to more instructions than a frame can address"
            );
            rules.push(prog);
        }
        let g = Grammar {
            rules,
            names,
            root,
            n_vocab: pieces.len(),
            pieces,
            eog,
            source: source.to_string(),
        };
        // Eagerly validate every rule's epsilon closure: left recursion and more
        // live stacks than the engine supports are compile errors here rather
        // than a panic on the first step of a generation.
        for r in 0..g.rules.len() {
            g.close(vec![vec![[r as u32, 0]]])
                .map_err(|e| format!("grammar: rule `{}`: {e}", g.names[r]))?;
        }
        Ok(g)
    }

    /// Compile a JSON Schema by generating GBNF from it and parsing that. The
    /// top-level `$defs`/`definitions` are the reference targets; anything else
    /// about `$ref` is refused.
    pub fn from_json_schema(
        schema: &Value,
        pieces: Vec<Option<Box<[u8]>>>,
        eog: Vec<bool>,
    ) -> Result<Grammar, String> {
        let gbnf = compile_schema_to_gbnf(schema)?;
        Grammar::from_gbnf(&gbnf, pieces, eog)
    }

    /// True when a byte string is a valid *prefix* of the grammar's language —
    /// the automaton never entered a dead state on it. This is the honest
    /// assertion for a generation truncated by the length limit: the whole text
    /// need not be a complete instance, but every byte emitted was legal.
    pub fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        self.consume(&self.initial_key(), bytes).is_ok()
    }

    /// True when the byte string is a complete sentence of the grammar.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        match self.consume(&self.initial_key(), bytes) {
            Ok(next) => self.state_key_accepting(&next),
            Err(_) => false,
        }
    }

    fn initial_key(&self) -> StateKey {
        StateKey {
            stacks: self
                .close(vec![vec![[self.root, 0]]])
                .expect("initial state is closable"),
            partial: Vec::new(),
            done: false,
        }
    }

    /// Rule/state description for a loud "no token is allowed" reason.
    pub fn describe(&self, st: &GrammarState) -> String {
        let rules: Vec<&str> = st
            .inner
            .stacks
            .iter()
            .filter_map(|s| s.last())
            .map(|f| self.names[f[0] as usize].as_str())
            .collect();
        format!(
            "{} live stack(s) in rule(s) [{}], {} pending UTF-8 byte(s), accepting={}",
            st.inner.stacks.len(),
            rules.join(", "),
            st.inner.partial.len(),
            st.accepting,
        )
    }

    /// The token mask for `st`: a packed bitset over token ids. Cached by state
    /// in `st`, so the O(vocabulary) walk happens at most once per distinct
    /// state even though every step creates a new one.
    pub fn mask(&self, st: &mut GrammarState) -> Result<Arc<[u64]>, String> {
        if let Some(cached) = st.cache.get(&st.inner) {
            return Ok(cached.clone());
        }
        let words = self.n_vocab.div_ceil(64);
        let mut bits = vec![0u64; words];
        let done = st.inner.done;
        // Per-call transition memo (a DFA construction on the fly): within one
        // mask computation the start state is fixed, so "consume one codepoint"
        // is a function of (state, codepoint). Interning the successor states
        // collapses the walk over ~150k token pieces into a few thousand
        // transitions — the difference between ~70 ms and ~5 ms per state on a
        // 150k-token vocabulary. Correctness is unchanged: a token is allowed
        // exactly when every codepoint of its piece is consumable (and a trailing
        // partial sequence can still complete).
        let mut states: Vec<Vec<Vec<[u32; 2]>>> = vec![st.inner.stacks.clone()];
        let mut intern: HashMap<Vec<Vec<[u32; 2]>>, usize> =
            HashMap::from([(st.inner.stacks.clone(), 0)]);
        let mut memo: HashMap<(usize, u32), Option<usize>> = HashMap::new();
        let carried = !st.inner.partial.is_empty();
        for id in 0..self.n_vocab {
            if self.eog[id] {
                // End of generation only from a complete, character-aligned state.
                if done || st.accepting {
                    bits[id / 64] |= 1u64 << (id % 64);
                }
                continue;
            }
            if done {
                continue;
            }
            let Some(piece) = self.pieces[id].as_deref() else {
                // An empty piece is not a member of any language: a control token
                // can never be "consumed" by the grammar.
                continue;
            };
            let allowed = if carried {
                // A carried partial sequence is rare: the general path handles
                // the piece + carry concatenation.
                self.consume(&st.inner, piece).is_ok()
            } else {
                self.mask_walk(piece, &mut states, &mut intern, &mut memo)
            };
            if allowed {
                bits[id / 64] |= 1u64 << (id % 64);
            }
        }
        let mask: Arc<[u64]> = bits.into();
        if st.cache.len() >= MAX_MASK_CACHE {
            st.cache.clear();
        }
        st.cache.insert(st.inner.clone(), mask.clone());
        Ok(mask)
    }

    /// Walk a token piece through the memoized transition table built by
    /// [`Self::mask`]. Returns true when every codepoint is consumable and a
    /// trailing partial sequence (if any) can still complete.
    fn mask_walk(
        &self,
        piece: &[u8],
        states: &mut Vec<Vec<Vec<[u32; 2]>>>,
        intern: &mut HashMap<Vec<Vec<[u32; 2]>>, usize>,
        memo: &mut HashMap<(usize, u32), Option<usize>>,
    ) -> bool {
        let mut cur = 0usize;
        let mut i = 0usize;
        while i < piece.len() {
            match decode_at(piece, i) {
                Utf8Step::Cp(cp, n) => {
                    if (0xD800..=0xDFFF).contains(&cp) {
                        return false;
                    }
                    match self.memo_step(states, intern, memo, cur, cp) {
                        Some(next) => cur = next,
                        None => return false,
                    }
                    i += n;
                }
                Utf8Step::Incomplete => {
                    return self.partial_can_complete(&states[cur], &piece[i..]);
                }
                Utf8Step::Invalid => return false,
            }
        }
        true
    }

    /// One memoized codepoint transition, interning the successor state.
    fn memo_step(
        &self,
        states: &mut Vec<Vec<Vec<[u32; 2]>>>,
        intern: &mut HashMap<Vec<Vec<[u32; 2]>>, usize>,
        memo: &mut HashMap<(usize, u32), Option<usize>>,
        cur: usize,
        cp: u32,
    ) -> Option<usize> {
        if let Some(cached) = memo.get(&(cur, cp)) {
            return *cached;
        }
        let next = self.step(&states[cur], cp).ok().map(|ns| {
            if let Some(&idx) = intern.get(&ns) {
                idx
            } else {
                let idx = states.len();
                states.push(ns.clone());
                intern.insert(ns, idx);
                idx
            }
        });
        memo.insert((cur, cp), next);
        next
    }

    /// Advance the state by a sampled token. A failure names the byte offset of
    /// the longest accepted prefix and leaves the state unchanged, so the caller
    /// can stop loudly instead of committing a token the grammar rejected.
    pub fn accept_token(&self, st: &mut GrammarState, id: u32) -> Result<(), String> {
        let idx = id as usize;
        if idx >= self.n_vocab {
            return Err(format!(
                "grammar: token id {id} is outside the vocabulary (0..{})",
                self.n_vocab
            ));
        }
        if st.inner.done {
            return Err("grammar: a token was sampled after the end of generation".to_string());
        }
        if self.eog[idx] {
            if !st.accepting {
                return Err(format!(
                    "grammar: end-of-generation token {id} accepted in a non-accepting state ({})",
                    self.describe(st)
                ));
            }
            st.inner.done = true;
            st.accepting = false;
            st.cache.clear();
            return Ok(());
        }
        let piece = self.pieces[idx].clone().ok_or_else(|| {
            format!("grammar: token {id} has an empty piece and cannot be matched")
        })?;
        match self.consume(&st.inner, &piece) {
            Ok(next) => {
                st.accepting = self.state_key_accepting(&next);
                st.inner = next;
                Ok(())
            }
            Err((accepted, why)) => Err(format!(
                "grammar: token {id} rejected: {why} (longest accepted prefix: {accepted} byte(s) \
                 of its {} byte piece)",
                piece.len()
            )),
        }
    }

    /// Can some stack still consume a codepoint this incomplete UTF-8 prefix can
    /// complete to?
    fn partial_can_complete(&self, stacks: &[Vec<[u32; 2]>], partial: &[u8]) -> bool {
        match completion_range(partial) {
            Some((lo, hi)) => self.any_stack_accepts_range(stacks, lo, hi),
            None => false,
        }
    }

    /// Any live stack whose current instruction accepts a codepoint in `[lo, hi]`
    /// (surrogates excluded — they have no UTF-8 encoding).
    fn any_stack_accepts_range(&self, stacks: &[Vec<[u32; 2]>], lo: u32, hi: u32) -> bool {
        non_surrogate_parts(lo, hi)
            .iter()
            .any(|&(a, b)| self.any_stack_accepts_plain_range(stacks, a, b))
    }

    fn any_stack_accepts_plain_range(&self, stacks: &[Vec<[u32; 2]>], lo: u32, hi: u32) -> bool {
        for st in stacks {
            let f = *st.last().expect("stacks are never empty");
            match &self.rules[f[0] as usize][f[1] as usize] {
                Inst::Any => return true,
                Inst::Cp(c) => {
                    if *c >= lo && *c <= hi {
                        return true;
                    }
                }
                // `ranges` is sorted ascending by construction (the GBNF parser
                // sorts a class, and emitted classes are written in order).
                Inst::Class {
                    negated: false,
                    ranges,
                } => {
                    if ranges.iter().any(|&(a, b)| a <= hi && b >= lo) {
                        return true;
                    }
                }
                Inst::Class {
                    negated: true,
                    ranges,
                } => {
                    if !covers(ranges, lo, hi) {
                        return true;
                    }
                }
                Inst::Ret | Inst::Split(_, _) | Inst::Jump(_) | Inst::Call(_) => {}
            }
        }
        false
    }

    fn state_key_accepting(&self, key: &StateKey) -> bool {
        !key.done
            && key.partial.is_empty()
            && key.stacks.iter().any(|s| {
                s.len() == 1 && matches!(self.rules[s[0][0] as usize][s[0][1] as usize], Inst::Ret)
            })
    }

    /// Epsilon-closure + dedup of a set of stacks, consuming nothing.
    fn close(&self, input: Vec<Vec<[u32; 2]>>) -> Result<Vec<Vec<[u32; 2]>>, String> {
        let mut out: Vec<Vec<[u32; 2]>> = Vec::new();
        let mut seen: HashSet<Vec<[u32; 2]>> = HashSet::new();
        let mut work = input;
        let mut steps = 0usize;
        while let Some(st) = work.pop() {
            steps += 1;
            if steps > MAX_CLOSURE_STEPS {
                return Err(format!(
                    "grammar: epsilon closure exceeded {MAX_CLOSURE_STEPS} steps (the grammar is \
                     too ambiguous or epsilon-heavy)"
                ));
            }
            if !seen.insert(st.clone()) {
                continue;
            }
            if st.len() > MAX_STACK_DEPTH {
                return Err(format!(
                    "grammar: parse stack exceeded depth {MAX_STACK_DEPTH} — a rule recurses \
                     without consuming input (left recursion is not supported)"
                ));
            }
            let frame = *st.last().expect("stacks are never empty");
            match self.flow(frame) {
                Flow::Split(a, b) => {
                    let mut s1 = st.clone();
                    s1.last_mut().unwrap()[1] = a as u32;
                    work.push(s1);
                    let mut s2 = st;
                    s2.last_mut().unwrap()[1] = b as u32;
                    work.push(s2);
                }
                Flow::Jump(a) => {
                    let mut s = st;
                    s.last_mut().unwrap()[1] = a as u32;
                    work.push(s);
                }
                Flow::Call(rule) => {
                    let mut s = st;
                    s.last_mut().unwrap()[1] = frame[1] + 1;
                    s.push([rule, 0]);
                    work.push(s);
                }
                Flow::Ret => {
                    if st.len() == 1 {
                        // The root rule finished: accepting, and nothing more can
                        // consume from this stack.
                        out.push(st);
                    } else {
                        let mut s = st;
                        s.pop();
                        work.push(s);
                    }
                }
                // A consuming instruction: this stack is done until input arrives.
                Flow::Consume => out.push(st),
            }
        }
        out.sort();
        out.dedup();
        if out.len() > MAX_STACKS {
            return Err(format!(
                "grammar: {} live parse stacks exceeds the limit of {MAX_STACKS}",
                out.len()
            ));
        }
        Ok(out)
    }

    fn flow(&self, frame: [u32; 2]) -> Flow {
        match &self.rules[frame[0] as usize][frame[1] as usize] {
            Inst::Split(a, b) => Flow::Split(*a, *b),
            Inst::Jump(a) => Flow::Jump(*a),
            Inst::Call(r) => Flow::Call(*r),
            Inst::Ret => Flow::Ret,
            Inst::Cp(_) | Inst::Class { .. } | Inst::Any => Flow::Consume,
        }
    }

    /// Consume one codepoint from every stack that can, then epsilon-close.
    fn step(&self, stacks: &[Vec<[u32; 2]>], cp: u32) -> Result<Vec<Vec<[u32; 2]>>, String> {
        let mut next: Vec<Vec<[u32; 2]>> = Vec::new();
        for st in stacks {
            let frame = *st.last().expect("stacks are never empty");
            let matched = match &self.rules[frame[0] as usize][frame[1] as usize] {
                Inst::Cp(c) => *c == cp,
                Inst::Any => true,
                Inst::Class { negated, ranges } => {
                    let inside = ranges.iter().any(|&(a, b)| cp >= a && cp <= b);
                    inside != *negated
                }
                // A Ret (accepting), Split, Jump or Call is unreachable here: the
                // closure only ever emits consuming instructions or a final Ret.
                Inst::Ret | Inst::Split(_, _) | Inst::Jump(_) | Inst::Call(_) => false,
            };
            if matched {
                let mut s = st.clone();
                s.last_mut().unwrap()[1] += 1;
                next.push(s);
            }
        }
        if next.is_empty() {
            return Err(format!("no live stack can consume U+{cp:04X}"));
        }
        self.close(next)
    }

    /// Consume raw bytes. Returns the successor state, or `(accepted_bytes,
    /// reason)` where `accepted_bytes` counts the bytes of *this* input that were
    /// consumed before the failure (the longest accepted prefix).
    fn consume(&self, from: &StateKey, bytes: &[u8]) -> Result<StateKey, (usize, String)> {
        let carried = from.partial.len();
        let mut buf: Vec<u8> = Vec::with_capacity(carried + bytes.len());
        buf.extend_from_slice(&from.partial);
        buf.extend_from_slice(bytes);
        let mut stacks = from.stacks.clone();
        let mut i = 0usize;
        while i < buf.len() {
            match decode_at(&buf, i) {
                Utf8Step::Cp(cp, n) => {
                    if (0xD800..=0xDFFF).contains(&cp) {
                        return Err((i.saturating_sub(carried), format!("surrogate U+{cp:04X}")));
                    }
                    stacks = self
                        .step(&stacks, cp)
                        .map_err(|e| (i.saturating_sub(carried), e))?;
                    i += n;
                }
                Utf8Step::Incomplete => {
                    // A trailing partial character is only acceptable when the
                    // state can still consume some codepoint this prefix can
                    // complete to. A token that carries a partial sequence the
                    // automaton can never finish would strand the run (and, if
                    // the state is already accepting, it is exactly the bug of
                    // accepting a character after the language ended).
                    if i < buf.len() && !self.partial_can_complete(&stacks, &buf[i..]) {
                        return Err((
                            i.saturating_sub(carried),
                            format!(
                                "no live stack can consume any character starting with the \
                                 incomplete UTF-8 sequence {:02X?}",
                                &buf[i..]
                            ),
                        ));
                    }
                    break;
                }
                Utf8Step::Invalid => {
                    return Err((
                        i.saturating_sub(carried),
                        format!("invalid UTF-8 byte 0x{:02X}", buf[i]),
                    ))
                }
            }
        }
        Ok(StateKey {
            stacks,
            partial: buf[i..].to_vec(),
            done: from.done,
        })
    }
}

enum Flow {
    Split(usize, usize),
    Jump(usize),
    Call(u32),
    Ret,
    Consume,
}

/// The contiguous codepoint range an incomplete-but-valid UTF-8 prefix can still
/// complete to (`[0xE4]` -> `U+4000..=U+4FFF`), or `None` when the bytes cannot
/// start a character at all.
///
/// Overlong encodings and the surrogate block are excluded here, not filtered
/// later: `[0xE0]` completes only to `U+0800..=U+0FFF` (a `U+0000..=U+07FF`
/// completion would be an overlong 3-byte form, which is not valid UTF-8), and
/// `[0xED]` only to `U+D000..=U+D7FF`. Getting this wrong is how a lone `0xE0`
/// came to be accepted after a closed JSON object — the run then had no legal
/// continuation and the response ended with U+FFFD.
fn completion_range(partial: &[u8]) -> Option<(u32, u32)> {
    let b0 = *partial.first()?;
    // (sequence length, payload mask, allowed range of the *first* continuation)
    let (len, value0, c_lo, c_hi) = match b0 {
        0xC2..=0xDF => (2u32, (b0 & 0x1F) as u32, 0x80u32, 0xBFu32),
        // 0xE0's first continuation must be >= 0xA0 (else the form is overlong).
        0xE0 => (3, (b0 & 0x0F) as u32, 0xA0, 0xBF),
        0xE1..=0xEC | 0xEE..=0xEF => (3, (b0 & 0x0F) as u32, 0x80, 0xBF),
        // 0xED's first continuation must be <= 0x9F (else it encodes a surrogate).
        0xED => (3, (b0 & 0x0F) as u32, 0x80, 0x9F),
        // 0xF0's first continuation must be >= 0x90 (else overlong).
        0xF0 => (4, (b0 & 0x07) as u32, 0x90, 0xBF),
        0xF1..=0xF3 => (4, (b0 & 0x07) as u32, 0x80, 0xBF),
        // 0xF4's first continuation must be <= 0x8F (else above U+10FFFF).
        0xF4 => (4, (b0 & 0x07) as u32, 0x80, 0x8F),
        // 0x80..=0xC1 (stray continuation / overlong 2-byte leader) and
        // 0xF5..=0xFF (above U+10FFFF) cannot start a character.
        _ => return None,
    };
    if partial.len() as u32 >= len {
        return None; // already complete, or not a prefix at all
    }
    let mut value = value0;
    for (idx, &b) in partial[1..].iter().enumerate() {
        if b & 0xC0 != 0x80 {
            return None;
        }
        if idx == 0 && !((c_lo..=c_hi).contains(&(b as u32))) {
            return None;
        }
        value = (value << 6) | (b & 0x3F) as u32;
    }
    let (lo, hi) = if partial.len() == 1 {
        // The first continuation is constrained; the rest are free.
        let tail = 6 * (len - 2);
        let lo = ((value << 6) | (c_lo - 0x80)) << tail;
        let hi = (((value << 6) | (c_hi - 0x80)) << tail) | ((1u32 << tail) - 1);
        (lo, hi)
    } else {
        // The given continuations were validated above; only the tail is free.
        let free = 6 * (len - partial.len() as u32);
        let lo = value << free;
        (lo, lo | ((1u32 << free) - 1))
    };
    Some((lo, hi))
}

/// `[lo, hi]` split around the UTF-8 surrogate block (which has no encoding).
fn non_surrogate_parts(lo: u32, hi: u32) -> Vec<(u32, u32)> {
    const SUR_LO: u32 = 0xD800;
    const SUR_HI: u32 = 0xDFFF;
    if hi < SUR_LO || lo > SUR_HI {
        return vec![(lo, hi)];
    }
    let mut parts = Vec::new();
    if lo < SUR_LO {
        parts.push((lo, SUR_LO - 1));
    }
    if hi > SUR_HI {
        parts.push((SUR_HI + 1, hi));
    }
    parts
}

/// Does the union of the (ascending, disjoint or overlapping) `ranges` cover
/// every codepoint in `[lo, hi]`?
fn covers(ranges: &[(u32, u32)], lo: u32, hi: u32) -> bool {
    let mut next = lo;
    for &(a, b) in ranges {
        if b < next {
            continue;
        }
        if a > next {
            return false;
        }
        if b >= hi {
            return true;
        }
        next = b + 1; // b < hi here, so this cannot overflow
    }
    false
}

/// One decoding step of the input buffer.
enum Utf8Step {
    /// A complete codepoint and the bytes it occupies.
    Cp(u32, usize),
    /// The tail is a valid but incomplete UTF-8 sequence.
    Incomplete,
    /// The next byte cannot begin/continue a UTF-8 sequence.
    Invalid,
}

fn decode_at(buf: &[u8], i: usize) -> Utf8Step {
    match std::str::from_utf8(&buf[i..]) {
        Ok(s) => {
            let c = s.chars().next().expect("non-empty by construction");
            Utf8Step::Cp(c as u32, c.len_utf8())
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if valid > 0 {
                let s = std::str::from_utf8(&buf[i..i + valid]).expect("valid_up_to is valid");
                let c = s.chars().next().expect("non-empty");
                Utf8Step::Cp(c as u32, c.len_utf8())
            } else if e.error_len().is_none() {
                Utf8Step::Incomplete
            } else {
                Utf8Step::Invalid
            }
        }
    }
}

// ===========================================================================
// GBNF parser and compiler
// ===========================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
enum Node {
    /// A literal string as codepoints.
    Str(Vec<u32>),
    /// A character class (possibly negated).
    Class {
        negated: bool,
        ranges: Vec<(u32, u32)>,
    },
    /// `.` — any codepoint.
    Any,
    /// A sequence (possibly empty = epsilon).
    Seq(Vec<Node>),
    /// An alternation of sequences.
    Alt(Vec<Node>),
    /// `m` mandatory repetitions and `n` total (`None` = unbounded above).
    Rep(Box<Node>, usize, Option<usize>),
    /// A reference to a named rule.
    Ref(String),
}

struct GbnfParser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> GbnfParser<'a> {
    fn parse(source: &str) -> Result<Vec<(String, Node)>, String> {
        let mut p = GbnfParser { s: source, i: 0 };
        let mut rules: Vec<(String, Node)> = Vec::new();
        loop {
            p.skip_trivia();
            if p.peek().is_none() {
                break;
            }
            let name = p.ident().map_err(|e| format!("grammar: {e}"))?;
            p.skip_trivia();
            p.expect_str("::=")?;
            let body = p.alternation()?;
            if rules.iter().any(|(n, _)| *n == name) {
                return Err(format!("grammar: duplicate rule name `{name}`"));
            }
            rules.push((name, body));
        }
        if rules.is_empty() {
            return Err("grammar: no rules (expected `root ::= ...`)".to_string());
        }
        // Every reference must resolve: an unknown rule name silently matching
        // nothing would be a guess.
        for (name, node) in &rules {
            check_refs(node, &rules, name)?;
        }
        Ok(rules)
    }

    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += c.len_utf8();
        Some(c)
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('#') => {
                    while let Some(c) = self.bump() {
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn expect_str(&mut self, lit: &str) -> Result<(), String> {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!(
                "grammar: expected `{lit}` at byte {} (near `{}`)",
                self.i,
                self.snippet()
            ))
        }
    }

    fn snippet(&self) -> String {
        self.s[self.i..].chars().take(24).collect()
    }

    fn ident(&mut self) -> Result<String, String> {
        let start = self.i;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                self.bump();
            }
            _ => {
                return Err(format!(
                    "expected a rule name at byte {} (near `{}`)",
                    self.i,
                    self.snippet()
                ))
            }
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                self.bump();
            } else {
                break;
            }
        }
        Ok(self.s[start..self.i].to_string())
    }

    fn alternation(&mut self) -> Result<Node, String> {
        let mut branches = vec![self.sequence()?];
        loop {
            self.skip_trivia();
            if self.peek() == Some('|') {
                self.bump();
                branches.push(self.sequence()?);
            } else {
                break;
            }
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Node::Alt(branches))
        }
    }

    fn sequence(&mut self) -> Result<Node, String> {
        let mut items: Vec<Node> = Vec::new();
        loop {
            self.skip_trivia();
            // A rule body ends where the next `name ::=` begins: without this the
            // next rule's name would be consumed as a rule reference.
            if self.at_rule_boundary() {
                break;
            }
            match self.peek() {
                None | Some('|') | Some(')') => break,
                _ => items.push(self.item()?),
            }
        }
        if items.len() == 1 {
            Ok(items.pop().unwrap())
        } else {
            Ok(Node::Seq(items))
        }
    }

    /// True when the input at the cursor is `ident` + trivia + `::=` — the start
    /// of the next rule, which terminates the current rule's body.
    fn at_rule_boundary(&self) -> bool {
        let rest = &self.s[self.i..];
        match rest.chars().next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return false,
        }
        let mut end = 0usize;
        for (idx, c) in rest.char_indices() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                end = idx + c.len_utf8();
            } else {
                break;
            }
        }
        let after = &rest[end..];
        let b = after.as_bytes();
        let mut j = 0usize;
        loop {
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'#' {
                while j < b.len() && b[j] != b'\n' {
                    j += 1;
                }
                continue;
            }
            break;
        }
        after[j..].starts_with("::=")
    }

    fn item(&mut self) -> Result<Node, String> {
        let mut node = self.atom()?;
        loop {
            self.skip_trivia();
            match self.peek() {
                Some('*') => {
                    self.bump();
                    node = Node::Rep(Box::new(node), 0, None);
                }
                Some('+') => {
                    self.bump();
                    node = Node::Rep(Box::new(node), 1, None);
                }
                Some('?') => {
                    self.bump();
                    node = Node::Rep(Box::new(node), 0, Some(1));
                }
                Some('{') => {
                    self.bump();
                    let m = self.number("repetition lower bound")?;
                    self.skip_trivia();
                    let n = match self.peek() {
                        Some(',') => {
                            self.bump();
                            self.skip_trivia();
                            if self.peek() == Some('}') {
                                None
                            } else {
                                Some(self.number("repetition upper bound")?)
                            }
                        }
                        _ => Some(m),
                    };
                    self.skip_trivia();
                    if self.peek() != Some('}') {
                        return Err(format!(
                            "grammar: expected `}}` closing a repetition at byte {} (near `{}`)",
                            self.i,
                            self.snippet()
                        ));
                    }
                    self.bump();
                    if let Some(n) = n {
                        if n < m {
                            return Err(format!(
                                "grammar: repetition {{{m},{n}}} has an upper bound below its \
                                 lower bound"
                            ));
                        }
                        if n > MAX_REPEAT {
                            return Err(format!(
                                "grammar: repetition upper bound {n} exceeds the supported limit \
                                 of {MAX_REPEAT}"
                            ));
                        }
                    }
                    node = Node::Rep(Box::new(node), m, n);
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn number(&mut self, what: &str) -> Result<usize, String> {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.i {
            return Err(format!(
                "grammar: expected a decimal {what} at byte {} (near `{}`)",
                self.i,
                self.snippet()
            ));
        }
        self.s[start..self.i]
            .parse::<usize>()
            .map_err(|_| format!("grammar: {what} `{}` is too large", &self.s[start..self.i]))
    }

    fn atom(&mut self) -> Result<Node, String> {
        self.skip_trivia();
        match self.peek() {
            Some('(') => {
                self.bump();
                let inner = self.alternation()?;
                self.skip_trivia();
                if self.peek() != Some(')') {
                    return Err(format!(
                        "grammar: missing `)` at byte {} (near `{}`)",
                        self.i,
                        self.snippet()
                    ));
                }
                self.bump();
                Ok(inner)
            }
            Some('"') => {
                self.bump();
                let mut cps = Vec::new();
                loop {
                    match self.bump() {
                        None => return Err("grammar: unterminated string literal".to_string()),
                        Some('"') => break,
                        Some('\\') => cps.push(self.escape()?),
                        Some(c) => cps.push(c as u32),
                    }
                }
                Ok(Node::Str(cps))
            }
            Some('[') => {
                self.bump();
                self.char_class()
            }
            Some('.') => {
                self.bump();
                Ok(Node::Any)
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => Ok(Node::Ref(self.ident()?)),
            _ => Err(format!(
                "grammar: expected a literal, class, group, `.` or rule name at byte {} \
                 (near `{}`)",
                self.i,
                self.snippet()
            )),
        }
    }

    /// One escape, after the backslash was consumed.
    fn escape(&mut self) -> Result<u32, String> {
        let c = self
            .bump()
            .ok_or_else(|| "grammar: unterminated escape".to_string())?;
        let cp = match c {
            'n' => '\n' as u32,
            'r' => '\r' as u32,
            't' => '\t' as u32,
            '\\' => '\\' as u32,
            '"' => '"' as u32,
            '\'' => '\'' as u32,
            '[' => '[' as u32,
            ']' => ']' as u32,
            '-' => '-' as u32,
            '^' => '^' as u32,
            '/' => '/' as u32,
            'x' => self.hex(2)?,
            'u' => self.hex(4)?,
            other => {
                return Err(format!(
                    "grammar: unsupported escape `\\{other}` at byte {} (supported: \\n \\r \\t \
                     \\\\ \\\" \\' \\[ \\] \\- \\^ \\/ \\xNN \\uNNNN; character classes such as \
                     \\d/\\w/\\p{{...}} are not supported — write [0-9], [A-Za-z0-9_], ...)",
                    self.i
                ))
            }
        };
        Ok(cp)
    }

    fn hex(&mut self, n: usize) -> Result<u32, String> {
        let start = self.i;
        for _ in 0..n {
            match self.peek() {
                Some(c) if c.is_ascii_hexdigit() => {
                    self.bump();
                }
                _ => {
                    return Err(format!(
                        "grammar: expected {n} hex digit(s) at byte {} (near `{}`)",
                        self.i,
                        self.snippet()
                    ))
                }
            }
        }
        u32::from_str_radix(&self.s[start..self.i], 16)
            .map_err(|_| format!("grammar: bad hex escape `{}`", &self.s[start..self.i]))
    }

    fn char_class(&mut self) -> Result<Node, String> {
        let negated = if self.peek() == Some('^') {
            self.bump();
            true
        } else {
            false
        };
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        loop {
            match self.peek() {
                None => return Err("grammar: unterminated character class".to_string()),
                Some(']') => {
                    self.bump();
                    break;
                }
                Some('\\') => {
                    self.bump();
                    let lo = self.escape()?;
                    self.push_class_item(&mut ranges, lo)?;
                }
                Some(c) => {
                    self.bump();
                    self.push_class_item(&mut ranges, c as u32)?;
                }
            }
        }
        if ranges.is_empty() {
            return Err("grammar: empty character class `[]`".to_string());
        }
        ranges.sort();
        Ok(Node::Class { negated, ranges })
    }

    /// Add `lo` or the range `lo-hi`; `-` directly before `]` is a literal `-`.
    fn push_class_item(&mut self, ranges: &mut Vec<(u32, u32)>, lo: u32) -> Result<(), String> {
        if self.peek() != Some('-') {
            ranges.push((lo, lo));
            return Ok(());
        }
        // Consume the `-`; `-]` (or EOF) means a literal minus.
        self.bump();
        match self.peek() {
            Some(']') | None => {
                ranges.push((lo, lo));
                ranges.push(('-' as u32, '-' as u32));
                Ok(())
            }
            Some('\\') => {
                self.bump();
                let hi = self.escape()?;
                self.push_range(ranges, lo, hi)
            }
            Some(c) => {
                self.bump();
                self.push_range(ranges, lo, c as u32)
            }
        }
    }

    fn push_range(&self, ranges: &mut Vec<(u32, u32)>, lo: u32, hi: u32) -> Result<(), String> {
        if hi < lo {
            return Err(format!(
                "grammar: character class range U+{lo:04X}-U+{hi:04X} is reversed"
            ));
        }
        ranges.push((lo, hi));
        Ok(())
    }
}

fn check_refs(node: &Node, rules: &[(String, Node)], owner: &str) -> Result<(), String> {
    match node {
        Node::Ref(name) => {
            if !rules.iter().any(|(n, _)| n == name) {
                return Err(format!(
                    "grammar: rule `{owner}` references undefined rule `{name}`"
                ));
            }
            Ok(())
        }
        Node::Seq(items) | Node::Alt(items) => {
            for it in items {
                check_refs(it, rules, owner)?;
            }
            Ok(())
        }
        Node::Rep(inner, _, _) => check_refs(inner, rules, owner),
        Node::Str(_) | Node::Class { .. } | Node::Any => Ok(()),
    }
}

/// Emit a node's instructions into `prog`. Forward jumps are patched in place,
/// so a single pass suffices (repetitions re-walk their sub-node).
fn emit_node(node: &Node, ids: &HashMap<&str, u32>, prog: &mut Vec<Inst>) -> Result<(), String> {
    match node {
        Node::Str(cps) => {
            for &c in cps {
                prog.push(Inst::Cp(c));
            }
        }
        Node::Class { negated, ranges } => prog.push(Inst::Class {
            negated: *negated,
            ranges: ranges.clone(),
        }),
        Node::Any => prog.push(Inst::Any),
        Node::Ref(name) => {
            let id = *ids
                .get(name.as_str())
                .ok_or_else(|| format!("grammar: undefined rule `{name}` (internal)"))?;
            prog.push(Inst::Call(id));
        }
        Node::Seq(items) => {
            for it in items {
                emit_node(it, ids, prog)?;
            }
        }
        Node::Alt(branches) => {
            let mut jumps: Vec<usize> = Vec::new();
            for (k, branch) in branches.iter().enumerate() {
                if k + 1 < branches.len() {
                    let split = prog.len();
                    prog.push(Inst::Split(0, 0));
                    let start = prog.len();
                    if let Inst::Split(a, _) = &mut prog[split] {
                        *a = start;
                    }
                    emit_node(branch, ids, prog)?;
                    let j = prog.len();
                    prog.push(Inst::Jump(0));
                    jumps.push(j);
                    let next = prog.len();
                    if let Inst::Split(_, b) = &mut prog[split] {
                        *b = next;
                    }
                } else {
                    emit_node(branch, ids, prog)?;
                }
            }
            let end = prog.len();
            for j in jumps {
                if let Inst::Jump(t) = &mut prog[j] {
                    *t = end;
                }
            }
        }
        Node::Rep(inner, m, n) => {
            for _ in 0..*m {
                emit_node(inner, ids, prog)?;
            }
            match n {
                None => {
                    // L1: split(body, end); body; jump L1; end:
                    let split = prog.len();
                    prog.push(Inst::Split(0, 0));
                    let body = prog.len();
                    if let Inst::Split(a, _) = &mut prog[split] {
                        *a = body;
                    }
                    emit_node(inner, ids, prog)?;
                    prog.push(Inst::Jump(split));
                    let end = prog.len();
                    if let Inst::Split(_, b) = &mut prog[split] {
                        *b = end;
                    }
                }
                Some(total) => {
                    for _ in *m..*total {
                        let split = prog.len();
                        prog.push(Inst::Split(0, 0));
                        let body = prog.len();
                        if let Inst::Split(a, _) = &mut prog[split] {
                            *a = body;
                        }
                        emit_node(inner, ids, prog)?;
                        let end = prog.len();
                        if let Inst::Split(_, b) = &mut prog[split] {
                            *b = end;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ===========================================================================
// JSON Schema -> GBNF
// ===========================================================================

/// The generic JSON value grammar, emitted once per compilation. Schema-specific
/// rules reference these; `additionalProperties: true` and `enum`/`const`
/// literals reuse them.
const GENERIC_RULES: &[(&str, &str)] = &[
    ("j-ws", r#"[ \t\n\r]*"#),
    ("j-char", r#"[^"\\\u0000-\u001f] | "\\" j-esc"#),
    (
        "j-esc",
        r#"["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]"#,
    ),
    ("j-string", r#""\"" j-char* "\"""#),
    ("j-int", r#""0" | [1-9] [0-9]*"#),
    ("j-frac", r#""." [0-9]+"#),
    ("j-exp", r#"[eE] [+-]? [0-9]+"#),
    ("j-number", r#""-"? j-int j-frac? j-exp?"#),
    ("j-integer", r#""-"? j-int"#),
    (
        "j-value",
        r#"j-object | j-array | j-string | j-number | "true" | "false" | "null""#,
    ),
    (
        "j-object",
        r#""{" j-ws "}" | "{" j-ws j-pair (j-ws "," j-ws j-pair)* j-ws "}""#,
    ),
    ("j-pair", r#"j-string j-ws ":" j-ws j-value"#),
    (
        "j-array",
        r#""[" j-ws "]" | "[" j-ws j-value (j-ws "," j-ws j-value)* j-ws "]""#,
    ),
];

/// Keywords that change the accepted language but are not implemented: each one
/// is a named refusal (design §3).
const REFUSED_KEYWORDS: &[&str] = &[
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "contentEncoding",
    "contentMediaType",
    "multipleOf",
    "minProperties",
    "maxProperties",
    "propertyNames",
    "patternProperties",
    "dependentRequired",
    "dependentSchemas",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
];

struct JsonCompiler {
    rules: Vec<(String, String)>,
    defs: HashMap<String, Value>,
    def_rules: HashMap<String, String>,
    counter: usize,
}

/// Compile a JSON Schema into GBNF text (the public entry point used by
/// [`Grammar::from_json_schema`] and printable as the enforced grammar).
pub fn compile_schema_to_gbnf(schema: &Value) -> Result<String, String> {
    let mut defs = HashMap::new();
    if let Some(obj) = schema.as_object() {
        for key in ["$defs", "definitions"] {
            if let Some(Value::Object(map)) = obj.get(key) {
                for (name, sub) in map {
                    defs.insert(name.clone(), sub.clone());
                }
            }
        }
    }
    let mut c = JsonCompiler {
        rules: GENERIC_RULES
            .iter()
            .map(|(n, b)| (n.to_string(), b.to_string()))
            .collect(),
        defs,
        def_rules: HashMap::new(),
        counter: 0,
    };
    let body = c.compile(schema, "root")?;
    c.rules.push(("root".to_string(), body));
    let mut out = String::new();
    for (name, body) in &c.rules {
        out.push_str(name);
        out.push_str(" ::= ");
        out.push_str(body);
        out.push('\n');
    }
    Ok(out)
}

impl JsonCompiler {
    fn fresh(&mut self, hint: &str) -> String {
        let clean: String = hint
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        self.counter += 1;
        format!("{clean}-{}", self.counter)
    }

    fn compile(&mut self, schema: &Value, hint: &str) -> Result<String, String> {
        match schema {
            Value::Bool(true) => return Ok("j-value".to_string()),
            Value::Bool(false) => {
                return Err(
                    "json-schema: `false` admits no value and cannot be compiled".to_string(),
                )
            }
            Value::Object(_) => {}
            _ => {
                return Err(format!(
                    "json-schema: a schema must be an object or a boolean, got {}",
                    json_type_name(schema)
                ))
            }
        }
        let obj = schema.as_object().unwrap();

        // `$ref`: local pointers only, and not combined with other constraints
        // (silently ignoring the siblings would drop constraints).
        if let Some(Value::String(r)) = obj.get("$ref") {
            let extra: Vec<&str> = obj
                .keys()
                .filter(|k| !is_annotation(k) && k.as_str() != "$ref")
                .map(|k| k.as_str())
                .collect();
            if !extra.is_empty() {
                return Err(format!(
                    "json-schema: `$ref` combined with {extra:?} is not supported (the siblings \
                     would be silently dropped); inline the referenced schema instead"
                ));
            }
            return self.resolve_ref(r);
        }

        // Explicitly refused keywords, wherever they appear.
        for key in REFUSED_KEYWORDS {
            if obj.contains_key(*key) {
                return Err(format!(
                    "json-schema: `{key}` is not supported by this compiler (see \
                     docs/GRAMMAR-DESIGN.md §3); remove it or constrain the model another way"
                ));
            }
        }

        // `const` / `enum` first: they are the whole language of the schema.
        if let Some(c) = obj.get("const") {
            self.reject_mixed(obj, &["const"])?;
            return literal_expr(c);
        }
        match obj.get("enum") {
            Some(Value::Array(values)) => {
                self.reject_mixed(obj, &["enum"])?;
                if values.is_empty() {
                    return Err("json-schema: an empty `enum` admits no value".to_string());
                }
                let mut alts = Vec::new();
                for v in values {
                    alts.push(literal_expr(v)?);
                }
                if alts.len() == 1 {
                    return Ok(alts.pop().unwrap());
                }
                return Ok(alts
                    .into_iter()
                    .map(|a| format!("({a})"))
                    .collect::<Vec<_>>()
                    .join(" | "));
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `enum` must be an array, got {}",
                    json_type_name(other)
                ))
            }
            None => {}
        }

        // `anyOf` / `oneOf`: a union, and only on its own (an AND with sibling
        // constraints cannot be expressed as a GBNF alternation).
        for key in ["anyOf", "oneOf"] {
            match obj.get(key) {
                Some(Value::Array(branches)) => {
                    self.reject_mixed(obj, &[key])?;
                    if branches.is_empty() {
                        return Err(format!(
                            "json-schema: `{key}` with no branches admits nothing"
                        ));
                    }
                    let mut alts = Vec::new();
                    for (i, b) in branches.iter().enumerate() {
                        alts.push(format!(
                            "({})",
                            self.compile(b, &format!("{hint}-{key}-{i}"))?
                        ));
                    }
                    if alts.len() == 1 {
                        return Ok(alts.pop().unwrap());
                    }
                    // `oneOf` is compiled as `anyOf` (design §3): for
                    // non-overlapping branches the two coincide; for overlapping
                    // ones this is a superset, stated in the record.
                    return Ok(alts.join(" | "));
                }
                Some(_) => {
                    return Err(format!("json-schema: `{key}` must be an array of schemas"));
                }
                None => {}
            }
        }

        // `type`, or inferred from the present keywords.
        let types: Vec<&str> = match obj.get("type") {
            None => Vec::new(),
            Some(Value::String(s)) => vec![s.as_str()],
            Some(Value::Array(list)) => {
                let mut out = Vec::new();
                for v in list {
                    match v {
                        Value::String(s) => out.push(s.as_str()),
                        other => {
                            return Err(format!(
                                "json-schema: `type` entries must be strings, got {}",
                                json_type_name(other)
                            ))
                        }
                    }
                }
                out
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `type` must be a string or an array of strings, got {}",
                    json_type_name(other)
                ))
            }
        };

        // Cross-type consistency: a keyword for a type the schema excludes is a
        // schema bug, not a constraint to ignore.
        let has_object_kw = ["properties", "required", "additionalProperties"]
            .iter()
            .any(|k| obj.contains_key(*k));
        let has_array_kw = ["items", "prefixItems", "minItems", "maxItems"]
            .iter()
            .any(|k| obj.contains_key(*k));
        let declares_object = types.is_empty() || types.contains(&"object");
        let declares_array = types.is_empty() || types.contains(&"array");
        if has_object_kw && !declares_object {
            return Err(format!(
                "json-schema: object keywords present but `type` is {types:?}"
            ));
        }
        if has_array_kw && !declares_array {
            return Err(format!(
                "json-schema: array keywords present but `type` is {types:?}"
            ));
        }

        if types.is_empty() {
            if has_object_kw {
                return self.compile_object(obj, hint);
            }
            if has_array_kw {
                return self.compile_array(obj, hint);
            }
            // No type and no type-specific keyword: any JSON value.
            return Ok("j-value".to_string());
        }
        if types.len() > 1 {
            let mut alts = Vec::new();
            for t in &types {
                let mut sub = obj.clone();
                sub.insert("type".to_string(), Value::String((*t).to_string()));
                alts.push(format!("({})", self.compile(&Value::Object(sub), hint)?));
            }
            return Ok(alts.join(" | "));
        }
        match types[0] {
            "object" => self.compile_object(obj, hint),
            "array" => self.compile_array(obj, hint),
            "string" => Ok("j-string".to_string()),
            "number" => {
                if numeric_bound_shape(obj)? {
                    return Err(
                        "json-schema: numeric bounds on `type: \"number\"` are not supported \
                         (only integer bounds are compiled; narrowing a number to its integer \
                         range would silently change the accepted language) — use \
                         `type: \"integer\"` or remove the bounds"
                            .to_string(),
                    );
                }
                Ok("j-number".to_string())
            }
            "integer" => self.compile_integer(obj),
            "boolean" => Ok("\"true\" | \"false\"".to_string()),
            "null" => Ok("\"null\"".to_string()),
            other => Err(format!(
                "json-schema: unsupported `type` value `{other}` (supported: object, array, \
                 string, number, integer, boolean, null)"
            )),
        }
    }

    /// Refuse keywords combined with a terminal form (`const`/`enum`/`anyOf`)
    /// where the combination would have to be intersected.
    fn reject_mixed(
        &self,
        obj: &serde_json::Map<String, Value>,
        allowed: &[&str],
    ) -> Result<(), String> {
        let extra: Vec<&str> = obj
            .keys()
            .filter(|k| !is_annotation(k) && !allowed.contains(&k.as_str()))
            .map(|k| k.as_str())
            .collect();
        if extra.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "json-schema: `{}` combined with {extra:?} is not supported (the constraint would \
                 have to be intersected with the union)",
                allowed[0]
            ))
        }
    }

    fn resolve_ref(&mut self, r: &str) -> Result<String, String> {
        let name = r
            .strip_prefix("#/$defs/")
            .or_else(|| r.strip_prefix("#/definitions/"))
            .ok_or_else(|| {
                format!(
                    "json-schema: `$ref: \"{r}\"` is not supported — only local \
                     `#/$defs/<name>` and `#/definitions/<name>` pointers are"
                )
            })?;
        if name.contains('/') {
            return Err(format!(
                "json-schema: nested `$ref` pointer `{r}` is not supported (only a whole \
                 `$defs`/`definitions` entry)"
            ));
        }
        if let Some(existing) = self.def_rules.get(name) {
            return Ok(existing.clone());
        }
        let schema = self.defs.get(name).cloned().ok_or_else(|| {
            format!("json-schema: `$ref: \"{r}\"` does not resolve to a definition")
        })?;
        // Register the rule *before* compiling its body so a recursive reference
        // resolves to the rule being built (JSON's nested-object shape).
        let rule = self.fresh(&format!("def-{name}"));
        self.def_rules.insert(name.to_string(), rule.clone());
        let idx = self.rules.len();
        self.rules.push((rule.clone(), String::new()));
        let body = self.compile(&schema, &format!("def-{name}"))?;
        self.rules[idx].1 = body;
        Ok(rule)
    }

    fn compile_object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        hint: &str,
    ) -> Result<String, String> {
        let props = match obj.get("properties") {
            None => serde_json::Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(other) => {
                return Err(format!(
                    "json-schema: `properties` must be an object, got {}",
                    json_type_name(other)
                ))
            }
        };
        let required: Vec<String> = match obj.get("required") {
            None => Vec::new(),
            Some(Value::Array(list)) => {
                let mut out = Vec::new();
                for v in list {
                    match v {
                        Value::String(s) => out.push(s.clone()),
                        other => {
                            return Err(format!(
                                "json-schema: `required` entries must be strings, got {}",
                                json_type_name(other)
                            ))
                        }
                    }
                }
                out
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `required` must be an array, got {}",
                    json_type_name(other)
                ))
            }
        };
        for name in &required {
            if !props.contains_key(name) {
                return Err(format!(
                    "json-schema: `required` names `{name}`, which is not declared in \
                     `properties`; this compiler cannot place an undeclared required property"
                ));
            }
        }
        let extra: Option<String> = match obj.get("additionalProperties") {
            None | Some(Value::Bool(true)) => Some("j-value".to_string()),
            Some(Value::Bool(false)) => None,
            Some(schema @ Value::Object(_)) => {
                Some(self.compile(schema, &format!("{hint}-additional"))?)
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `additionalProperties` must be a boolean or a schema, got {}",
                    json_type_name(other)
                ))
            }
        };

        let names: Vec<String> = props.keys().cloned().collect();
        let req: Vec<bool> = names.iter().map(|n| required.contains(n)).collect();
        let member: Vec<String> = names
            .iter()
            .map(|n| {
                let key = json_string_literal(n);
                let body = self.compile(&props[n], &format!("{hint}-prop-{n}"))?;
                Ok(format!("{key} j-ws \":\" j-ws ({body})"))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Two rule families per gap between declared properties:
        //
        //   `first-i` — nothing has been emitted yet, so the next entry has no
        //               leading comma; `rest-i` — at least one entry is in, so
        //               every further entry is comma-prefixed.
        //
        // Declared members appear in declaration order, a required member can
        // never be skipped (the epsilon and skip alternatives are withheld while
        // one remains), and an optional member may be skipped. `first-0` is the
        // entry point, so a schema with no required property may produce `{}`.
        // Extra (undeclared) entries may be interleaved wherever no required
        // member is still pending.
        let k = member.len();
        let mut rest: Vec<String> = vec![String::new(); k + 1];
        let mut first: Vec<String> = vec![String::new(); k + 1];
        for i in (0..=k).rev() {
            let req_remaining = i < k && req[i..].iter().any(|b| *b);
            let rest_name = format!("{hint}-rest-{i}");
            let mut ralts: Vec<String> = Vec::new();
            if !req_remaining {
                ralts.push(String::new()); // epsilon: stop here
            }
            if i < k {
                ralts.push(format!("j-ws \",\" j-ws {} {}", member[i], rest[i + 1]));
                if !req[i] {
                    ralts.push(rest[i + 1].clone()); // skip this optional member
                }
                if let Some(e) = &extra {
                    if !req[i] {
                        ralts.push(format!(
                            "j-ws \",\" j-ws j-string j-ws \":\" j-ws ({e}) {rest_name}"
                        ));
                    }
                }
            } else if let Some(e) = &extra {
                ralts.push(format!(
                    "j-ws \",\" j-ws j-string j-ws \":\" j-ws ({e}) {rest_name}"
                ));
            }
            self.rules.push((rest_name.clone(), ralts.join(" | ")));
            rest[i] = rest_name;

            if i == k {
                first[i] = rest[i].clone();
            } else {
                let first_name = format!("{hint}-first-{i}");
                let mut falts: Vec<String> = Vec::new();
                if !req_remaining {
                    falts.push(String::new());
                }
                falts.push(format!("{} {}", member[i], rest[i + 1]));
                if !req[i] {
                    falts.push(first[i + 1].clone()); // skip this optional member
                }
                if let Some(e) = &extra {
                    if !req[i] {
                        falts.push(format!("j-string j-ws \":\" j-ws ({e}) {}", rest[i]));
                    }
                }
                self.rules.push((first_name.clone(), falts.join(" | ")));
                first[i] = first_name;
            }
        }
        Ok(format!("\"{{\" j-ws {} j-ws \"}}\"", first[0]))
    }

    fn compile_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        hint: &str,
    ) -> Result<String, String> {
        if let Some(v) = obj.get("items") {
            if v.is_array() {
                return Err(
                    "json-schema: `items` as an array (the draft-07 tuple form) is not supported; \
                     use `prefixItems`"
                        .to_string(),
                );
            }
        }
        let prefix: Vec<Value> = match obj.get("prefixItems") {
            None => Vec::new(),
            Some(Value::Array(list)) => list.clone(),
            Some(other) => {
                return Err(format!(
                    "json-schema: `prefixItems` must be an array, got {}",
                    json_type_name(other)
                ))
            }
        };
        if prefix.len() > 64 {
            return Err(format!(
                "json-schema: `prefixItems` of {} entries exceeds the supported limit of 64",
                prefix.len()
            ));
        }
        let min_items = get_count(obj, "minItems")?.unwrap_or(0);
        let max_items = get_count(obj, "maxItems")?;
        if let Some(hi) = max_items {
            if min_items > hi {
                return Err(format!(
                    "json-schema: `minItems` ({min_items}) exceeds `maxItems` ({hi}) — no value \
                     can satisfy it"
                ));
            }
        }

        let items_expr: Option<String> = match obj.get("items") {
            None => Some("j-value".to_string()),
            Some(Value::Bool(true)) => Some("j-value".to_string()),
            Some(Value::Bool(false)) => None,
            Some(schema @ Value::Object(_)) => {
                Some(self.compile(schema, &format!("{hint}-items"))?)
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `items` must be a schema, got {}",
                    json_type_name(other)
                ))
            }
        };
        let p = prefix.len() as u64;
        let elem: Vec<String> = prefix
            .iter()
            .enumerate()
            .map(|(i, s)| self.compile(s, &format!("{hint}-prefix-{i}")))
            .collect::<Result<Vec<_>, String>>()?;

        // `core(n)` is the n elements without the surrounding brackets.
        let core = |n: u64| -> String {
            let mut parts: Vec<String> = Vec::with_capacity(n as usize);
            for i in 0..n {
                let e = if (i as usize) < elem.len() {
                    elem[i as usize].clone()
                } else {
                    items_expr.clone().unwrap_or_else(|| "j-value".to_string())
                };
                parts.push(e);
            }
            parts.join(" j-ws \",\" j-ws ")
        };
        let exact = |n: u64| -> String {
            let body = core(n);
            if body.is_empty() {
                "\"[\" j-ws \"]\"".to_string()
            } else {
                format!("\"[\" j-ws {body} j-ws \"]\"")
            }
        };

        let mut alts: Vec<String> = Vec::new();
        match &items_expr {
            // A closed tuple (`items: false`) fixes the maximum length at `p`.
            None => {
                let hi = max_items.unwrap_or(p).min(p);
                if min_items > hi {
                    return Err(format!(
                        "json-schema: `minItems` ({min_items}) exceeds the available {hi} \
                         `prefixItems` while `items: false` forbids more — no value can satisfy it"
                    ));
                }
                for n in min_items..=hi {
                    alts.push(exact(n));
                }
            }
            // `items` present: the per-index schemas cover counts below `p`; from
            // the "grow" point on, every further element uses the `items` schema.
            Some(items) => {
                // Counts strictly below `p` use the per-index schemas and are
                // enumerated; the empty array is its own alternative when `p` is 0.
                let exact_hi = max_items
                    .map(|h| h.min(p.saturating_sub(1)))
                    .unwrap_or(p.saturating_sub(1));
                for n in min_items..=exact_hi {
                    if n < p {
                        alts.push(exact(n));
                    }
                }
                if p == 0 && min_items == 0 {
                    alts.push(exact(0));
                }
                // From the grow point on every further element uses `items`. The
                // head carries at least one element, because a comma can only be
                // attached to a preceding element.
                let grow = min_items.max(p).max(1);
                let grows = max_items.map(|h| h >= grow).unwrap_or(true);
                if grows {
                    let rep = match max_items {
                        None => format!("(j-ws \",\" j-ws {items})*"),
                        Some(hi) => {
                            let extra = hi - grow;
                            if extra > MAX_REPEAT as u64 {
                                return Err(format!(
                                    "json-schema: an array of up to {hi} items needs a repetition \
                                     bound of {extra}, above the supported limit of {MAX_REPEAT}"
                                ));
                            }
                            if extra == 0 {
                                String::new()
                            } else {
                                format!("(j-ws \",\" j-ws {items}){{0,{extra}}}")
                            }
                        }
                    };
                    let head = core(grow);
                    let body = format!("{head}{rep}");
                    if body.is_empty() {
                        alts.push("\"[\" j-ws \"]\"".to_string());
                    } else {
                        alts.push(format!("\"[\" j-ws {body} j-ws \"]\""));
                    }
                }
            }
        }
        if alts.is_empty() {
            return Err("json-schema: the array bounds admit no length".to_string());
        }
        if alts.len() == 1 {
            return Ok(alts.pop().unwrap());
        }
        Ok(alts
            .into_iter()
            .map(|a| format!("({a})"))
            .collect::<Vec<_>>()
            .join(" | "))
    }

    fn compile_integer(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String, String> {
        let lo = integer_bound(obj, "minimum", "exclusiveMinimum", 1)?;
        let hi = integer_bound(obj, "maximum", "exclusiveMaximum", -1)?;
        match (lo, hi) {
            (None, None) => Ok("j-integer".to_string()),
            (lo, hi) => {
                let lo = lo.unwrap_or(i64::MIN as i128);
                let hi = hi.unwrap_or(i64::MAX as i128);
                int_range_expr(lo, hi)
            }
        }
    }
}

/// True when any numeric-bound keyword is present; validates their *shape* so a
/// malformed bound is reported as malformed rather than as unsupported.
fn numeric_bound_shape(obj: &serde_json::Map<String, Value>) -> Result<bool, String> {
    for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
        if let Some(v) = obj.get(key) {
            if !v.is_number() {
                return Err(format!(
                    "json-schema: `{key}` must be a number, got {}",
                    json_type_name(v)
                ));
            }
        }
    }
    Ok(
        ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
            .iter()
            .any(|k| obj.contains_key(*k)),
    )
}

fn get_count(obj: &serde_json::Map<String, Value>, key: &str) -> Result<Option<u64>, String> {
    match obj.get(key) {
        None => Ok(None),
        Some(Value::Number(n)) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("json-schema: `{key}` must be a non-negative integer, got {n}")),
        Some(other) => Err(format!(
            "json-schema: `{key}` must be a non-negative integer, got {}",
            json_type_name(other)
        )),
    }
}

/// Read one inclusive integer bound, folding an exclusive bound into it. The
/// value must be integer-valued (a real-valued bound is a named refusal) and
/// within the exactly-representable `i64` range. Both spellings at once are
/// ambiguous and refused rather than silently picking one.
fn integer_bound(
    obj: &serde_json::Map<String, Value>,
    inclusive: &str,
    exclusive: &str,
    exclusive_delta: i64,
) -> Result<Option<i128>, String> {
    if obj.contains_key(inclusive) && obj.contains_key(exclusive) {
        return Err(format!(
            "json-schema: both `{inclusive}` and `{exclusive}` are present; their conjunction is \
             not compiled — keep one"
        ));
    }
    if let Some(v) = obj.get(inclusive) {
        return Ok(Some(integer_bound_value(v, inclusive)?));
    }
    if let Some(v) = obj.get(exclusive) {
        let base = integer_bound_value(v, exclusive)?;
        return Ok(Some(base + exclusive_delta as i128));
    }
    Ok(None)
}

fn integer_bound_value(v: &Value, key: &str) -> Result<i128, String> {
    let n = v.as_f64().ok_or_else(|| {
        format!(
            "json-schema: `{key}` must be a number, got {}",
            json_type_name(v)
        )
    })?;
    if n.fract() != 0.0 {
        return Err(format!(
            "json-schema: `{key}: {n}` is not integer-valued; only integer bounds are compiled \
             (see docs/GRAMMAR-DESIGN.md §3)"
        ));
    }
    if !(i64::MIN as f64..=i64::MAX as f64).contains(&n) || n.abs() > 9.007_199_254_740_992e15 {
        return Err(format!(
            "json-schema: `{key}: {n}` is outside the exactly-representable i64 range"
        ));
    }
    Ok(n as i128)
}

/// True for keywords that do not change the accepted language.
fn is_annotation(key: &str) -> bool {
    matches!(
        key,
        "$schema"
            | "$id"
            | "$comment"
            | "title"
            | "description"
            | "default"
            | "examples"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
            | "name"
            | "strict"
            | "$defs"
            | "definitions"
    )
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// A GBNF expression for exactly this JSON value (used by `enum`/`const`).
fn literal_expr(v: &Value) -> Result<String, String> {
    match v {
        Value::Null => Ok("\"null\"".to_string()),
        Value::Bool(true) => Ok("\"true\"".to_string()),
        Value::Bool(false) => Ok("\"false\"".to_string()),
        Value::Number(n) => Ok(format!("\"{n}\"")),
        Value::String(s) => Ok(json_string_literal(s)),
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(literal_expr)
                .collect::<Result<Vec<_>, String>>()?;
            if parts.is_empty() {
                Ok("\"[\" j-ws \"]\"".to_string())
            } else {
                Ok(format!(
                    "\"[\" j-ws {} j-ws \"]\"",
                    parts.join(" j-ws \",\" j-ws ")
                ))
            }
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    Ok(format!(
                        "{} j-ws \":\" j-ws {}",
                        json_string_literal(k),
                        literal_expr(v)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            if parts.is_empty() {
                Ok("\"{{\" j-ws \"}}\"".to_string())
            } else {
                Ok(format!(
                    "\"{{\" j-ws {} j-ws \"}}\"",
                    parts.join(" j-ws \",\" j-ws ")
                ))
            }
        }
    }
}

/// A GBNF string literal matching this exact JSON string (escaped per JSON's
/// string rules and emitted as GBNF escapes).
fn json_string_literal(s: &str) -> String {
    let mut out = String::from("\"\\\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\\\\\""),
            '\\' => out.push_str("\\\\\\\\"),
            '\n' => out.push_str("\\\\n"),
            '\r' => out.push_str("\\\\r"),
            '\t' => out.push_str("\\\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push_str("\\\"\"");
    out
}

// ===========================================================================
// Integer range -> GBNF expression
// ===========================================================================

/// A GBNF expression accepting exactly the decimal integers in `[lo, hi]`
/// (with a leading `-` for negatives): split by digit count, then descend the
/// common prefix of the bound strings.
pub fn int_range_expr(lo: i128, hi: i128) -> Result<String, String> {
    if lo > hi {
        return Err(format!(
            "json-schema: the integer bounds admit no value ({lo}..={hi})"
        ));
    }
    if lo >= 0 {
        return Ok(uint_range(lo as u128, hi as u128));
    }
    if hi <= 0 {
        let mag_lo = (-hi) as u128;
        let mag_hi = (-lo) as u128;
        if mag_lo == 0 {
            return Ok(format!("\"-\" ({}) | \"0\"", uint_range(1, mag_hi)));
        }
        return Ok(format!("\"-\" ({})", uint_range(mag_lo, mag_hi)));
    }
    // lo < 0 < hi
    Ok(format!(
        "(\"-\" ({}) | {})",
        uint_range(1, (-lo) as u128),
        uint_range(0, hi as u128)
    ))
}

/// Decimal strings (no leading zeros, `0` for zero) for values in `[lo, hi]`,
/// `0 <= lo <= hi`.
fn uint_range(lo: u128, hi: u128) -> String {
    if lo == 0 {
        return uint_up_to(hi);
    }
    let lo_s = lo.to_string();
    let hi_s = hi.to_string();
    let dl = lo_s.len();
    let dh = hi_s.len();
    if dl == dh {
        return fixed_len_range(&lo_s, &hi_s);
    }
    let mut alts = vec![fixed_len_range(&lo_s, &"9".repeat(dl))];
    for len in dl + 1..dh {
        alts.push(format!("[1-9][0-9]{{{}}}", len - 1));
    }
    alts.push(fixed_len_range(&format!("1{}", "0".repeat(dh - 1)), &hi_s));
    alts.join(" | ")
}

/// Decimal strings for values in `[0, n]`.
fn uint_up_to(n: u128) -> String {
    if n < 10 {
        return format!("[0-{}]", char::from(b'0' + n as u8));
    }
    let s = n.to_string();
    let d = s.len();
    let mut alts = vec!["\"0\"".to_string(), "[1-9]".to_string()];
    for len in 2..d {
        alts.push(format!("[1-9][0-9]{{{}}}", len - 1));
    }
    alts.push(fixed_len_range(&format!("1{}", "0".repeat(d - 1)), &s));
    alts.join(" | ")
}

/// Decimal strings of the same length in `[lo, hi]` (both without leading
/// zeros, `lo <= hi`).
fn fixed_len_range(lo: &str, hi: &str) -> String {
    debug_assert_eq!(lo.len(), hi.len());
    if lo == hi {
        return format!("\"{lo}\"");
    }
    let n = lo.len();
    let lb = lo.as_bytes();
    let hb = hi.as_bytes();
    let mut i = 0;
    while i < n && lb[i] == hb[i] {
        i += 1;
    }
    let prefix = &lo[..i];
    let l = (lb[i] - b'0') as u32;
    let h = (hb[i] - b'0') as u32;
    let suffix_len = n - i - 1;
    let mut alts: Vec<String> = Vec::new();
    if h > l + 1 {
        // A bare digit is an identifier to the GBNF parser, so literals are quoted.
        let mut a = String::new();
        if !prefix.is_empty() {
            a.push_str(&format!("\"{prefix}\""));
        }
        a.push_str(&format!(
            "[{}-{}]",
            char::from(b'0' + l as u8 + 1),
            char::from(b'0' + h as u8 - 1)
        ));
        if suffix_len > 0 {
            a.push_str(&format!("[0-9]{{{suffix_len}}}"));
        }
        alts.push(a);
    }
    // The `lo` branch: this digit fixed to lo[i], the suffix ranging up to 9…9.
    {
        let lit = format!("\"{prefix}{}\"", char::from(b'0' + l as u8));
        if suffix_len == 0 {
            alts.push(lit);
        } else {
            let sub_lo = &lo[i + 1..];
            let sub_hi = "9".repeat(suffix_len);
            alts.push(format!("{lit}({})", fixed_len_range(sub_lo, &sub_hi)));
        }
    }
    // The `hi` branch: this digit fixed to hi[i], the suffix ranging from 0…0.
    {
        let lit = format!("\"{prefix}{}\"", char::from(b'0' + h as u8));
        if suffix_len == 0 {
            alts.push(lit);
        } else {
            let sub_lo = "0".repeat(suffix_len);
            let sub_hi = &hi[i + 1..];
            alts.push(format!("{lit}({})", fixed_len_range(&sub_lo, sub_hi)));
        }
    }
    alts.join(" | ")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A synthetic vocabulary: every single byte, a few multi-byte / multi-char
    /// pieces, and two end-of-generation ids plus one empty piece. With every
    /// byte present, any JSON text can be driven token by token.
    fn byte_vocab() -> (Vec<Option<Box<[u8]>>>, Vec<bool>) {
        let mut pieces: Vec<Option<Box<[u8]>>> = Vec::new();
        for b in 0..=255u16 {
            pieces.push(Some(vec![b as u8].into_boxed_slice()));
        }
        // Extra multi-character / multi-byte pieces and the special ids. The
        // 中 piece is the 3 raw UTF-8 bytes, and the two prefixes exercise the
        // carried partial sequence; `b"ac"` exercises the longest-prefix path.
        for extra in [
            &b"true"[..],
            b"false",
            b"null",
            b"{\"",
            b"\":",
            b"}",
            b"[",
            b"]",
            b"\"a\"",
            b"ac",
            b"ab",
            b"abc",
            "\u{4e2d}".as_bytes(),
            &[0xE4u8, 0xB8],
            b"<eos>",
            b"<|im_end|>",
        ] {
            pieces.push(Some(extra.to_vec().into_boxed_slice()));
        }
        pieces.push(None); // an empty piece: never allowed
        let n = pieces.len();
        let mut eog = vec![false; n];
        eog[id_of(&pieces, b"<eos>") as usize] = true;
        eog[id_of(&pieces, b"<|im_end|>") as usize] = true;
        (pieces, eog)
    }

    fn id_of(pieces: &[Option<Box<[u8]>>], piece: &[u8]) -> u32 {
        pieces
            .iter()
            .position(|p| p.as_deref() == Some(piece))
            .expect("piece is in the synthetic vocabulary") as u32
    }

    fn gbnf(src: &str) -> Grammar {
        let (pieces, eog) = byte_vocab();
        Grammar::from_gbnf(src, pieces, eog)
            .unwrap_or_else(|e| panic!("compiling `{src}` failed: {e}"))
    }

    fn schema(s: Value) -> Grammar {
        let (pieces, eog) = byte_vocab();
        Grammar::from_json_schema(&s, pieces, eog)
            .unwrap_or_else(|e| panic!("compiling schema {s} failed: {e}"))
    }

    fn schema_err(s: Value) -> String {
        let (pieces, eog) = byte_vocab();
        Grammar::from_json_schema(&s, pieces, eog).expect_err("schema must be refused")
    }

    fn allowed(mask: &[u64], id: u32) -> bool {
        mask[(id / 64) as usize] & (1u64 << (id % 64)) != 0
    }

    fn mask_of(g: &Grammar, st: &mut GrammarState) -> Arc<[u64]> {
        g.mask(st).expect("mask")
    }

    // --- GBNF ---------------------------------------------------------------

    #[test]
    fn gbnf_literals_classes_and_dot() {
        let g = gbnf(r#"root ::= "a" [b-d] [^x-z] ."#);
        assert!(g.accepts(b"abcQ"), "grammar:\n{}", g.source());
        assert!(g.accepts("ade\u{00e9}".as_bytes()));
        assert!(!g.accepts(b"axcQ"), "x is excluded by [^x-z]");
        assert!(!g.accepts(b"ab"), "one codepoint short");
        assert!(!g.accepts(b"abcQ\n"), "the trailing byte is not allowed");
    }

    #[test]
    fn gbnf_any_dot_matches_newline_and_multibyte() {
        let g = gbnf("root ::= .");
        assert!(g.accepts(b"\n"));
        assert!(g.accepts("\u{4e2d}".as_bytes()));
        assert!(!g.accepts(b"ab"));
        assert!(
            !g.accepts(&[0xE4]),
            "an incomplete character is a prefix, not a sentence"
        );
        assert!(g.accepts_prefix(&[0xE4]));
    }

    #[test]
    fn gbnf_repetition_ranges() {
        let g = gbnf(r#"root ::= "a"{2,4}"#);
        assert!(!g.accepts(b"a"));
        assert!(g.accepts(b"aa"));
        assert!(g.accepts(b"aaaa"));
        assert!(!g.accepts(b"aaaaa"));

        let g = gbnf(r#"root ::= "a"{2,}"#);
        assert!(!g.accepts(b"a"));
        assert!(g.accepts(b"aa"));
        assert!(g.accepts(b"aaaaaaaaaaaaaaaaaaaa"));

        let g = gbnf(r#"root ::= "ab"+"c"?"#);
        assert!(g.accepts(b"abc"));
        assert!(g.accepts(b"abab"));
        assert!(!g.accepts(b"a"));

        let g = gbnf(r#"root ::= "a"{0}"#);
        assert!(g.accepts(b""));
        assert!(!g.accepts(b"a"));
    }

    #[test]
    fn gbnf_alternation_grouping_and_refs() {
        let g = gbnf(
            r#"
            root ::= greeting (" " name)?
            greeting ::= "hi" | "yo"
            name ::= [A-Z][a-z]+
            "#,
        );
        assert!(g.accepts(b"hi"));
        assert!(g.accepts(b"yo Bob"));
        assert!(!g.accepts(b"hey"));
        assert!(
            !g.accepts(b"hi bob"),
            "name needs an uppercase first letter"
        );
    }

    #[test]
    fn gbnf_comments_and_whitespace() {
        let g = gbnf("# leading comment\nroot  ::=  \"a\" # trailing\n   \"b\"\n");
        assert!(g.accepts(b"ab"));
    }

    #[test]
    fn gbnf_escapes() {
        let g = gbnf(r#"root ::= "\n" "\t" "\\" "\"" "\x41" "\u4e2d" [\]\-]+"#);
        let mut text = Vec::new();
        text.extend_from_slice(b"\n\t\\\"A");
        text.extend_from_slice("\u{4e2d}".as_bytes());
        text.extend_from_slice(b"]-");
        assert!(g.accepts(&text), "grammar:\n{}", g.source());
    }

    #[test]
    fn gbnf_refuses_unsupported_or_malformed_constructs() {
        let (pieces, eog) = byte_vocab();
        let cases: &[(&str, &str)] = &[
            (r#"root ::= [\d]"#, "unsupported escape"),
            (r#"root ::= "\p{L}""#, "unsupported escape"),
            (r#"root ::= "a" "unterminated"#, "unterminated string"),
            (r#"root ::= []"#, "empty character class"),
            (r#"root ::= [z-a]"#, "reversed"),
            (r#"root ::= "a"{3,2}"#, "upper bound below"),
            (r#"root ::= "a"{1,2000}"#, "exceeds the supported limit"),
            (r#"root ::= "a" ("#, "missing `)`"),
            (r#"root ::= missing"#, "undefined rule"),
            (
                r#"root ::= "a"
                root ::= "b""#,
                "duplicate rule",
            ),
            (r#"other ::= "a""#, "no rule named `root`"),
        ];
        for (src, needle) in cases {
            let err = Grammar::from_gbnf(src, pieces.clone(), eog.clone())
                .err()
                .unwrap_or_else(|| panic!("`{src}` should be refused"));
            if !needle.is_empty() {
                assert!(
                    err.contains(needle),
                    "`{src}` -> `{err}` (wanted `{needle}`)"
                );
            }
        }
        // Trailing whitespace/comments after the last rule are not a rule.
        assert!(Grammar::from_gbnf("root ::= \"a\"\n# done\n", pieces, eog).is_ok());
    }

    #[test]
    fn gbnf_detects_left_recursion() {
        let (pieces, eog) = byte_vocab();
        let err = Grammar::from_gbnf("root ::= root \"a\"\n", pieces, eog)
            .expect_err("left recursion must be refused");
        assert!(err.contains("left recursion"), "{err}");
    }

    // --- JSON Schema --------------------------------------------------------

    #[test]
    fn json_object_required_properties_and_closed_shape() {
        let g = schema(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false
        }));
        for ok in [
            &br#"{"a":"x"}"#[..],
            br#"{"a":"x","b":1}"#,
            br#"{ "a" : "x" , "b" : -3 }"#,
        ] {
            assert!(
                g.accepts(ok),
                "{} must parse under\n{}",
                String::from_utf8_lossy(ok),
                g.source()
            );
        }
        for bad in [
            &br#"{}"#[..],
            br#"{"b":1}"#,
            br#"{"a":1}"#,
            br#"{"a":"x","c":2}"#,
            br#"{"a":"x","b":1,}"#,
            br#"{"b":1,"a":"x"}"#,
            br#"[1]"#,
        ] {
            assert!(
                !g.accepts(bad),
                "{} must NOT parse under\n{}",
                String::from_utf8_lossy(bad),
                g.source()
            );
        }
    }

    #[test]
    fn json_object_additional_properties_forms() {
        // Open by default: extras allowed anywhere a required member is not pending.
        let open = schema(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        }));
        assert!(open.accepts(br#"{"a":1}"#));
        assert!(open.accepts(br#"{"a":1,"x":[1,2]}"#));
        assert!(
            !open.accepts(br#"{"x":true,"a":1}"#),
            "an extra may not precede a required declared member (declaration order)"
        );
        assert!(!open.accepts(br#"{}"#));

        // A schema-valued additionalProperties.
        let typed = schema(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "boolean"}
        }));
        assert!(typed.accepts(br#"{"a":1,"x":true}"#));
        assert!(!typed.accepts(br#"{"a":1,"x":3}"#));
        assert!(typed.accepts(br#"{}"#), "nothing is required here");
    }

    #[test]
    fn json_array_items_and_length_bounds() {
        let g = schema(json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 3
        }));
        assert!(!g.accepts(b"[]"));
        assert!(!g.accepts(b"[1]"));
        assert!(g.accepts(b"[1,2]"));
        assert!(g.accepts(b"[1, 2 , 3]"));
        assert!(!g.accepts(b"[1,2,3,4]"));
        assert!(!g.accepts(br#"["a","b"]"#));

        let open = schema(json!({"type": "array", "items": {"type": "string"}}));
        assert!(open.accepts(b"[]"));
        assert!(open.accepts(br#"["a","b","c","d"]"#));
        assert!(!open.accepts(b"[1]"));

        let loose = schema(json!({"type": "array", "minItems": 1}));
        assert!(!loose.accepts(b"[]"));
        assert!(loose.accepts(br#"[{"deep":[1,2]},null]"#));
    }

    #[test]
    fn json_array_prefix_items() {
        let closed = schema(json!({
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}],
            "items": false,
            "minItems": 1
        }));
        assert!(closed.accepts(br#"["a"]"#));
        assert!(closed.accepts(br#"["a",1]"#));
        assert!(!closed.accepts(b"[]"));
        assert!(!closed.accepts(b"[1]"));
        assert!(!closed.accepts(br#"["a",1,2]"#));

        let open = schema(json!({
            "type": "array",
            "prefixItems": [{"type": "string"}],
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 3
        }));
        assert!(open.accepts(br#"["a"]"#));
        assert!(open.accepts(br#"["a",1,2]"#));
        assert!(!open.accepts(br#"["a",1,2,3]"#));
        assert!(!open.accepts(br#"[1]"#));
    }

    #[test]
    fn json_scalars_and_integer_bounds() {
        let g = schema(json!({
            "type": "integer",
            "minimum": -3,
            "maximum": 5
        }));
        for ok in ["-3", "5", "0", "-1", "4"] {
            assert!(g.accepts(ok.as_bytes()), "{ok} must be accepted");
        }
        for bad in ["-4", "6", "3.5", "05", "+3", ""] {
            assert!(!g.accepts(bad.as_bytes()), "{bad} must be rejected");
        }

        let excl = schema(json!({
            "type": "integer",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 3
        }));
        assert!(!excl.accepts(b"0"));
        assert!(excl.accepts(b"1"));
        assert!(excl.accepts(b"2"));
        assert!(!excl.accepts(b"3"));

        let string = schema(json!({"type": "string"}));
        assert!(string.accepts(br#""""#));
        assert!(string.accepts(br#""a\"b\n\u00e9""#));
        assert!(!string.accepts(br#""unterminated"#));
        assert!(
            !string.accepts(br#""raw\q""#),
            "an invalid JSON escape must be refused"
        );

        let number = schema(json!({"type": "number"}));
        assert!(number.accepts(b"-1.5e-3"));
        assert!(number.accepts(b"0"));
        assert!(!number.accepts(b"1."));

        assert!(schema(json!({"type": "boolean"})).accepts(b"true"));
        assert!(!schema(json!({"type": "boolean"})).accepts(b"1"));
        assert!(schema(json!({"type": "null"})).accepts(b"null"));

        let union = schema(json!({"type": ["string", "null"]}));
        assert!(union.accepts(br#""x""#));
        assert!(union.accepts(b"null"));
        assert!(!union.accepts(b"1"));
    }

    #[test]
    fn json_enum_and_const() {
        let g = schema(json!({"enum": ["a", 1, null, true]}));
        for ok in [br#""a""#.as_slice(), b"1", b"null", b"true"] {
            assert!(
                g.accepts(ok),
                "{} must be accepted",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [br#""b""#.as_slice(), b"1.5", b"false", br#"["a"]"#] {
            assert!(!g.accepts(bad));
        }

        let c = schema(json!({"const": {"a": 1, "b": [true, null]}}));
        assert!(
            c.accepts(br#"{"a":1,"b":[true,null]}"#),
            "grammar:\n{}",
            c.source()
        );
        assert!(!c.accepts(br#"{"a":1,"b":[true,false]}"#));
        assert!(!c.accepts(br#"{"a":1}"#));
    }

    #[test]
    fn json_any_of_union() {
        let g = schema(json!({
            "anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]
        }));
        assert!(g.accepts(br#""x""#));
        assert!(g.accepts(b"7"));
        assert!(!g.accepts(b"-1"));
        assert!(!g.accepts(b"1.5"));
        assert!(!g.accepts(b"true"));

        let one = schema(json!({"oneOf": [{"const": 1}, {"const": 2}]}));
        assert!(one.accepts(b"1"));
        assert!(one.accepts(b"2"));
        assert!(!one.accepts(b"3"));
    }

    #[test]
    fn json_defs_ref_and_recursion() {
        let g = schema(json!({
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "v": {"type": "integer"},
                        "next": {"$ref": "#/$defs/node"}
                    },
                    "required": ["v"],
                    "additionalProperties": false
                }
            },
            "$ref": "#/$defs/node"
        }));
        assert!(g.accepts(br#"{"v":1}"#), "grammar:\n{}", g.source());
        assert!(g.accepts(br#"{"next":{"next":{"v":3},"v":2},"v":1}"#));
        assert!(!g.accepts(br#"{"next":{"v":1}}"#));
        assert!(!g.accepts(br#"{"next":{},"v":1}"#));
        assert!(!g.accepts(br#"{"next":{"w":2},"v":1}"#));
    }

    #[test]
    fn json_schema_refusals_are_named() {
        let cases: &[(Value, &str)] = &[
            (json!({"type": "string", "pattern": "^a"}), "pattern"),
            (json!({"type": "string", "minLength": 2}), "minLength"),
            (json!({"allOf": [{"type": "string"}]}), "allOf"),
            (json!({"not": {"type": "string"}}), "not"),
            (
                json!({"if": {"type": "string"}, "then": {"minLength": 1}}),
                "if",
            ),
            (json!({"type": "number", "minimum": 0}), "numeric bounds"),
            (json!({"type": "integer", "minimum": 0.5}), "integer-valued"),
            (json!({"$ref": "https://example.com/schema.json"}), "$ref"),
            (json!({"$ref": "#/$defs/missing"}), "does not resolve"),
            (
                json!({"type": "array", "items": [{"type": "string"}]}),
                "draft-07",
            ),
            (
                json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
                "required",
            ),
            (json!({"enum": []}), "empty `enum`"),
            (
                json!({"type": "array", "minItems": 3, "maxItems": 1}),
                "exceeds `maxItems`",
            ),
            (
                json!({"type": "string", "properties": {}}),
                "object keywords",
            ),
            (
                json!({"type": "integer", "minimum": 5, "maximum": 1}),
                "admit no value",
            ),
            (json!({"const": 1, "minimum": 0}), "combined with"),
            (
                json!({"type": "integer", "minimum": 0, "exclusiveMinimum": 1}),
                "both",
            ),
        ];
        for (s, needle) in cases {
            let err = schema_err(s.clone());
            assert!(
                err.contains(needle),
                "schema {s} -> `{err}` (wanted `{needle}`)"
            );
        }
    }

    // --- Token advancement --------------------------------------------------

    #[test]
    fn token_advancement_handles_partial_utf8() {
        let (pieces, eog) = byte_vocab();
        let g = Grammar::from_gbnf("root ::= \"\u{4e2d}\"", pieces.clone(), eog).unwrap();
        let mut st = g.state();

        // Byte at a time: E4, B8, AD, with the partial sequence carried.
        let e4 = id_of(&pieces, &[0xE4]);
        let b8 = id_of(&pieces, &[0xB8]);
        let ad = id_of(&pieces, &[0xAD]);
        let whole = id_of(&pieces, "\u{4e2d}".as_bytes());
        let eos = id_of(&pieces, b"<eos>");

        let mask = mask_of(&g, &mut st);
        assert!(
            allowed(&mask, e4),
            "the first byte of the character must be allowed"
        );
        assert!(allowed(&mask, whole), "the whole character must be allowed");
        assert!(
            !allowed(&mask, id_of(&pieces, &[0x80])),
            "an invalid UTF-8 byte must never be allowed"
        );
        assert!(
            !allowed(&mask, eos),
            "EOG is not legal before the character"
        );

        g.accept_token(&mut st, e4).unwrap();
        assert_eq!(st.pending_bytes(), 1, "E4 is carried");
        g.accept_token(&mut st, b8).unwrap();
        assert_eq!(st.pending_bytes(), 2);
        g.accept_token(&mut st, ad).unwrap();
        assert_eq!(st.pending_bytes(), 0);
        assert!(st.is_accepting(), "the complete character ends the grammar");

        // The same character in one token, from a fresh state.
        let mut st = g.state();
        g.accept_token(&mut st, whole).unwrap();
        assert!(st.is_accepting());

        // A two-byte prefix piece is allowed and carried, then completed.
        let mut st = g.state();
        g.accept_token(&mut st, id_of(&pieces, &[0xE4, 0xB8]))
            .unwrap();
        assert_eq!(st.pending_bytes(), 2);
        g.accept_token(&mut st, ad).unwrap();
        assert!(st.is_accepting());
    }

    #[test]
    fn token_advancement_rejects_invalid_utf8() {
        let (pieces, eog) = byte_vocab();
        let g = Grammar::from_gbnf("root ::= .", pieces.clone(), eog).unwrap();
        let mut st = g.state();
        let stray = id_of(&pieces, &[0x80]);
        let mask = mask_of(&g, &mut st);
        assert!(!allowed(&mask, stray));
        let err = g.accept_token(&mut st, stray).expect_err("must be refused");
        assert!(err.contains("invalid UTF-8"), "{err}");
        assert_eq!(
            st.pending_bytes(),
            0,
            "a refused token must not mutate the state"
        );
    }

    /// A partial-UTF-8 token is only legal when the state can actually consume a
    /// character that the pending bytes can complete to. This is the gate that a
    /// real-model run tripped: after a JSON object closed, a lone 0xE4 leader was
    /// accepted and stranded the run (the response then ended with U+FFFD).
    #[test]
    fn partial_utf8_is_only_allowed_when_it_can_still_complete() {
        let (pieces, eog) = byte_vocab();
        let e4 = id_of(&pieces, &[0xE4]);

        // `root ::= "a"`: U+0061 cannot start with 0xE4 (which covers U+4000..U+4FFF).
        let g = Grammar::from_gbnf("root ::= \"a\"", pieces.clone(), eog.clone()).unwrap();
        let mut st = g.state();
        assert!(!allowed(&mask_of(&g, &mut st), e4));
        assert!(g.accept_token(&mut st, e4).is_err());

        // `.` accepts any codepoint, so the leader is a legal prefix.
        let g = Grammar::from_gbnf("root ::= .", pieces.clone(), eog.clone()).unwrap();
        let mut st = g.state();
        assert!(allowed(&mask_of(&g, &mut st), e4));

        // A class that contains U+4E2D (the completed character) accepts the leader.
        let g =
            Grammar::from_gbnf(r"root ::= [\u4e00-\u4e2f]", pieces.clone(), eog.clone()).unwrap();
        let mut st = g.state();
        assert!(allowed(&mask_of(&g, &mut st), e4));

        // A class whose codepoints are two-byte encoded does not.
        let g =
            Grammar::from_gbnf(r"root ::= [\u0100-\u0200]", pieces.clone(), eog.clone()).unwrap();
        let mut st = g.state();
        assert!(!allowed(&mask_of(&g, &mut st), e4));

        // A JSON string's `[^"\\\u0000-\u001f]` still accepts a multi-byte
        // character, so the leader is legal inside a string.
        let g = Grammar::from_json_schema(&json!({"type": "string"}), pieces.clone(), eog).unwrap();
        let mut st = g.state();
        g.accept_token(&mut st, id_of(&pieces, b"\"")).unwrap();
        assert!(allowed(&mask_of(&g, &mut st), e4));
    }

    #[test]
    fn completion_ranges_are_exact() {
        assert_eq!(completion_range(&[0xE4]), Some((0x4000, 0x4FFF)));
        assert_eq!(completion_range(&[0xE4, 0xB8]), Some((0x4E00, 0x4E3F)));
        assert_eq!(completion_range(&[0x41]), None, "a complete ASCII byte");
        assert_eq!(completion_range(&[0x80]), None, "a stray continuation byte");
        assert_eq!(
            completion_range(&[0xE4, 0xB8, 0xAD]),
            None,
            "already complete"
        );
        assert_eq!(covers(&[(0x00, 0x1F), (0x22, 0x22)], 0x00, 0x1F), true);
        assert_eq!(covers(&[(0x00, 0x1F), (0x22, 0x22)], 0x00, 0x22), false);
        assert_eq!(non_surrogate_parts(0xD000, 0xE100).len(), 2);

        // Overlong forms and the special leaders: only the *valid* completions
        // count. A 3-byte form of U+0061 (0xE0 0x80 ...) is not valid UTF-8, so
        // `[0xE0]` must not be treated as a way to reach ASCII.
        assert_eq!(completion_range(&[0xC0]), None, "overlong 2-byte leader");
        assert_eq!(completion_range(&[0xC1]), None, "overlong 2-byte leader");
        assert_eq!(completion_range(&[0xC2]), Some((0x80, 0xBF)));
        assert_eq!(completion_range(&[0xE0]), Some((0x800, 0xFFF)));
        assert_eq!(
            completion_range(&[0xE0, 0x80]),
            None,
            "overlong first continuation"
        );
        assert_eq!(completion_range(&[0xED]), Some((0xD000, 0xD7FF)));
        assert_eq!(completion_range(&[0xED, 0xA0]), None, "surrogate block");
        assert_eq!(completion_range(&[0xF0]), Some((0x10000, 0x3FFFF)));
        assert_eq!(
            completion_range(&[0xF0, 0x80]),
            None,
            "overlong first continuation"
        );
        assert_eq!(completion_range(&[0xF4]), Some((0x100000, 0x10FFFF)));
        assert_eq!(completion_range(&[0xF5]), None, "above U+10FFFF");
        assert_eq!(
            completion_range(&[0xE4, 0xB8, 0x41]),
            None,
            "bad continuation"
        );

        // U+0061 ('a') is not in [0x800, 0xFFF], so an 0xE0 leader can never
        // reach it: the mask must reject that token (the live-server defect).
        let (lo, hi) = completion_range(&[0xE0]).unwrap();
        assert!(!(lo..=hi).contains(&0x61));
    }

    #[test]
    fn accept_reports_the_longest_accepted_prefix() {
        let (pieces, eog) = byte_vocab();
        let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
        let mut st = g.state();
        let partial = id_of(&pieces, b"ac");
        let err = g
            .accept_token(&mut st, partial)
            .expect_err("'ac' must be refused");
        assert!(
            err.contains("1 byte(s)"),
            "the error must name the longest accepted prefix: {err}"
        );
        assert!(g.accepts_prefix(b"a"));
        assert!(!g.accepts_prefix(b"ac"));
    }

    #[test]
    fn mask_is_empty_when_nothing_can_continue() {
        // A vocabulary without 'b' cannot complete `"ab"` after 'a'.
        let pieces = vec![Some(b"a".to_vec().into_boxed_slice())];
        let eog = vec![false];
        let g = Grammar::from_gbnf("root ::= \"ab\"", pieces, eog).unwrap();
        let mut st = g.state();
        g.accept_token(&mut st, 0).unwrap();
        let mask = mask_of(&g, &mut st);
        assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), 0);
    }

    #[test]
    fn eog_is_allowed_only_at_an_accepting_state() {
        let (pieces, eog) = byte_vocab();
        let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
        let eos = id_of(&pieces, b"<eos>");
        let im_end = id_of(&pieces, b"<|im_end|>");
        let empty = (0..pieces.len() as u32)
            .find(|id| pieces[*id as usize].is_none())
            .expect("the empty-piece token is in the vocabulary");

        let mut st = g.state();
        let mask = mask_of(&g, &mut st);
        assert!(!allowed(&mask, eos));
        assert!(!allowed(&mask, empty), "an empty piece is never a member");
        g.accept_token(&mut st, id_of(&pieces, b"a")).unwrap();
        let mask = mask_of(&g, &mut st);
        assert!(!allowed(&mask, eos), "not accepting yet");
        g.accept_token(&mut st, id_of(&pieces, b"b")).unwrap();
        assert!(st.is_accepting());
        let mask = mask_of(&g, &mut st);
        assert!(allowed(&mask, eos));
        assert!(allowed(&mask, im_end));
        g.accept_token(&mut st, eos).unwrap();
        let mask = mask_of(&g, &mut st);
        assert!(
            allowed(&mask, eos),
            "EOG stays legal once generation has ended"
        );
        assert!(
            !allowed(&mask, id_of(&pieces, b"a")),
            "nothing else is legal after EOG"
        );
        g.accept_token(&mut st, im_end)
            .expect_err("a second EOG must be refused");
    }

    #[test]
    fn mask_is_cached_per_state() {
        let (pieces, eog) = byte_vocab();
        let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
        let mut st = g.state();
        let _ = mask_of(&g, &mut st);
        assert_eq!(st.cached_states(), 1);
        let _ = mask_of(&g, &mut st);
        assert_eq!(st.cached_states(), 1, "the same state must reuse its mask");
        g.accept_token(&mut st, id_of(&pieces, b"a")).unwrap();
        let _ = mask_of(&g, &mut st);
        assert_eq!(st.cached_states(), 2, "a new state computes a new mask");
    }

    // --- Integer ranges -----------------------------------------------------

    #[test]
    fn integer_range_expressions_match_their_bounds() {
        let ranges: &[(i128, i128)] = &[
            (0, 0),
            (0, 9),
            (5, 17),
            (10, 10),
            (99, 102),
            (0, 1000),
            (-5, -1),
            (-100, -99),
            (-5, 5),
            (-3, 0),
            (250, 250),
        ];
        let (pieces, eog) = byte_vocab();
        for &(lo, hi) in ranges {
            let expr = int_range_expr(lo, hi).expect("expression");
            let g = Grammar::from_gbnf(&format!("root ::= {expr}"), pieces.clone(), eog.clone())
                .unwrap_or_else(|e| panic!("[{lo},{hi}] `{expr}` failed to compile: {e}"));
            for v in [lo, hi, (lo + hi) / 2] {
                assert!(
                    g.accepts(v.to_string().as_bytes()),
                    "[{lo},{hi}] must accept {v}\ngrammar:\n{}",
                    g.source()
                );
            }
            for v in [lo - 1, hi + 1] {
                assert!(
                    !g.accepts(v.to_string().as_bytes()),
                    "[{lo},{hi}] must reject {v}\ngrammar:\n{}",
                    g.source()
                );
            }
        }
        assert!(int_range_expr(3, 2).is_err(), "an empty range is refused");
    }

    #[test]
    fn generated_schema_grammar_is_reparseable() {
        // The schema front end emits GBNF; compiling it twice must give the same
        // language (and proves the emitted text is in the supported subset).
        let src = json!({
            "type": "object",
            "properties": {
                "when": {"type": "integer", "minimum": 1900, "maximum": 2100},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "kind": {"enum": ["a", "b"]}
            },
            "required": ["when", "kind"],
            "additionalProperties": false
        });
        let g = schema(src.clone());
        let again = gbnf(g.source());
        for text in [
            &br#"{"kind":"a","when":2024}"#[..],
            br#"{"kind":"b","tags":["x","y"],"when":1900}"#,
        ] {
            assert!(g.accepts(text));
            assert!(again.accepts(text), "round-trip must preserve the language");
        }
        for text in [&br#"{"kind":"a","when":1899}"#[..], br#"{"kind":"a"}"#] {
            assert!(!g.accepts(text));
            assert!(!again.accepts(text));
        }
    }
}
