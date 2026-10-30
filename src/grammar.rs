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
    /// Pieces the last mask computation walked. Test-only instrumentation
    /// (#126): `#[cfg(test)]` keeps the counter out of production builds.
    #[cfg(test)]
    walked: usize,
}

impl GrammarState {
    /// The state is a complete, acceptable string (end-of-generation is legal).
    /// Test-only (#238): driven by `grammar::tests::eog_is_allowed_only_at_an_accepting_state and sampler::tests::grammar_mask_decides_the_greedy_choice_and_advances`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn is_accepting(&self) -> bool {
        self.accepting
    }

    /// Number of states whose mask is cached (bounded by `MAX_MASK_CACHE`).
    /// Test-only (#238): driven by `grammar::tests::mask_is_cached_per_state and server::chat::tests::real_model_json_schema_generation_parses`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn cached_states(&self) -> usize {
        self.cache.len()
    }

    /// Record the work the last mask computation did. The production arm is a
    /// no-op: the count exists so a gate can bound the mask by work instead of
    /// by a clock (GATE-CONTRACT §4), and `#[cfg(test)]` is what keeps the
    /// counter out of the shipping struct.
    #[cfg(test)]
    fn set_walked(&mut self, walked: usize) {
        self.walked = walked;
    }

    #[cfg(not(test))]
    fn set_walked(&mut self, _walked: usize) {}

    /// Token pieces the last [`Grammar::mask`] call walked — the work the mask
    /// did, which is the machine-independent half of its cost (#126). Test-only
    /// (#126): driven by
    /// `grammar::tests::the_index_walks_only_the_accepted_buckets` and
    /// `server::chat::tests::real_model_json_schema_generation_parses`;
    /// `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn pieces_walked(&self) -> usize {
        self.walked
    }
}

/// Token ids grouped by the first codepoint of their piece, built once when the
/// grammar is compiled.
///
/// The mask only needs a piece's **first** codepoint to know whether the piece
/// can begin at all; the memoized transition table decides the rest. Indexing on
/// it lets a state look at the buckets its frontier accepts instead of walking
/// the whole vocabulary.
///
/// Two kinds of token are deliberately absent:
/// - a piece whose leading bytes are an **incomplete** UTF-8 sequence has no
///   first codepoint, and whether it is accepted depends on the codepoint the
///   frontier would complete it to — those ids go to `headless` and are always
///   walked;
/// - a `None` (empty) piece, or one starting with an **invalid** UTF-8 byte, is
///   never accepted on the piece path and is not indexed at all.
///
/// End-of-generation ids are held separately by [`Grammar`]: they are admitted
/// on the state's own condition, not on a codepoint.
struct TokenIndex {
    /// Ascending by first codepoint, one entry per distinct codepoint.
    buckets: Vec<(u32, Vec<u32>)>,
    /// Token ids whose piece has no complete first codepoint.
    headless: Vec<u32>,
}

impl TokenIndex {
    fn build(pieces: &[Option<Box<[u8]>>], eog_ids: &[u32]) -> TokenIndex {
        let mut by_cp: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut headless: Vec<u32> = Vec::new();
        for (id, piece) in pieces.iter().enumerate() {
            let Some(piece) = piece.as_deref() else {
                continue;
            };
            if eog_ids.contains(&(id as u32)) {
                continue;
            }
            match decode_at(piece, 0) {
                Utf8Step::Cp(cp, _) => by_cp.entry(cp).or_default().push(id as u32),
                Utf8Step::Incomplete => headless.push(id as u32),
                Utf8Step::Invalid => {}
            }
        }
        let mut buckets: Vec<(u32, Vec<u32>)> = by_cp.into_iter().collect();
        buckets.sort_unstable_by_key(|&(cp, _)| cp);
        TokenIndex { buckets, headless }
    }
}

/// The codepoints a frontier can consume next, for the first-codepoint filter.
///
/// `wide` marks the cases a filter cannot help — a `.` (`Inst::Any`) and a
/// negated class both accept nearly the whole codepoint space — so the mask takes
/// the whole-vocabulary walk instead of enumerating buckets.
struct Accepted {
    wide: bool,
    /// Ascending, disjoint, non-adjacent ranges.
    ranges: Vec<(u32, u32)>,
}

impl Accepted {
    /// Total number of codepoints accepted.
    fn width(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(a, b)| u64::from(b) - u64::from(a) + 1)
            .sum()
    }

    /// Is `cp` accepted? Asked on the bucket-scan path, where `ranges` is the
    /// whole answer.
    fn contains(&self, cp: u32) -> bool {
        let i = self.ranges.partition_point(|&(a, _)| a <= cp);
        i > 0 && self.ranges[i - 1].1 >= cp
    }

    /// Every accepted codepoint, in order. Only called when `width()` is known
    /// to be no larger than the bucket count (see [`Grammar::mask_buckets`]).
    fn codepoints(&self) -> impl Iterator<Item = u32> + '_ {
        self.ranges.iter().flat_map(|&(a, b)| a..=b)
    }
}

/// How one state's mask is computed (see [`Grammar::plan`]).
enum Walk {
    /// Every token id, in order.
    Whole,
    /// Only the tokens whose first codepoint the frontier accepts.
    Buckets(Accepted),
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
    /// End-of-generation token ids, ascending.
    eog_ids: Vec<u32>,
    /// The GBNF text this grammar was compiled from (generated for a schema).
    source: String,
    /// Token ids grouped by the first codepoint of their piece (#126).
    index: TokenIndex,
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grammar")
            .field("rules", &self.rules.len())
            .field("root", &self.names[self.root as usize])
            .field("n_vocab", &self.n_vocab)
            .field("eog", &self.eog_ids.len())
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
            #[cfg(test)]
            walked: 0,
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
        let eog_ids: Vec<u32> = eog
            .iter()
            .enumerate()
            .filter(|&(_, flag)| *flag)
            .map(|(id, _)| id as u32)
            .collect();
        let index = TokenIndex::build(&pieces, &eog_ids);
        let g = Grammar {
            rules,
            names,
            root,
            n_vocab: pieces.len(),
            pieces,
            eog_ids,
            source: source.to_string(),
            index,
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
    /// Test-only (#238): driven by `grammar::tests::accept_reports_the_longest_accepted_prefix and server::chat::tests::real_model_max_length_output_is_a_valid_prefix`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn accepts_prefix(&self, bytes: &[u8]) -> bool {
        self.consume(&self.initial_key(), bytes).is_ok()
    }

    /// True when the byte string is a complete sentence of the grammar.
    /// Test-only (#238): driven by `grammar::tests::gbnf_literals_classes_and_dot`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn accepts(&self, bytes: &[u8]) -> bool {
        match self.consume(&self.initial_key(), bytes) {
            Ok(next) => self.state_key_accepting(&next),
            Err(_) => false,
        }
    }

    /// Test-only (#238): driven by `grammar::tests::gbnf_literals_classes_and_dot (through accepts/accepts_prefix)`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
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
    ///
    /// [`Grammar::plan`] decides which walk runs: a frontier that accepts few
    /// codepoints is filtered through the first-codepoint index (#126), and one
    /// that accepts nearly everything keeps the whole-vocabulary walk.
    pub fn mask(&self, st: &mut GrammarState) -> Result<Arc<[u64]>, String> {
        if let Some(cached) = st.cache.get(&st.inner) {
            return Ok(cached.clone());
        }
        let (mask, walked) = match self.plan(&st.inner) {
            Walk::Whole => self.mask_whole(&st.inner, st.accepting)?,
            Walk::Buckets(accepted) => self.mask_buckets(&st.inner, st.accepting, &accepted)?,
        };
        st.set_walked(walked);
        if st.cache.len() >= MAX_MASK_CACHE {
            st.cache.clear();
        }
        st.cache.insert(st.inner.clone(), mask.clone());
        Ok(mask)
    }

    /// How a state's mask is computed.
    fn plan(&self, key: &StateKey) -> Walk {
        // A carried partial UTF-8 sequence is matched continuation-first against
        // the piece's bytes, so a first-codepoint bucket says nothing about it;
        // and a finished state admits only EOG.
        if !key.partial.is_empty() || key.done {
            return Walk::Whole;
        }
        let accepted = self.first_codepoints(&key.stacks);
        if accepted.wide || accepted.width() >= self.n_vocab as u64 {
            // The frontier accepts at least as many codepoints as there are
            // tokens: the buckets would add one test per bucket to a walk that
            // already visits nearly every token.
            return Walk::Whole;
        }
        Walk::Buckets(accepted)
    }

    /// The codepoints a closed set of stacks can consume next. Only each stack's
    /// current instruction matters, and `close` leaves every live stack on a
    /// consuming instruction or on `Ret`.
    fn first_codepoints(&self, stacks: &[Vec<[u32; 2]>]) -> Accepted {
        let mut wide = false;
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for st in stacks {
            let f = *st.last().expect("stacks are never empty");
            match &self.rules[f[0] as usize][f[1] as usize] {
                Inst::Any => {
                    wide = true;
                    break;
                }
                Inst::Cp(c) => {
                    if !(0xD800..=0xDFFF).contains(c) {
                        ranges.push((*c, *c));
                    }
                }
                Inst::Class {
                    negated: false,
                    ranges: rs,
                } => {
                    for &(a, b) in rs {
                        ranges.extend(non_surrogate_parts(a, b));
                    }
                }
                // A negated class accepts all but a few codepoints -- a `.` with
                // exceptions -- so filtering on it removes nothing.
                Inst::Class { negated: true, .. } => {
                    wide = true;
                    break;
                }
                Inst::Ret | Inst::Split(_, _) | Inst::Jump(_) | Inst::Call(_) => {}
            }
        }
        Accepted {
            wide,
            ranges: if wide {
                Vec::new()
            } else {
                merge_ranges(ranges)
            },
        }
    }

    /// The whole-vocabulary walk: what the mask did before the index, and still
    /// the path for a wide frontier and for a carried partial sequence. Returns
    /// the mask and the number of pieces it examined.
    fn mask_whole(&self, key: &StateKey, accepting: bool) -> Result<(Arc<[u64]>, usize), String> {
        let mut bits = self.eog_bits(key, accepting);
        if key.done {
            return Ok((bits.into(), 0));
        }
        let carried = !key.partial.is_empty();
        let mut walk = MaskWalk::new(self, &key.stacks);
        for id in 0..self.n_vocab {
            if self.eog_ids.contains(&(id as u32)) {
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
                walk.walked += 1;
                self.consume(key, piece).is_ok()
            } else {
                walk.allows(piece)
            };
            if allowed {
                set_bit(&mut bits, id);
            }
        }
        Ok((bits.into(), walk.walked))
    }

    /// The indexed walk (#126): first the tokens whose piece has no first
    /// codepoint, then the tokens the frontier's first codepoints can begin.
    fn mask_buckets(
        &self,
        key: &StateKey,
        accepting: bool,
        accepted: &Accepted,
    ) -> Result<(Arc<[u64]>, usize), String> {
        let mut bits = self.eog_bits(key, accepting);
        let mut walk = MaskWalk::new(self, &key.stacks);
        for &id in &self.index.headless {
            if walk.allows(self.piece(id)) {
                set_bit(&mut bits, id as usize);
            }
        }
        if accepted.width() <= self.index.buckets.len() as u64 {
            // Few accepted codepoints: visit only their buckets.
            for cp in accepted.codepoints() {
                if let Ok(i) = self.index.buckets.binary_search_by_key(&cp, |&(c, _)| c) {
                    walk.bucket(&self.index.buckets[i].1, &mut bits);
                }
            }
        } else {
            // More accepted codepoints than buckets: one range test per bucket is
            // cheaper than one bucket lookup per codepoint.
            for (cp, ids) in &self.index.buckets {
                if accepted.contains(*cp) {
                    walk.bucket(ids, &mut bits);
                }
            }
        }
        Ok((bits.into(), walk.walked))
    }

    /// The mask's starting bits: an EOG id is admitted exactly when the state is
    /// a complete sentence (`done`) or a character-aligned prefix of one.
    fn eog_bits(&self, key: &StateKey, accepting: bool) -> Vec<u64> {
        let mut bits = vec![0u64; self.n_vocab.div_ceil(64)];
        if key.done || accepting {
            for &id in &self.eog_ids {
                set_bit(&mut bits, id as usize);
            }
        }
        bits
    }

    fn piece(&self, id: u32) -> &[u8] {
        self.pieces[id as usize]
            .as_deref()
            .expect("indexed token ids always have a piece")
    }

    /// The whole-vocabulary walk for the state a [`GrammarState`] holds, which
    /// [`Grammar::mask`] also runs whenever the frontier is wide. Test-only
    /// (#126): driven by
    /// `server::chat::tests::real_model_json_schema_generation_parses`, which
    /// compares it against the indexed mask bit for bit over the real
    /// 151,936-token vocabulary; `#[cfg(test)]` keeps it out of production
    /// builds.
    #[cfg(test)]
    pub(crate) fn mask_whole_for_tests(&self, st: &GrammarState) -> Arc<[u64]> {
        self.mask_whole(&st.inner, st.accepting).expect("mask").0
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
        if self.eog_ids.contains(&id) {
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

/// The per-mask-computation machinery: the interned successor states and the
/// DFA-style transition memo.
///
/// Within one mask computation the start state is fixed, so "consume one
/// codepoint" is a function of (state, codepoint). Interning the successor states
/// collapses the walk over ~150k token pieces into a few thousand transitions --
/// the difference between ~70 ms and ~5 ms per state on a 150k-token vocabulary.
/// Correctness is unchanged: a token is allowed exactly when every codepoint of
/// its piece is consumable, and a trailing partial sequence can still complete.
struct MaskWalk<'a> {
    g: &'a Grammar,
    states: Vec<Vec<Vec<[u32; 2]>>>,
    intern: HashMap<Vec<Vec<[u32; 2]>>, usize>,
    memo: HashMap<(usize, u32), Option<usize>>,
    /// Pieces examined. Read by `GrammarState::pieces_walked` in tests only.
    walked: usize,
}

impl<'a> MaskWalk<'a> {
    fn new(g: &'a Grammar, seed: &[Vec<[u32; 2]>]) -> MaskWalk<'a> {
        MaskWalk {
            g,
            states: vec![seed.to_vec()],
            intern: HashMap::from([(seed.to_vec(), 0)]),
            memo: HashMap::new(),
            walked: 0,
        }
    }

    /// Walk a token piece through the memoized transition table. Returns true
    /// when every codepoint is consumable and a trailing partial sequence (if
    /// any) can still complete.
    fn allows(&mut self, piece: &[u8]) -> bool {
        self.walked += 1;
        let mut cur = 0usize;
        let mut i = 0usize;
        while i < piece.len() {
            match decode_at(piece, i) {
                Utf8Step::Cp(cp, n) => {
                    if (0xD800..=0xDFFF).contains(&cp) {
                        return false;
                    }
                    match self.memo_step(cur, cp) {
                        Some(next) => cur = next,
                        None => return false,
                    }
                    i += n;
                }
                Utf8Step::Incomplete => {
                    return self.g.partial_can_complete(&self.states[cur], &piece[i..]);
                }
                Utf8Step::Invalid => return false,
            }
        }
        true
    }

    /// Walk one first-codepoint bucket, setting the bit for every id whose whole
    /// piece is consumable.
    fn bucket(&mut self, ids: &[u32], bits: &mut [u64]) {
        let g = self.g;
        for &id in ids {
            if self.allows(g.piece(id)) {
                set_bit(bits, id as usize);
            }
        }
    }

    /// One memoized codepoint transition, interning the successor state.
    fn memo_step(&mut self, cur: usize, cp: u32) -> Option<usize> {
        if let Some(cached) = self.memo.get(&(cur, cp)) {
            return *cached;
        }
        let stepped = self.g.step(&self.states[cur], cp).ok();
        let next = stepped.map(|ns| match self.intern.get(&ns) {
            Some(&idx) => idx,
            None => {
                let idx = self.states.len();
                self.states.push(ns.clone());
                self.intern.insert(ns, idx);
                idx
            }
        });
        self.memo.insert((cur, cp), next);
        next
    }
}

/// Set token `id` in a packed bitset.
fn set_bit(bits: &mut [u64], id: usize) {
    bits[id / 64] |= 1u64 << (id % 64);
}

/// Sort ranges and merge every overlapping or adjacent pair, so the result is
/// ascending, disjoint and non-adjacent.
fn merge_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        match out.last_mut() {
            Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
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
                     \\\\ \\\" \\' \\[ \\] \\- \\^ \\/ \\xNN \\uNNNN; the class shorthands \
                     \\d/\\w/\\s are only valid inside `[...]`, and \\p{{...}} is not supported)",
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

    /// A class shorthand (`\d`, `\w`, `\s`) as ranges, if the next escape is one.
    /// The caller has already consumed the backslash.
    ///
    /// These are **sets**, not codepoints, so they are only legal inside a
    /// character class: `"\d"` has no single character it could mean. The
    /// definitions are ECMA-262's `\d`/`\w`/`\s` over ASCII — `\s` is the ASCII
    /// whitespace set (space and U+0009..U+000D), not the Unicode space
    /// separators, and `\w` includes `_`.
    fn class_shorthand(&mut self) -> Result<Option<Vec<(u32, u32)>>, String> {
        let ranges: Vec<(u32, u32)> = match self.peek() {
            Some('d') => vec![('0' as u32, '9' as u32)],
            Some('w') => vec![
                ('0' as u32, '9' as u32),
                ('A' as u32, 'Z' as u32),
                ('_' as u32, '_' as u32),
                ('a' as u32, 'z' as u32),
            ],
            Some('s') => vec![('\t' as u32, '\r' as u32), (' ' as u32, ' ' as u32)],
            _ => return Ok(None),
        };
        self.bump();
        // A set cannot be a range endpoint: `[\d-x]` would otherwise read as the
        // set plus a `-`..`x` range, accepting far more than was written.
        // `[\d-]` stays a literal minus, as everywhere else in a class.
        if self.peek() == Some('-')
            && !matches!(self.s[self.i + 1..].chars().next(), Some(']') | None)
        {
            return Err(format!(
                "grammar: a class shorthand cannot be a range endpoint at byte {} (near `{}`)",
                self.i,
                self.snippet()
            ));
        }
        Ok(Some(ranges))
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
                    if let Some(short) = self.class_shorthand()? {
                        ranges.extend(short);
                        continue;
                    }
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
///
/// `pub(crate)` for the tool-call grammar (`src/server/toolcall.rs`), which wraps
/// `j-object` in the model's `<tool_call>` syntax (#614) — one JSON grammar, not
/// a second copy that can drift from the schema compiler's.
pub(crate) const GENERIC_RULES: &[(&str, &str)] = &[
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
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// Keywords that narrow `type: "string"` and are implemented only there. On any
/// other type the constraint would be silently dropped, which this compiler
/// refuses rather than ignores.
const STRING_ONLY_KEYWORDS: &[&str] = &["format", "minLength", "maxLength", "pattern"];

/// The three `format` bodies the reference also implements. `date` and `time`
/// are `STRING_FORMAT_RULES` in llama.cpp's `json-schema-to-grammar.cpp`; the
/// `uuid` primitive there accepts `uuid` and `uuid1`..`uuid5` (version digit
/// unchecked), which is what [`format_rule`] does.
const GBNF_DATE: &str =
    r#"[0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" ("0" [1-9] | [1-2][0-9] | "3" [0-1])"#;
const GBNF_TIME: &str = r#"([01][0-9] | "2" [0-3]) ":" [0-5][0-9] ":" [0-5][0-9] ("." [0-9]{3})? ("Z" | ("+" | "-") ([01][0-9] | "2" [0-3]) ":" [0-5][0-9])"#;
const GBNF_UUID: &str = r#"[0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12}"#;

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

        // `allOf`: the intersection of the branches, compiled by merging them.
        // Only object schemas can be merged exactly; anything else would need a
        // real intersection, so it is refused by name.
        if let Some(branches) = obj.get("allOf") {
            let extra: Vec<&str> = obj
                .keys()
                .filter(|k| !is_annotation(k) && k.as_str() != "allOf" && k.as_str() != "type")
                .map(|k| k.as_str())
                .collect();
            if !extra.is_empty() {
                return Err(format!(
                    "json-schema: `allOf` combined with {extra:?} is not supported (the siblings \
                     would have to be intersected with the union of the branches)"
                ));
            }
            match obj.get("type") {
                None => {}
                Some(Value::String(t)) if t == "object" => {}
                Some(other) => {
                    return Err(format!(
                        "json-schema: `allOf` is compiled for object schemas only \
                         (`type: \"object\"`), got {}",
                        json_type_name(other)
                    ))
                }
            }
            let Value::Array(branches) = branches else {
                return Err("json-schema: `allOf` must be an array of schemas".to_string());
            };
            return self.compile_all_of(branches, hint);
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
        // The string keywords are implemented for `type: "string"` alone.
        let string_kws: Vec<&str> = STRING_ONLY_KEYWORDS
            .iter()
            .copied()
            .filter(|k| obj.contains_key(*k))
            .collect();
        if !string_kws.is_empty() && !(types.len() == 1 && types[0] == "string") {
            return Err(format!(
                "json-schema: {string_kws:?} is only supported on `type: \"string\"` — here the \
                 constraint would be silently dropped; give the schema that single type, or split \
                 a union into `anyOf` branches"
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
            "string" => self.compile_string(obj),
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

    /// `type: "string"`: the generic `j-string`, narrowed by `minLength` /
    /// `maxLength` and/or a `format`.
    ///
    /// The bounds count characters, not bytes: `j-char` matches one plain
    /// character **or one complete escape** (`j-esc` swallows the backslash and
    /// the whole `\uXXXX` body), so `"\u0041"` is one character, exactly as
    /// `"\A"` would be. A surrogate pair written as two escapes counts as two.
    fn compile_string(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String, String> {
        let format = obj.get("format");
        let pattern = obj.get("pattern");
        let min = get_count(obj, "minLength")?;
        let max = get_count(obj, "maxLength")?;
        if let Some(hi) = max {
            if min.unwrap_or(0) > hi {
                return Err(format!(
                    "json-schema: `minLength` ({}) exceeds `maxLength` ({hi}) — no value can \
                     satisfy it",
                    min.unwrap_or(0)
                ));
            }
        }
        // Any two of these three would have to be intersected, which a GBNF
        // alternation cannot express.
        let given: Vec<&str> = [
            ("pattern", pattern.is_some()),
            ("format", format.is_some()),
            ("minLength/maxLength", min.is_some() || max.is_some()),
        ]
        .iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| *name)
        .collect();
        if given.len() > 1 {
            return Err(format!(
                "json-schema: {} combined on one string is not supported (the constraints would \
                 have to be intersected); keep one of them",
                given.join(" + ")
            ));
        }
        if let Some(f) = format {
            let name = f.as_str().ok_or_else(|| {
                format!(
                    "json-schema: `format` must be a string, got {}",
                    json_type_name(f)
                )
            })?;
            return format_rule(name);
        }
        if let Some(p) = pattern {
            let text = p.as_str().ok_or_else(|| {
                format!(
                    "json-schema: `pattern` must be a string, got {}",
                    json_type_name(p)
                )
            })?;
            return pattern_rule(text);
        }
        match (min.unwrap_or(0), max) {
            (0, None) => Ok("j-string".to_string()),
            (m, None) => Ok(format!(r#""\"" j-char{{{m},}} "\"""#)),
            (m, Some(n)) => Ok(format!(r#""\"" j-char{{{m},{n}}} "\"""#)),
        }
    }

    /// Compile `allOf` by **merging** its branches: the union of the declared
    /// properties (restricted to the names every closed branch allows), the union
    /// of `required`, and the `additionalProperties` the branches imply.
    ///
    /// That is the exact intersection for object schemas — which is why a conflict
    /// is refused rather than resolved: the same property declared twice with
    /// different schemas, two different open-ended `additionalProperties`
    /// schemas, a required name a closed branch forbids, or a branch that is not
    /// an object schema all need a real intersection, and a guess would silently
    /// widen or narrow the language.
    fn compile_all_of(&mut self, branches: &[Value], hint: &str) -> Result<String, String> {
        if branches.is_empty() {
            return Err("json-schema: `allOf` with no branches admits nothing".to_string());
        }
        // One pass to validate the branches and collect what they say.
        let mut declared: Vec<(String, Value, usize)> = Vec::new();
        let mut required: Vec<String> = Vec::new();
        let mut open_additional: Option<Value> = None;
        let mut closed: Vec<Vec<String>> = Vec::new();
        for (i, branch) in branches.iter().enumerate() {
            let schema = self.resolve_branch(branch, 0)?;
            let obj = schema.as_object().ok_or_else(|| {
                format!("json-schema: `allOf` branch {i} must be an object schema")
            })?;
            // The type is checked first: a branch that is not an object schema is
            // a more fundamental refusal than one of its keywords.
            match obj.get("type") {
                None => {}
                Some(Value::String(t)) if t == "object" => {}
                Some(other) => {
                    return Err(format!(
                        "json-schema: `allOf` branch {i} is not an object schema (`type` is {})",
                        json_type_name(other)
                    ))
                }
            }
            for key in obj.keys() {
                if is_annotation(key)
                    || matches!(
                        key.as_str(),
                        "$ref" | "type" | "properties" | "required" | "additionalProperties"
                    )
                {
                    continue;
                }
                return Err(format!(
                    "json-schema: `allOf` branch {i} uses `{key}`, which cannot be intersected \
                     with the other branches (only `properties`, `required` and \
                     `additionalProperties` merge)"
                ));
            }
            let names: Vec<String> = match obj.get("properties") {
                None => Vec::new(),
                Some(Value::Object(props)) => {
                    for (name, sub) in props {
                        declared.push((name.clone(), sub.clone(), i));
                    }
                    props.keys().cloned().collect()
                }
                Some(_) => {
                    return Err(format!(
                        "json-schema: `allOf` branch {i} has a non-object `properties`"
                    ))
                }
            };
            match obj.get("required") {
                None => {}
                Some(Value::Array(req)) => {
                    for r in req {
                        let name = r.as_str().ok_or_else(|| {
                            format!(
                                "json-schema: `allOf` branch {i} has a non-string `required` entry"
                            )
                        })?;
                        if !required.iter().any(|x| x == name) {
                            required.push(name.to_string());
                        }
                    }
                }
                Some(_) => {
                    return Err(format!(
                        "json-schema: `allOf` branch {i} has a non-array `required`"
                    ))
                }
            }
            match obj.get("additionalProperties") {
                None => {}
                Some(Value::Bool(false)) => closed.push(names),
                Some(v) => {
                    match &open_additional {
                        None => open_additional = Some(v.clone()),
                        Some(prev) if prev == v => {}
                        Some(_) => return Err(
                            "json-schema: `allOf` branches disagree on `additionalProperties` — \
                             the intersection is not expressible"
                                .to_string(),
                        ),
                    }
                }
            }
        }
        // A closed branch allows exactly the names it declares, so the merged
        // object keeps the intersection of every closed branch's set.
        let allowed: Option<Vec<String>> = if closed.is_empty() {
            None
        } else {
            let mut set = closed[0].clone();
            for c in &closed[1..] {
                set.retain(|n| c.contains(n));
            }
            Some(set)
        };
        let permits = |name: &str| {
            allowed
                .as_ref()
                .is_none_or(|set| set.contains(&name.to_string()))
        };
        let mut properties: serde_json::Map<String, Value> = serde_json::Map::new();
        for (name, sub, branch) in &declared {
            if !permits(name) {
                continue;
            }
            match properties.get(name) {
                None => {
                    properties.insert(name.clone(), sub.clone());
                }
                Some(existing) if existing == sub => {}
                Some(_) => {
                    return Err(format!(
                        "json-schema: `allOf` declares `{name}` in more than one branch with \
                         different schemas (branch {branch} differs) — the intersection is not \
                         expressible"
                    ))
                }
            }
        }
        for name in &required {
            if !permits(name) {
                return Err(format!(
                    "json-schema: `allOf` admits nothing: `{name}` is required by one branch and \
                     forbidden by a closed one"
                ));
            }
        }
        let mut merged = serde_json::Map::new();
        merged.insert("type".to_string(), Value::String("object".to_string()));
        if !properties.is_empty() {
            merged.insert("properties".to_string(), Value::Object(properties));
        }
        if !required.is_empty() {
            merged.insert(
                "required".to_string(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }
        if let Some(closed) = closed.first() {
            // The merged object is closed too: the names above are the only ones
            // every closed branch allows.
            let _ = closed;
            merged.insert("additionalProperties".to_string(), Value::Bool(false));
        } else if let Some(v) = open_additional {
            merged.insert("additionalProperties".to_string(), v);
        }
        self.compile_object(&merged, hint)
    }

    /// A branch of `allOf`, with a local `$ref` resolved to the schema it names so
    /// the merge sees its properties. A cycle is refused rather than looped.
    fn resolve_branch(&self, branch: &Value, depth: usize) -> Result<Value, String> {
        if depth > 8 {
            return Err("json-schema: `allOf` branch `$ref` chain exceeds 8 levels".to_string());
        }
        let obj = branch
            .as_object()
            .ok_or_else(|| "json-schema: `allOf` branch must be an object schema".to_string())?;
        let Some(Value::String(r)) = obj.get("$ref") else {
            return Ok(branch.clone());
        };
        let extra: Vec<&str> = obj
            .keys()
            .filter(|k| !is_annotation(k) && k.as_str() != "$ref")
            .map(|k| k.as_str())
            .collect();
        if !extra.is_empty() {
            return Err(format!(
                "json-schema: `$ref` combined with {extra:?} is not supported (the siblings \
                 would be silently dropped)"
            ));
        }
        let name = r
            .strip_prefix("#/$defs/")
            .or_else(|| r.strip_prefix("#/definitions/"))
            .ok_or_else(|| {
                format!("json-schema: `$ref: \"{r}\"` is not supported inside `allOf`")
            })?;
        let schema = self.defs.get(name).cloned().ok_or_else(|| {
            format!("json-schema: `$ref: \"{r}\"` does not resolve to a definition")
        })?;
        self.resolve_branch(&schema, depth + 1)
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

/// The GBNF for the empty string, used for an empty regex branch or group.
const GBNF_EMPTY: &str = r#""""#;

/// One positive shorthand a pattern may name.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shorthand {
    Digit,
    Word,
    Space,
}

impl Shorthand {
    /// The ECMA-262 ASCII set.
    fn ranges(self) -> &'static [(u32, u32)] {
        match self {
            Shorthand::Digit => &[(0x30, 0x39)],
            Shorthand::Word => &[(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)],
            Shorthand::Space => &[(0x09, 0x0D), (0x20, 0x20)],
        }
    }
}

/// A compiler for the subset of ECMA-262 `pattern` syntax
/// `docs/GRAMMAR-DESIGN.md` §3 names. It emits GBNF text, which the GBNF parser
/// then compiles — the same two-stage shape the rest of the JSON compiler uses.
///
/// Supported: literals (ASCII and non-ASCII), `.`, positive classes `[...]` over
/// U+0000..=U+00FF with `\d`/`\w`/`\s`, `|`, `(...)`/`(?:...)`, the quantifiers
/// `*` `+` `?` `{m}` `{m,}` `{m,n}` (with the lazy `?` suffix, which does not
/// change the accepted language), and `^`/`$` at the very start/end. Everything
/// else is a named refusal — never a widened language.
struct RegexParser<'a> {
    s: &'a [char],
    i: usize,
}

impl<'a> RegexParser<'a> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn err(&self, what: impl std::fmt::Display) -> String {
        format!("json-schema: pattern: {what} (at offset {})", self.i)
    }

    /// `alternation := sequence ('|' sequence)*`
    fn alternation(&mut self, depth: usize) -> Result<String, String> {
        let mut branches = vec![self.sequence(depth)?];
        while self.eat('|') {
            branches.push(self.sequence(depth)?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(format!("({})", branches.join(" | ")))
        }
    }

    /// `sequence := term*`
    fn sequence(&mut self, depth: usize) -> Result<String, String> {
        let mut parts: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                None | Some('|') | Some(')') => break,
                Some('*') | Some('+') | Some('?') => {
                    return Err(self.err("a quantifier with nothing to repeat"))
                }
                _ => {}
            }
            let atom = self.atom(depth)?;
            parts.push(match self.quantifier()? {
                Some(q) => format!("({atom}){q}"),
                None => atom,
            });
        }
        if parts.is_empty() {
            Ok(GBNF_EMPTY.to_string())
        } else {
            Ok(parts.join(" "))
        }
    }

    /// `atom := group | class | '.' | escape | literal`
    ///
    /// `^` and `$` never reach here as anchors: the caller strips a leading `^`
    /// and a trailing `$`, so any that remain are misplaced.
    fn atom(&mut self, depth: usize) -> Result<String, String> {
        let c = self
            .peek()
            .ok_or_else(|| self.err("expected an expression"))?;
        match c {
            '(' => {
                self.bump();
                if self.eat('?') && !self.eat(':') {
                    return Err(self.err(
                        "only the non-capturing group `(?:...)` is supported (lookaround, named \
                         groups, flags and atomic groups are not)",
                    ));
                }
                let body = self.alternation(depth + 1)?;
                if !self.eat(')') {
                    return Err(self.err("unmatched `(`"));
                }
                Ok(format!("({body})"))
            }
            '[' => {
                self.bump();
                self.class()
            }
            '.' => {
                self.bump();
                char_set_rule(&complement_ranges(&[
                    (0x0A, 0x0A),
                    (0x0D, 0x0D),
                    (0x2028, 0x2028),
                    (0x2029, 0x2029),
                ]))
            }
            '^' | '$' => Err(self.err(
                "an anchor is only supported as a leading `^` or a trailing `$` (an anchor \
                 combined with a top-level `|` is not)",
            )),
            '*' | '+' | '?' => Err(self.err("a quantifier with nothing to repeat")),
            ')' => Err(self.err("unmatched `)`")),
            '\\' => {
                self.bump();
                self.escape()
            }
            other => {
                self.bump();
                char_expr(other)
            }
        }
    }

    /// `quantifier := ('*' | '+' | '?' | '{' m (',' n?)? '}') '?'?`
    ///
    /// A trailing `?` is the lazy form, which accepts exactly the same language
    /// and is therefore free to accept; a trailing `+` is possessive and does
    /// change the language, so it is refused.
    fn quantifier(&mut self) -> Result<Option<String>, String> {
        let q = match self.peek() {
            Some('*') | Some('+') | Some('?') => self.bump().unwrap().to_string(),
            Some('{') => {
                self.bump();
                let start = self.i;
                let mut m = String::new();
                while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    m.push(self.bump().unwrap());
                }
                if m.is_empty() {
                    return Err(self.err("a `{` quantifier needs at least one digit"));
                }
                if self.eat('}') {
                    format!("{{{m}}}")
                } else if self.eat(',') {
                    let mut n = String::new();
                    while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                        n.push(self.bump().unwrap());
                    }
                    if !self.eat('}') {
                        self.i = start;
                        return Err(self.err("unterminated `{...}` quantifier"));
                    }
                    if n.is_empty() {
                        format!("{{{m},}}")
                    } else {
                        format!("{{{m},{n}}}")
                    }
                } else {
                    self.i = start;
                    return Err(self.err("a `{` quantifier needs `}`, or `,n}`"));
                }
            }
            _ => return Ok(None),
        };
        if self.peek() == Some('?') {
            self.bump();
        } else if self.peek() == Some('+') {
            return Err(self.err("a possessive quantifier (`*+`, `++`, `?+`) is not supported"));
        }
        Ok(Some(q))
    }

    /// One escape, after the backslash was consumed.
    fn escape(&mut self) -> Result<String, String> {
        let c = self
            .bump()
            .ok_or_else(|| self.err("a trailing backslash"))?;
        let shorthand = match c {
            'd' => Some(Shorthand::Digit),
            'w' => Some(Shorthand::Word),
            's' => Some(Shorthand::Space),
            _ => None,
        };
        if let Some(sh) = shorthand {
            return char_set_rule(sh.ranges());
        }
        match c {
            'D' | 'W' | 'S' => {
                let sh = match c {
                    'D' => Shorthand::Digit,
                    'W' => Shorthand::Word,
                    _ => Shorthand::Space,
                };
                return char_set_rule(&complement_ranges(sh.ranges()));
            }
            'b' | 'B' => {
                return Err(self.err("a word-boundary assertion (`\\b`, `\\B`) is not supported"))
            }
            'p' | 'P' => {
                return Err(
                    self.err("a Unicode property escape (`\\p{...}`, `\\P{...}`) is not supported")
                )
            }
            'k' => return Err(self.err("a named backreference (`\\k<...>`) is not supported")),
            'c' => return Err(self.err("a control escape (`\\cX`) is not supported")),
            '1'..='9' => return Err(self.err("a backreference (`\\1`..`\\9`) is not supported")),
            '0' => {
                if matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    return Err(self.err("an octal escape is not supported"));
                }
                return char_expr('\0');
            }
            'n' => return char_expr('\n'),
            'r' => return char_expr('\r'),
            't' => return char_expr('\t'),
            'f' => return char_expr('\u{0C}'),
            'v' => return char_expr('\u{0B}'),
            'x' => return Ok(char_expr(self.hex_char(2)?)?),
            'u' => {
                if self.peek() == Some('{') {
                    return Err(
                        self.err("the `\\u{...}` escape is not supported (write `\\uXXXX`)")
                    );
                }
                return Ok(char_expr(self.hex_char(4)?)?);
            }
            // An identity escape: any non-alphanumeric character stands for itself.
            other if !other.is_ascii_alphanumeric() => return Ok(char_expr(other)?),
            other => return Err(self.err(format!("unsupported escape `\\{other}`"))),
        }
    }

    /// `n` hex digits as a character. A lone surrogate is not encodable, so it is
    /// refused rather than silently dropped.
    fn hex_char(&mut self, n: usize) -> Result<char, String> {
        let mut value: u32 = 0;
        for _ in 0..n {
            let d = self
                .peek()
                .filter(char::is_ascii_hexdigit)
                .ok_or_else(|| self.err(format!("`\\x`/`\\u` needs {n} hex digits")))?;
            self.bump();
            value = value * 16 + d.to_digit(16).unwrap();
        }
        char::from_u32(value).ok_or_else(|| {
            self.err(format!(
                "U+{value:04X} is a lone surrogate, which no JSON string can encode"
            ))
        })
    }

    /// A positive `[...]` class, with `[` already consumed.
    fn class(&mut self) -> Result<String, String> {
        let negated = self.eat('^');
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| self.err("unterminated character class"))?;
            if c == ']' {
                self.bump();
                break;
            }
            if let Some(lo) = self.class_member(&mut ranges)? {
                self.maybe_range(lo, &mut ranges)?;
            }
        }
        if ranges.is_empty() && !negated {
            return Err(self.err("an empty character class `[]`"));
        }
        let written = merge_ranges(ranges);
        char_set_rule(&if negated {
            complement_ranges(&written)
        } else {
            written
        })
    }

    /// One class member: a shorthand set (expanded into `ranges`), or a codepoint
    /// that may start a range. A shorthand is a whole set, so it cannot be a
    /// range endpoint — `[\d-x]` would otherwise read as the set plus a `-`..`x`
    /// range.
    fn class_member(&mut self, ranges: &mut Vec<(u32, u32)>) -> Result<Option<u32>, String> {
        if self.peek() != Some('\\') {
            return Ok(Some(self.class_char()?));
        }
        let save = self.i;
        self.bump();
        let (shorthand, negated) = match self.peek() {
            Some('d') => (Some(Shorthand::Digit), false),
            Some('w') => (Some(Shorthand::Word), false),
            Some('s') => (Some(Shorthand::Space), false),
            Some('D') => (Some(Shorthand::Digit), true),
            Some('W') => (Some(Shorthand::Word), true),
            Some('S') => (Some(Shorthand::Space), true),
            _ => (None, false),
        };
        let Some(sh) = shorthand else {
            self.i = save;
            return Ok(Some(self.class_char()?));
        };
        self.bump();
        if self.peek() == Some('-') && !matches!(self.s.get(self.i + 1).copied(), Some(']') | None)
        {
            return Err(self.err("a class shorthand cannot be a range endpoint"));
        }
        if negated {
            ranges.extend(complement_ranges(sh.ranges()));
        } else {
            ranges.extend_from_slice(sh.ranges());
        }
        Ok(None)
    }

    /// One class member character (an escape or a literal), as a codepoint.
    fn class_char(&mut self) -> Result<u32, String> {
        match self.bump() {
            None => Err(self.err("unterminated character class")),
            Some('\\') => {
                let c = self
                    .bump()
                    .ok_or_else(|| self.err("a trailing backslash"))?;
                match c {
                    'n' => Ok('\n' as u32),
                    'r' => Ok('\r' as u32),
                    't' => Ok('\t' as u32),
                    'f' => Ok(0x0C),
                    'v' => Ok(0x0B),
                    'x' => Ok(self.hex_char(2)? as u32),
                    'u' => Ok(self.hex_char(4)? as u32),
                    'b' => Err(self.err("`\\b` inside a class is not supported")),
                    'd' | 'w' | 's' => Err(self.err("a class shorthand is not a single character")),
                    'D' | 'W' | 'S' => Err(self.err("a class shorthand is not a single character")),
                    '1'..='9' => Err(self.err("a backreference is not supported")),
                    other if !other.is_ascii_alphanumeric() => Ok(other as u32),
                    other => Err(self.err(format!("unsupported escape `\\{other}` in a class"))),
                }
            }
            Some(c) => Ok(c as u32),
        }
    }

    /// Extend `lo` into `hi` when a `-` follows, else push it as a single member.
    fn maybe_range(&mut self, lo: u32, ranges: &mut Vec<(u32, u32)>) -> Result<(), String> {
        if self.peek() == Some('-') && !matches!(self.s.get(self.i + 1).copied(), Some(']') | None)
        {
            self.bump();
            let hi = self.class_char()?;
            if hi < lo {
                return Err(self.err(format!(
                    "the character class range U+{lo:04X}-U+{hi:04X} is reversed"
                )));
            }
            ranges.push((lo, hi));
        } else {
            ranges.push((lo, lo));
        }
        Ok(())
    }
}

/// One decoded character as a GBNF expression, escape-aware: the literal spelling
/// plus every escape spelling that decodes to it. Up to U+00FF that is
/// [`class_rule`]; above it the
/// JSON text has exactly two spellings (the character itself, and `\uXXXX` or a
/// surrogate pair).
fn char_expr(c: char) -> Result<String, String> {
    let cp = c as u32;
    if cp <= 0xFF {
        return char_set_rule(&[(cp, cp)]);
    }
    Ok(format!(
        "({} | {})",
        gbnf_string(&c.to_string()),
        gbnf_u_escape(cp)
    ))
}

/// The `\uXXXX` spelling of a codepoint as GBNF: each digit is a class so either
/// hex case matches, and a non-BMP codepoint becomes its surrogate pair.
fn gbnf_u_escape(cp: u32) -> String {
    let digits = |value: u32| -> String {
        let mut out = String::from(r#""\\u""#);
        for shift in [12u32, 8, 4, 0] {
            let nibble = (value >> shift) & 0xF;
            let c = char::from_digit(nibble, 16).expect("a nibble is a hex digit");
            if nibble >= 10 {
                out.push_str(&format!(" [{c}{}]", c.to_ascii_uppercase()));
            } else {
                out.push_str(&format!(" [{c}]"));
            }
        }
        out
    };
    if cp <= 0xFFFF {
        return digits(cp);
    }
    let v = cp - 0x10000;
    format!(
        "{} {}",
        digits(0xD800 + (v >> 10)),
        digits(0xDC00 + (v & 0x3FF))
    )
}

/// A GBNF string literal for `text`.
fn gbnf_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// One codepoint as a GBNF class member, for a class over the *text*.
fn gbnf_class_char(cp: u32) -> Result<String, String> {
    let c = char::from_u32(cp).ok_or_else(|| {
        format!("json-schema: pattern: U+{cp:04X} is a lone surrogate and cannot appear in a class")
    })?;
    Ok(match c {
        '\\' | ']' | '-' | '^' | '[' => format!("\\{c}"),
        c if (c as u32) < 0x20 || c as u32 == 0x7F => format!("\\u{:04X}", c as u32),
        c => c.to_string(),
    })
}

/// The codepoints JSON allows to appear in a string **literally**: everything
/// except the control characters, `"` and `\` — and the surrogate block, which
/// has no UTF-8 encoding.
fn plain_char_set(set: &[(u32, u32)]) -> Vec<(u32, u32)> {
    set_subtract(
        set,
        &[(0x00, 0x1F), (0x22, 0x22), (0x5C, 0x5C), (0xD800, 0xDFFF)],
    )
}

/// A GBNF alternation matching **one decoded character** from `set`, escape-aware.
///
/// The JSON text may spell a character literally or as an escape, so both are
/// accepted: a class over the literal members, the short escapes, a four-hex-digit
/// automaton for the BMP part, and the surrogate-pair form for everything above
/// U+FFFF. Every part of the set is *enumerated* rather than approximated — a
/// language that accepts what the schema rejects is what this compiler refuses to
/// produce — which is why a negated class is as expressible as a positive one and
/// no class has to be refused for its size.
fn char_set_rule(set: &[(u32, u32)]) -> Result<String, String> {
    let set = merge_ranges(set.to_vec());
    if set.is_empty() {
        return Err("json-schema: pattern: a character set that admits nothing".to_string());
    }
    let mut alts: Vec<String> = Vec::new();
    let plain = plain_char_set(&set);
    if !plain.is_empty() {
        alts.push(format!("[{}]", class_members(&plain)?));
    }
    let mut escapes: Vec<String> = Vec::new();
    for (value, spelling) in [
        (0x22u32, r#""\\\"""#),
        (0x5C, r#""\\\\""#),
        (0x2F, r#""\\/""#),
        (0x08, r#""\\b""#),
        (0x0C, r#""\\f""#),
        (0x0A, r#""\\n""#),
        (0x0D, r#""\\r""#),
        (0x09, r#""\\t""#),
    ] {
        if set_contains(&set, value) {
            escapes.push(spelling.to_string());
        }
    }
    let bmp = set_subtract(&intersect(&set, &[(0x0000, 0xFFFF)]), &[(0xD800, 0xDFFF)]);
    if !bmp.is_empty() {
        escapes.push(format!(r#""\\u" {}"#, hex_expr(&bmp, 4)));
    }
    let non_bmp = intersect(&set, &[(0x10000, 0x10FFFF)]);
    if !non_bmp.is_empty() {
        escapes.push(surrogate_pair_expr(&non_bmp));
    }
    if !escapes.is_empty() {
        alts.push(format!("({})", escapes.join(" | ")));
    }
    Ok(format!("({})", alts.join(" | ")))
}

/// A GBNF class body for codepoints that may appear literally in the text.
fn class_members(ranges: &[(u32, u32)]) -> Result<String, String> {
    let mut body = String::new();
    for &(a, b) in ranges {
        body.push_str(&gbnf_class_char(a)?);
        if b != a {
            body.push('-');
            body.push_str(&gbnf_class_char(b)?);
        }
    }
    Ok(body)
}

/// A GBNF expression matching exactly `nibbles` hex digits whose value is in
/// `ranges` (everything below `16^nibbles`). A subtree that is fully covered
/// collapses to a repetition, and nibbles that share a subtree share a class, so
/// even a wide set stays a short expression.
fn hex_expr(ranges: &[(u32, u32)], nibbles: u32) -> String {
    if nibbles == 1 {
        let values: Vec<u32> = (0..16).filter(|v| set_contains(ranges, *v)).collect();
        return hex_class(&values);
    }
    let span = 1u32 << (4 * (nibbles - 1));
    let mut groups: Vec<(String, Vec<u32>)> = Vec::new();
    for hi in 0..16u32 {
        let base = hi * span;
        let sub: Vec<(u32, u32)> = ranges
            .iter()
            .filter_map(|&(a, b)| {
                let lo = a.max(base);
                let hi_max = b.min(base + span - 1);
                (lo <= hi_max).then(|| (lo - base, hi_max - base))
            })
            .collect();
        if sub.is_empty() {
            continue;
        }
        let expr = if covers_exactly(&sub, span - 1) {
            format!("[0-9a-fA-F]{{{}}}", nibbles - 1)
        } else {
            hex_expr(&sub, nibbles - 1)
        };
        match groups.iter_mut().find(|(e, _)| *e == expr) {
            Some((_, his)) => his.push(hi),
            None => groups.push((expr, vec![hi])),
        }
    }
    let alts: Vec<String> = groups
        .into_iter()
        .map(|(expr, his)| format!("{} {expr}", hex_class(&his)))
        .collect();
    format!("({})", alts.join(" | "))
}

/// A GBNF class of hex-digit characters for the nibble values in `values`.
fn hex_class(values: &[u32]) -> String {
    if values.len() == 16 {
        return "[0-9a-fA-F]".to_string();
    }
    let mut body = String::new();
    for &v in values {
        let c = char::from_digit(v, 16).expect("a nibble is a hex digit");
        body.push(c);
        if v >= 10 {
            body.push(c.to_ascii_uppercase());
        }
    }
    format!("[{body}]")
}

/// Is `value` covered by the ascending, disjoint `ranges`?
fn set_contains(ranges: &[(u32, u32)], value: u32) -> bool {
    let i = ranges.partition_point(|&(a, _)| a <= value);
    i > 0 && ranges[i - 1].1 >= value
}

/// `ranges` minus `holes`, both ascending and disjoint.
fn set_subtract(ranges: &[(u32, u32)], holes: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &(a, b) in ranges {
        let mut cursor = a;
        for &(hlo, hhi) in holes {
            if hhi < cursor {
                continue;
            }
            if hlo > b {
                break;
            }
            if hlo > cursor {
                out.push((cursor, hlo - 1));
            }
            if hhi >= b {
                cursor = b + 1;
                break;
            }
            cursor = hhi + 1;
        }
        if cursor <= b {
            out.push((cursor, b));
        }
    }
    out
}

/// The surrogate-pair spelling (`\uD8xx\uDCxx`) of every codepoint in `ranges`
/// (all above U+FFFF), as a GBNF expression.
fn surrogate_pair_expr(ranges: &[(u32, u32)]) -> String {
    let mut groups: Vec<(String, Vec<u32>)> = Vec::new();
    for hi_index in 0..0x400u32 {
        let base = 0x10000 + hi_index * 0x400;
        let sub: Vec<(u32, u32)> = ranges
            .iter()
            .filter_map(|&(a, b)| {
                let lo = a.max(base);
                let hi_max = b.min(base + 0x3FF);
                (lo <= hi_max).then(|| (lo - base, hi_max - base))
            })
            .collect();
        if sub.is_empty() {
            continue;
        }
        let lo_values: Vec<(u32, u32)> =
            sub.iter().map(|&(a, b)| (a + 0xDC00, b + 0xDC00)).collect();
        // The low half is matched as its **escape spelling**, never as a literal
        // surrogate (no UTF-8 text can hold one).
        groups.push((hex_expr(&lo_values, 4), vec![hi_index + 0xD800]));
    }
    let mut merged: Vec<(String, Vec<u32>)> = Vec::new();
    for (expr, his) in groups {
        match merged.iter_mut().find(|(e, _)| *e == expr) {
            Some((_, all)) => all.extend(his),
            None => merged.push((expr, his)),
        }
    }
    let alts: Vec<String> = merged
        .into_iter()
        .map(|(lo, hi)| {
            let hi_ranges: Vec<(u32, u32)> = hi.iter().map(|&v| (v, v)).collect();
            format!(r#""\\u" {} "\\u" {lo}"#, hex_expr(&hi_ranges, 4))
        })
        .collect();
    format!("({})", alts.join(" | "))
}

/// Are the ascending ranges exactly one range covering `[0, hi]`?
fn covers_exactly(ranges: &[(u32, u32)], hi: u32) -> bool {
    ranges == [(0, hi)]
}

/// The intersection of two ascending, disjoint range lists.
fn intersect(a: &[(u32, u32)], b: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for &(alo, ahi) in a {
        for &(blo, bhi) in b {
            let lo = alo.max(blo);
            let hi = ahi.min(bhi);
            if lo <= hi {
                out.push((lo, hi));
            }
        }
    }
    merge_ranges(out)
}

/// The complement of `ranges` over the whole codepoint space, less the surrogate
/// block (which no JSON string can encode).
fn complement_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    const MAX: u32 = 0x10FFFF;
    let mut out = Vec::new();
    let mut next: u32 = 0;
    for &(a, b) in ranges {
        if a > next {
            out.extend(non_surrogate_parts(next, a - 1));
        }
        if b >= MAX {
            return out;
        }
        next = next.max(b + 1);
    }
    out.extend(non_surrogate_parts(next, MAX));
    out
}

/// Compile a `pattern` into the GBNF for a JSON string whose content matches it.
///
/// Following ECMA-262, the pattern matches a **substring** unless it is anchored:
/// `^` at the very start and `$` at the very end turn it into a whole-string
/// match. Both anchors are optional and independent. The reference instead
/// requires both and widens to "any string" with a warning when they are absent —
/// a silent superset this compiler never produces (§8).
fn pattern_rule(pattern: &str) -> Result<String, String> {
    let chars: Vec<char> = pattern.chars().collect();
    // A top-level `|` makes a leading/trailing anchor apply to one branch only,
    // which this emitter cannot express; refuse rather than widen.
    let top_level_alternation = has_top_level_alternation(&chars);
    let mut start = 0usize;
    let mut end = chars.len();
    let mut anchored_start = false;
    let mut anchored_end = false;
    if !top_level_alternation {
        if chars.first() == Some(&'^') {
            anchored_start = true;
            start = 1;
        }
        if end > start && chars[end - 1] == '$' && !is_escaped(&chars, end - 1) {
            anchored_end = true;
            end -= 1;
        }
    }
    let mut parser = RegexParser {
        s: &chars[start..end],
        i: 0,
    };
    let body = parser.alternation(0)?;
    if parser.i != parser.s.len() {
        return Err(parser.err(if parser.peek() == Some(')') {
            "unmatched `)`"
        } else {
            "unexpected trailing input"
        }));
    }
    let expr = match (anchored_start, anchored_end) {
        (true, true) => body,
        (true, false) => format!("({body}) j-char*"),
        (false, true) => format!("j-char* ({body})"),
        (false, false) => {
            if body == GBNF_EMPTY {
                // An empty pattern matches every string.
                return Ok("j-string".to_string());
            }
            format!("j-char* ({body}) j-char*")
        }
    };
    Ok(format!(r#""\"" {expr} "\"""#))
}

/// Is there a `|` outside every group, class and escape?
fn has_top_level_alternation(chars: &[char]) -> bool {
    let mut depth = 0usize;
    let mut i = 0usize;
    let mut in_class = false;
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 1,
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => depth = depth.saturating_sub(1),
            '|' if !in_class && depth == 0 => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

/// Is the character at `at` escaped by an odd number of backslashes?
fn is_escaped(chars: &[char], at: usize) -> bool {
    let mut backslashes = 0usize;
    let mut i = at;
    while i > 0 && chars[i - 1] == '\\' {
        backslashes += 1;
        i -= 1;
    }
    backslashes % 2 == 1
}

/// The quoted GBNF for one supported `format`, or a refusal that names it.
///
/// Only the validators the reference actually implements are supported. The
/// others — `email`, `uri`, `hostname`, `ipv4`, `ipv6` — are **refused by
/// name**: a loose rule for them would accept instances the schema rejects, and
/// llama.cpp only uses those spellings to infer "this is a string".
fn format_rule(name: &str) -> Result<String, String> {
    let body = match name {
        "date" => GBNF_DATE.to_string(),
        "time" => GBNF_TIME.to_string(),
        "date-time" => format!(r#"{GBNF_DATE} "T" {GBNF_TIME}"#),
        "uuid" | "uuid1" | "uuid2" | "uuid3" | "uuid4" | "uuid5" => GBNF_UUID.to_string(),
        other => {
            return Err(format!(
                "json-schema: `format: \"{other}\"` is not supported (supported: date, time, \
                 date-time, uuid, uuid1..uuid5 — the four validators the reference implements; \
                 a loose rule for the others would accept what the schema rejects). See \
                 docs/GRAMMAR-DESIGN.md §3"
            ))
        }
    };
    Ok(format!(r#""\"" {body} "\"""#))
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
mod tests;
