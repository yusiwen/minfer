//! The `quantize` target: one concrete encoder type, or one of llama.cpp's two
//! K-quant mixtures.
//!
//! `llama-quantize`'s bare `q4_K`/`q5_K` are CLI *aliases* for
//! `LLAMA_FTYPE_MOSTLY_Q4_K_M`/`_Q5_K_M`, and the file they write is selected
//! tensor by tensor by `llama_tensor_get_type_impl` (llama.cpp
//! `src/llama-quant.cpp`): the `OUTPUT`/tied-embedding branch, the
//! `use_more_bits` bumps for `attn_v` and `ffn_down`, and the fused-`qkv` rule.
//! Until #203 `minfer quantize --type q4_K` wrote one uniform type — what
//! `llama-quantize --pure` writes — and said so in its help text.
//!
//! A mixture is a *selection policy*, not an encoder, so it lives here instead of
//! as another spelling in [`QuantTarget`]: every concrete target keeps exactly
//! one encoder and [`crate::quantize::quantize_row`] stays total, while
//! [`crate::convert::QuantizePlan`] resolves each tensor to a concrete target
//! before any weight is encoded.
//!
//! Two properties this module must not get wrong, both pinned in `tests.rs`:
//!
//! * **Selection runs before the row-length demotion.** llama.cpp picks the type
//!   first and applies `tensor_type_fallback` second ([ADR-0041]). On the 0.5B
//!   (hidden 896) that is why the 12 `attn_v` that `use_more_bits` bumped are
//!   asked for `Q6_K` and land on `q8_0`.
//! * **The `OUTPUT`/tied branch is a row-length question before it is a type
//!   one**, and it runs for the legacy sub-8-bit targets too — that half of #203
//!   is not a mixture. It answers `Q8_0` when the requested type's block cannot
//!   divide the row and `Q6_K` when it can, which is exactly why the 0.5B's tied
//!   embedding is `q8_0` with **no** demotion (896 % 256 != 0) while the 0.6B's
//!   (hidden 1024) is `q6_K`. The rule this replaced asked for `q8_0`
//!   unconditionally, agreeing with llama.cpp on the 0.5B and disagreeing on the
//!   0.6B.
//!
//! [ADR-0041]: ../../docs/adr/0041-a-row-that-cannot-take-the-targets-block-is-demoted.md

use super::QuantTarget;

/// One of llama.cpp's `_M` mixtures: a per-tensor type *policy*.
///
/// The variants are spelled the way the CLI names them (`q4_K_M`), because that
/// is the vocabulary every doc, gate and reference file uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // q4_K_M/q5_K_M are llama.cpp's on-disk spellings
pub enum Mixture {
    Q4_K_M,
    Q5_K_M,
}

impl Mixture {
    /// The type every tensor falls back to when no rule bumps it.
    pub fn base(self) -> QuantTarget {
        match self {
            Mixture::Q4_K_M => QuantTarget::Q4_K,
            Mixture::Q5_K_M => QuantTarget::Q5_K,
        }
    }
}

/// What `--type` names: a concrete encoder target, or a mixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Concrete(QuantTarget),
    Mixture(Mixture),
}

impl From<QuantTarget> for Target {
    fn from(t: QuantTarget) -> Self {
        Target::Concrete(t)
    }
}

/// Why a tensor was moved off the target's own type.
///
/// Reported per tensor by the plan so the CLI can say *which* llama.cpp rule did
/// it, instead of one opaque "retargeted" list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The target's own type — no rule applied.
    Requested,
    /// llama.cpp's `OUTPUT`/tied-embedding branch.
    OutputBranch,
    /// A mixture bump (`use_more_bits`, or the fused-`qkv` rule).
    Mixture,
}

/// The type a tensor is *asked* for, before the row-length demotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub target: QuantTarget,
    pub reason: Reason,
}

/// What llama.cpp's selection reads about one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorFacts {
    pub category: TensorCategory,
    pub layer: Option<usize>,
    /// `ne[0]`: the row length a block must divide.
    pub row: i64,
}

/// The model-level facts llama.cpp's selection needs, computed once per plan.
///
/// `n_layers` is llama.cpp's `n_layer`/`n_ffn_down` and `n_attention_wv` its
/// counter of the same name; on a Qwen2/Qwen3 GGUF (one `attn_v` and one
/// `ffn_down` per block) both equal the block count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixCtx {
    pub n_layers: usize,
    pub n_attention_wv: usize,
    pub has_tied_embeddings: bool,
}

/// The tensor categories `llama_tensor_get_type_impl` switches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    TokenEmbd,
    Output,
    AttnQ,
    AttnK,
    AttnV,
    AttnQkv,
    AttnKvB,
    AttnOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Other,
}

/// llama.cpp's `tensor_get_category`, by GGUF tensor name.
///
/// The order is llama.cpp's, and it matters: the matchers are `find`-style
/// substrings, so `attn_qkv`/`attn_kv_b` are tested before `attn_v`, and
/// `attn_v` before `attn_k`/`attn_q` (a fused name contains no `attn_v` today,
/// but the order is the contract, not the coincidence).
pub fn category(name: &str) -> TensorCategory {
    if name == "output.weight" {
        return TensorCategory::Output;
    }
    if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
        return TensorCategory::TokenEmbd;
    }
    if name.contains("attn_qkv.weight") {
        return TensorCategory::AttnQkv;
    }
    if name.contains("attn_kv_b.weight") {
        return TensorCategory::AttnKvB;
    }
    if name.contains("attn_v.weight") {
        return TensorCategory::AttnV;
    }
    if name.contains("attn_k.weight") {
        return TensorCategory::AttnK;
    }
    if name.contains("attn_q.weight") {
        return TensorCategory::AttnQ;
    }
    if name.contains("attn_output.weight") {
        return TensorCategory::AttnOutput;
    }
    if name.contains("ffn_up") {
        return TensorCategory::FfnUp;
    }
    if name.contains("ffn_gate") {
        return TensorCategory::FfnGate;
    }
    if name.contains("ffn_down") {
        return TensorCategory::FfnDown;
    }
    TensorCategory::Other
}

/// llama.cpp's `category_is_attn_v`: the categories counted by `i_attention_wv`.
pub fn category_is_attn_v(cat: TensorCategory) -> bool {
    matches!(
        cat,
        TensorCategory::AttnV | TensorCategory::AttnQkv | TensorCategory::AttnKvB
    )
}

/// The block index in a `blk.N.…` GGUF name.
///
/// llama.cpp uses its visit counter for `ffn_down` (and parses the name only for
/// an MoE model); for the dense architectures this engine supports the ordinal
/// and the block index are the same number, and the index is the one a reader can
/// check against the file.
pub fn layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

/// llama.cpp's `use_more_bits`: the layers whose `attn_v`/`ffn_down` are bumped
/// to `Q6_K` under a mixture — the first and last eighths, plus every third layer
/// inside the middle.
pub fn use_more_bits(i: usize, n: usize) -> bool {
    let n8 = n / 8;
    if i < n8 {
        return true;
    }
    if i >= 7 * n / 8 {
        return true;
    }
    // Only reached when `i >= n8`, so the subtraction cannot wrap: llama.cpp
    // writes `(i - n/8) % 3 == 2` on signed ints and the short-circuit above is
    // what keeps it non-negative.
    (i - n8) % 3 == 2
}

/// The `OUTPUT` branch's trigger: an `output.weight`, or the *tied* token
/// embedding (llama.cpp treats a model with no `output.weight` as tied).
fn is_output(cat: TensorCategory, ctx: &MixCtx) -> bool {
    cat == TensorCategory::Output || (ctx.has_tied_embeddings && cat == TensorCategory::TokenEmbd)
}

impl Target {
    /// Parse a CLI target, refusing loudly and by name.
    ///
    /// The mixtures are matched before the concrete targets so their spellings
    /// are never mistaken for a uniform one; everything else is
    /// [`QuantTarget::parse`], whose refusals are the ones the CLI has always
    /// printed.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "q4_k_m" => Ok(Target::Mixture(Mixture::Q4_K_M)),
            "q5_k_m" => Ok(Target::Mixture(Mixture::Q5_K_M)),
            _ => QuantTarget::parse(s).map(Target::Concrete),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Target::Concrete(t) => t.name(),
            Target::Mixture(Mixture::Q4_K_M) => "q4_K_M",
            Target::Mixture(Mixture::Q5_K_M) => "q5_K_M",
        }
    }

    /// llama.cpp's `general.file_type` for this target.
    ///
    /// A mixture records the ftype it *is* (15/17, the same numbers as the
    /// uniform `q4_K`/`q5_K`, exactly as `llama-quantize --pure` does), so the
    /// metadata cannot tell a mixture from a uniform file and no code may infer
    /// "pure" from it.
    pub fn file_type(self) -> u32 {
        match self {
            Target::Concrete(t) => t.file_type(),
            Target::Mixture(Mixture::Q4_K_M) => 15, // LLAMA_FTYPE_MOSTLY_Q4_K_M
            Target::Mixture(Mixture::Q5_K_M) => 17, // LLAMA_FTYPE_MOSTLY_Q5_K_M
        }
    }

    /// The concrete type the policy is stated in terms of.
    pub fn base(self) -> QuantTarget {
        match self {
            Target::Concrete(t) => t,
            Target::Mixture(m) => m.base(),
        }
    }

    pub fn is_mixture(self) -> bool {
        matches!(self, Target::Mixture(_))
    }

    /// True for the `f32` cast, the one target that converts 1-D tensors too.
    pub fn is_f32(self) -> bool {
        self == Target::Concrete(QuantTarget::F32)
    }

    /// llama.cpp's `OUTPUT`/tied-embedding branch, which is a *row-length*
    /// question before it is a type one.
    ///
    /// `llama_tensor_get_type_impl` asks whether the requested type's block
    /// divides the row: if it does not, the answer is `Q8_0` (which the caller's
    /// row-length rule then re-examines, and which is where the 0.5B's tied
    /// embedding lands — 896 % 256 != 0 — with no demotion at all); if it does,
    /// the answer is `Q6_K`. A `q8_0` request is already its own answer, which
    /// is why that target is not run through this branch.
    fn output_branch(self, row: i64) -> QuantTarget {
        if row % self.base().blck_size() as i64 != 0 {
            QuantTarget::Q8_0
        } else {
            QuantTarget::Q6_K
        }
    }

    /// The type this tensor is *asked* for, before the row-length demotion.
    ///
    /// The resolution order is llama.cpp's: the `OUTPUT`/tied branch first (it
    /// overrides everything, for every quantizing ftype), then the mixture's
    /// per-category bumps.
    pub fn select(self, facts: &TensorFacts, ctx: &MixCtx) -> Selection {
        let cat = facts.category;
        let layer = facts.layer;
        match self {
            Target::Concrete(t) => {
                // The pure K-quant targets are the `--pure` reference and stay
                // uniform; `q6_K`'s own branch answer is `Q6_K` anyway, and
                // f16/f32 are element casts llama.cpp never feeds through
                // `llama_tensor_get_type`. The five legacy quantizers do take the
                // branch, and `q8_0`'s answer under it is `q8_0`.
                if matches!(
                    t,
                    QuantTarget::Q4_0 | QuantTarget::Q4_1 | QuantTarget::Q5_0 | QuantTarget::Q5_1
                ) && is_output(cat, ctx)
                {
                    return Selection {
                        target: self.output_branch(facts.row),
                        reason: Reason::OutputBranch,
                    };
                }
                Selection {
                    target: t,
                    reason: Reason::Requested,
                }
            }
            Target::Mixture(m) => {
                if is_output(cat, ctx) {
                    return Selection {
                        target: self.output_branch(facts.row),
                        reason: Reason::OutputBranch,
                    };
                }
                let base = m.base();
                let bumped = match (m, cat) {
                    // The fused-QKV rule: Q4_K_M -> Q5_K, Q5_K_M -> Q6_K.
                    (Mixture::Q4_K_M, TensorCategory::AttnQkv) => Some(QuantTarget::Q5_K),
                    (Mixture::Q5_K_M, TensorCategory::AttnQkv) => Some(QuantTarget::Q6_K),
                    // `use_more_bits` on the two weight categories llama.cpp
                    // gives extra bits to.
                    (_, TensorCategory::AttnV) => layer
                        .filter(|i| use_more_bits(*i, ctx.n_attention_wv))
                        .map(|_| QuantTarget::Q6_K),
                    (_, TensorCategory::FfnDown) => layer
                        .filter(|i| use_more_bits(*i, ctx.n_layers))
                        .map(|_| QuantTarget::Q6_K),
                    _ => None,
                };
                match bumped {
                    Some(t) => Selection {
                        target: t,
                        reason: Reason::Mixture,
                    },
                    None => Selection {
                        target: base,
                        reason: Reason::Requested,
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
