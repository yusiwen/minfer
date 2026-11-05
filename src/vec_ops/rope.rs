//! RoPE style vocabulary (the family the graph's RoPE op names) and the
//! per-dimension frequency table a `llama3`-scaled model carries (#724).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeStyle {
    NonInterleaved = 0, // Qwen2: pairs [i, i+half_dim]
    /// LLaMA/Mistral: pairs [2*i, 2*i+1] — kept for llama.cpp parity.
    ///
    /// Constructed by `graph::op_matrix` (its `RoPE` cases) only, so the
    /// allowance names `not(test)`; what would construct it in production is an
    /// architecture whose `rope_style` is interleaved (a LLaMA-family loader).
    /// [#244]'s verdict: keep the vocabulary, report the membership question.
    ///
    /// [#244]: https://github.com/yusiwen/minfer/issues/244
    #[cfg_attr(not(test), allow(dead_code))]
    Interleaved = 1, // LLaMA/Mistral: pairs [2*i, 2*i+1]
}

/// One entry per rotated pair, and `hd` never exceeds 256 anywhere the engine
/// runs (the same bound `cpu_backend::cpu_rope` already relied on).
pub const ROPE_FREQS_MAX: usize = 128;

/// #724: the per-pair frequency **factor** a `llama3`-scaled GGUF carries —
/// either as the `blk.N.rope_freqs.weight` tensor llama.cpp's converter writes,
/// or synthesised from `*.rope.scaling.*` metadata (see [`llama3_freqs`]).
///
/// Entry `d` scales pair `d` by dividing its inverse frequency:
///
/// ```text
/// freq(d) = freq_scale / base^(2d/hd) / f[d]
/// ```
///
/// `n == 0` means "no scaling": every kernel then evaluates exactly the
/// expression it has always evaluated, which is what makes an unscaled model's
/// output **bit-identical** across this change (issue #724's second acceptance
/// criterion) instead of merely close.
///
/// `#[repr(C)]` because both device backends copy this struct into a `constant`
/// buffer as raw bytes: the Rust layout and the kernel layout are one contract.
#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct RopeFreqs {
    n: i32,
    v: [f32; ROPE_FREQS_MAX],
}

impl RopeFreqs {
    /// No scaling: the kernels' original expression.
    pub fn none() -> Self {
        Self {
            n: 0,
            v: [0.0; ROPE_FREQS_MAX],
        }
    }

    /// Adopt a table read from a GGUF tensor (or synthesised). `None` for an
    /// empty table or one wider than this build's vocabulary — the caller
    /// refuses with a named error rather than truncating a frequency table.
    pub fn from_slice(f: &[f32]) -> Option<Self> {
        if f.is_empty() || f.len() > ROPE_FREQS_MAX {
            return None;
        }
        let mut v = [0.0; ROPE_FREQS_MAX];
        v[..f.len()].copy_from_slice(f);
        Some(Self {
            n: f.len() as i32,
            v,
        })
    }

    pub fn len(&self) -> usize {
        self.n.max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n <= 0
    }

    /// The factor for pair `d` (`1.0` outside the table, i.e. "unscaled").
    pub fn at(&self, d: usize) -> f32 {
        if d < self.len() {
            self.v[d]
        } else {
            1.0
        }
    }
}

impl Default for RopeFreqs {
    fn default() -> Self {
        Self::none()
    }
}

/// Compact on purpose: `NodeMeta` is formatted in every backend's error path, and
/// 128 floats per node would bury the message it is part of.
impl std::fmt::Debug for RopeFreqs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return f.write_str("RopeFreqs(none)");
        }
        write!(
            f,
            "RopeFreqs(n={}, first={:?}, last={:?})",
            self.len(),
            self.v[0],
            self.v[self.len() - 1]
        )
    }
}

/// Everything a KV re-rope needs: the base, the frequency scale and (for a
/// `llama3`-scaled model) the per-pair table. One struct so a caller cannot pass
/// two of the three and rotate by the wrong angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub freqs: Option<RopeFreqs>,
}

/// The inverse frequency of pair `d` before any scaling: `1 / base^(2d/hd)`.
pub fn inv_freq(freq_base: f32, d: usize, hd: usize) -> f32 {
    1.0 / freq_base.powf((2 * d) as f32 / hd as f32)
}

/// The inverse frequency a RoPE kernel uses, in **one** place so the CPU
/// reference and the issue's gates cannot drift from the kernels:
/// unscaled models keep today's `freq_scale / base^(2d/hd)`, and a table divides
/// it per pair. The device kernels (`rope.metal`, `common.cuh`) mirror this
/// expression line for line.
pub fn rope_inv_freq(
    freq_base: f32,
    freq_scale: f32,
    freqs: Option<&RopeFreqs>,
    d: usize,
    hd: usize,
) -> f32 {
    let unscaled = freq_scale / freq_base.powf((2 * d) as f32 / hd as f32);
    match freqs {
        Some(f) if !f.is_empty() => unscaled / f.at(d),
        _ => unscaled,
    }
}

/// #724: `llama3` rope scaling, transcribed from llama.cpp's converter
/// (`conversion/llama.py`, `LlamaModel.generate_extra_tensors`, build b11829 /
/// `184f65360`), which is where the `rope_freqs` tensor's values come from.
///
/// The band is chosen by wavelength: the high-frequency pairs (wavelength below
/// `orig_ctx / high_freq_factor`) are untouched, the low-frequency ones
/// (wavelength above `orig_ctx / low_freq_factor`) are divided by `factor`, and
/// the pairs in between are interpolated so the table is continuous at both
/// edges. The converter's comment names the one degenerate input — a band whose
/// two factors coincide (Llama 4) — and this refuses it by name instead of
/// dividing by zero.
pub fn llama3_freqs(
    freq_base: f32,
    factor: f32,
    orig_ctx: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    hd: usize,
) -> Result<RopeFreqs, String> {
    if hd < 2 || hd % 2 != 0 {
        return Err(format!(
            "llama3 rope scaling needs an even head dim, got {hd}"
        ));
    }
    if hd / 2 > ROPE_FREQS_MAX {
        return Err(format!(
            "llama3 rope scaling needs {} pairs, over this build's limit of {ROPE_FREQS_MAX} (hd={hd})",
            hd / 2
        ));
    }
    if !(factor > 0.0) || !(orig_ctx > 0.0) {
        return Err(format!(
            "llama3 rope scaling needs factor > 0 and original_context_length > 0, got \
             factor={factor}, original_context_length={orig_ctx}"
        ));
    }
    if !(high_freq_factor > low_freq_factor) {
        return Err(format!(
            "llama3 rope scaling needs high_freq_factor > low_freq_factor (the smooth band is \
             empty otherwise), got low={low_freq_factor}, high={high_freq_factor}"
        ));
    }
    // The converter iterates its f32 frequency tensor but does the wavelength and
    // band arithmetic in python floats (f64), rounding once when the table becomes
    // a `float32` tensor. Computing in f64 here and rounding once is the same
    // shape — the last ulp still differs, because torch's f32 `pow` and this
    // crate's `powf` need not agree on the final bit, which is why the gate pins
    // the real files' numbers with a 1e-6 relative tolerance.
    let (lo, hi, fac) = (
        low_freq_factor as f64,
        high_freq_factor as f64,
        factor as f64,
    );
    let orig = orig_ctx as f64;
    let low_wavelen = orig / lo;
    let high_wavelen = orig / hi;
    let mut v = [0.0f32; ROPE_FREQS_MAX];
    for (d, slot) in v.iter_mut().enumerate().take(hd / 2) {
        let wavelen = 2.0 * std::f64::consts::PI / inv_freq(freq_base, d, hd) as f64;
        *slot = if wavelen < high_wavelen {
            1.0
        } else if wavelen > low_wavelen {
            factor
        } else {
            let smooth = (orig / wavelen - lo) / (hi - lo);
            (1.0 / ((1.0 - smooth) / fac + smooth)) as f32
        };
    }
    // `from_slice` cannot fail here: the loop above is bounded by the checks.
    RopeFreqs::from_slice(&v[..hd / 2]).ok_or_else(|| "llama3 table out of range".to_string())
}
