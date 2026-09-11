# 10 · CPU matmul: quantized weights × Q8_0 activations

> **Stage**: prefill invoked (doc 09) → **this stage: inside a CPU `MatMul` node** → attention and vec ops (doc 11).
> **Code**: `src/kernel.rs` (`cpu_quant_matmul_f32`, `mm_rows`, the thread pool, `embed_tokens`), `src/quants.rs` (`quantize_row_q8_0`, `quantize_row_q8_k_buf`, the `dot_*` kernels), `src/block.rs` (block layouts).

## 1. Background — where this stage sits

Doc 09 followed the prefill call from `main.rs` into the graph machinery. Doc 08 ended at the moment the scheduler walks a split and calls `execute_node` for every node. For a `MatMul` node on the CPU backend, that call bottoms out in the code this document walks: the quantized matrix multiplication that produces essentially every number a transformer computes.

First, the vocabulary, because the rest of the series assumes it:

- A **matrix multiply (matmul)** takes a matrix of weights `W` and a matrix of activations `X` and produces an output where element `(t, o)` is the **dot product** of output row `o` of `W` with token row `t` of `X`. A dot product multiplies matching elements and adds them up: `Σ W[o][i] · X[t][i]`. In a transformer, roughly 169 of the 440 nodes of a 0.5B prefill graph are matmuls (doc 05's census) — they dominate both compute time and model size.
- **Weights** are the learned matrices loaded from the GGUF file (doc 03). In a 7B model they total ~4.4 GB. Storing them as plain 32-bit floats would need ~14 GB; storing them **quantized** — fewer bits per number, organized in small groups with a shared scale — is what makes local inference practical at all.
- **Quantization** (for this doc) means: take a group of 32 floating-point values, find the largest absolute value in the group, store that as one shared half-precision **scale** (`d`), and store the 32 values as small integers relative to that scale. Dequantizing gives `value ≈ q · d` — the integers times the scale. The error is bounded by half a scale step per value, which for a well-chosen group size is small enough that llama.cpp ships this design at 7B+ scale with usable quality.

minfer's CPU matmul follows llama.cpp's core trick, which this document makes explicit:

> **The weights stay quantized on disk and in memory — they are never dequantized as a whole. Activations are quantized to int8 on the fly, one block at a time, and the inner loop is an int8×int8 dot product with per-block scales folded in.**

Why that is the winning design is §2. The implementation walk is §3: the dispatch (`mm_rows`), the activation quantizers (`quantize_row_q8_0` and the K-quant `quantize_row_q8_k_buf`), the AVX2 kernel line by line, its scalar reference, the NEON/SDOT counterpart, the persistent thread pool, and the `repr(C)` block layouts that make all of it possible. Doc 14 and doc 15 reuse the same math on the GPU with different execution models.

## 2. Principle — why quantized weights × Q8_0 activations

### 2.1 The bandwidth argument (why not dequantize once at load?)

The obvious alternative to on-the-fly work: at load time, dequantize every weight to `f32` and run a plain float matmul. The numbers kill it.

During **decode** (doc 13), the engine produces one token per forward pass. Every matmul must read its **entire weight matrix** to produce that one token's outputs — with one token there is nothing to amortize over. The 7B model's ~4.4 GB of `q4_K_M` weights therefore stream from RAM **per token**. On a memory subsystem moving ~60–100 GB/s, that alone sets a ceiling of roughly 15–25 tokens/s — and that ceiling is *the* decode physics on CPU (the same argument appears on the GPU side, where doc 15 calls it "weight streaming").

Now compare the two storage options:

| Storage | 7B weight bytes | Stream time @ ~80 GB/s | Decode ceiling |
|---|---|---|---|
| `f32` weights | ~28 GB (7B × 4 B) | ~350 ms/token | ~3 tok/s |
| `q4_K_M` weights (4.5–5.5 bits/weight) | ~4.4 GB | ~55 ms/token | ~18 tok/s |

The same arithmetic per byte-of-precision: Q4_0 spends `18 B / 32 values = 0.5625 B` per weight — **7.1× less traffic per multiply** than the 4 bytes an `f32` weight costs. Quantization is not a quality knob here — it is the difference between usable and unusable. Dequantizing 4.4 GB into 28 GB of `f32` at load would also cost the load itself seconds and 6× the RSS.

### 2.2 Why the *activations* also go int8 (the Q8_0 trick)

The weights being 4-bit is only half the story. The inner loop must multiply weight elements by activation elements. If activations stayed `f32`, every inner-loop iteration would need a convert (int→float) plus a float multiply-add — and the CPU's fastest small-integer machinery goes unused.

So minfer (following llama.cpp) quantizes the activations to **Q8_0** — per 32-value block, one `f16` scale + 32 `int8` values — right before the matmul (§3.2). The inner loop then becomes an **int8×int8 dot product**, for which x86 has `vpmaddubsw`/`vpmaddwd` (the AVX2 path below) and ARM has `SDOT` (the NEON path): single instructions that multiply and add **8–16 integer pairs each**. The per-block scales are folded in once per block, outside the integer loop, as one float multiply-add.

The accuracy story: activations are quantized per 32 values with their own exact scale, so the input side carries no cross-tensor approximation drift; weights are 4-bit but *their* scales were chosen at model-conversion time by the quantizer with the whole tensor in view. The repo's verification records show the resulting CPU logits matching llama.cpp bit-for-bit — the format is the same, the kernel math is the same, and the accumulated error across 28 layers lands identically.

### 2.3 The dispatch in one picture

For a matmul node with weight type `T`, output `od` rows, input dim `id`, `nt` tokens:

```text
x: [nt][id] f32 (token-major, from the allocator's f32 pool — doc 07)
        │
        ▼  quantize_row_q8_0_buf   (K-quant weights → quantize_row_q8_k_buf instead)
x_q: [nt][id] int8 blocks + f16 scales (34 B per 32 values, or 306 B per 256)
        │
        ▼  cpu_quant_matmul → thread pool → mm_rows (one row per worker)
w row o: [id] quantized blocks (18/22/24/34/144/176/210 B per 32/256 values)
        │
        ▼  dot_<T>_q8_0(wrow, xrow)   ← the inner loop, per (output, token)
out: [nt][od] f32
```

The block sizes come from `block.rs` and are the contract between the GGUF file (doc 02), the loader (doc 03), and the kernels:

```rust
// src/block.rs:18-27
pub const Q4B: usize = 18;  // sizeof(block_q4_0)
pub const Q8B: usize = 34;  // sizeof(block_q8_0)
pub const Q4KB: usize = 144; // sizeof(block_q4_k)
pub const Q6KB: usize = 210; // sizeof(block_q6_k)
...
pub const Q8KB: usize = 2 + 256 + 16 + 32; // 306  (block_q8_k)
```

Sanity check with a real number: `token_embd` of Qwen2.5-0.5B in Q4_0 with `id = 896` has `896/32 = 28` blocks per row, so one weight row is `28 × 18 = 504` bytes — and doc 02's byte-size formula `(n / blck_size) × type_size` lands on exactly that.

### 2.4 Two activation formats, one pairing rule

The entry-point branch (`kernel.rs:12-33`, below) is not cosmetic — each weight family pairs with a **specific activation format**, fixed by the kernels' inner loops:

| Weight family | Weight block | Paired activation format | Activation block |
|---|---|---|---|
| Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 | 32 values (18/20/22/24/34 B) | **Q8_0** | 32 values, 34 B: `f16 d + 32 × i8` |
| Q4_K / Q5_K / Q6_K | 256-value super-blocks (144/176/210 B) | **Q8_K** | 256 values, 306 B: `f16 d + 256 × i8 + 16 reserved + 16 × i16 bsums` |

The Q8_K pairing exists because the K-quant kernels unpack one 256-value weight super-block — with its 8 sub-block scales and mins — and want the matching 256-value activation super-block with its own group sums (`bsums`, used by the K-quant dot's correction term) adjacent. The layout comment is pinned in the source:

```text
// src/quants.rs:790 — activation q8_K block: d(f16) + qs[256 i8] +
// bsums[16 i16] = 306 bytes (crate::block::Q8KB).
```

(Note a subtlety doc 02 flagged: `block.rs`'s `BlockQ8_K` *struct* — a `f32 d` ggml-style layout, `block.rs:173-177` — is not the same 306-byte arrangement the activation quantizer writes. The activation-side format is defined by the quantizer + dot kernel pair, and that is the only contract the matmul path relies on.)

## 3. Implementation

### 3.1 Data in / data out

| Item | Shape / layout | Where it comes from |
|---|---|---|
| Activations `x` | `[nt][id]` f32, token-major | the allocator's f32 pool (doc 07), filled by the previous node's output |
| Quantized activations `x_q` | `[nt]` rows of `id/32` × 34 B (Q8_0) or `id/256` × 306 B (Q8_K) | allocated per call by `cpu_quant_matmul_f32` |
| Weights `w` | `[od][id]` row-major **quantized bytes**, exactly as the GGUF has them | `Tensor.data` — a borrowed slice of the mmap'd file (docs 02/03); row stride = `blocks_per_row × block_size` |
| Output `out` | `[nt][od]` f32, token-major | the node's output buffer in the pool (doc 07) |

Note what is *absent*: any `f32` copy of the weights, anywhere. The GGUF bytes flow untouched from file page cache → `Tensor.data` → the dot kernel.

**One call, in real numbers.** Prefill of 100 tokens through Qwen2.5-0.5B's `blk.0.attn_q` (an `[896, 896]` Q4_0 weight, so `od = id = 896`, 28 blocks/row) with `--threads 8`:

| Step | Work | Bytes touched |
|---|---|---|
| quantize activations | `quantize_row_q8_0_buf` on `100 × 896` f32 | read 358 KB, write `100 × 28 × 34 = 95 KB` |
| submit | `MmJob` under `job`, bump `gen`, pool wakes | ~64 B |
| 8 workers × 112 rows each | `mm_rows` rows 0..896 | weights: `896 × 28 × 18 = 451 KB`; activations re-read per row: `112 × 95 KB` from cache |
| dot kernels | 896 × 100 = 89,600 calls to `dot_q4_0_q8_0` | 28 block-iterations each → 2.5M block dots |
| output | `out[t][o] = Σ` | write `100 × 896 × 4 = 358 KB` f32 |

Every number here comes straight from the formulas of §2.3/§3.2 — this table is the sanity check to run in your head whenever a matmul result looks wrong: block counts (`id/32`), row strides (`blocks × block_size`), and output size (`nt × od × 4`) must all be whole and consistent.

### 3.2 Key code

**The entry point: quantize activations, then dispatch.** `cpu_quant_matmul_f32` is what the CPU backend's `MatMul` arm calls (doc 08):

```rust
// src/kernel.rs:12-33
pub fn cpu_quant_matmul_f32(w: &Tensor, x: &[f32], out: &mut [f32],
                            od: usize, id: usize, nt: usize) {
    match w.ttype {
        TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => {
            let n_super = id / 256;
            let mut qb = vec![0u8; nt * n_super * Q8KB];
            crate::quants::quantize_row_q8_k_buf(x, nt, id, &mut qb);   // → Q8_K
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
        _ => {
            let nbe = id / 32;
            let mut qb = vec![0u8; nt * nbe * Q8B];
            crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut qb);   // → Q8_0
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
    }
}
```

The two activation formats of §2.4 are chosen *here*, once, so no call site can ever pair the wrong kernel with the wrong activation bytes.

**The row kernel: `mm_rows`.** One function handles every weight type; the type only changes the byte arithmetic and which `dot_*` gets called:

```rust
// src/kernel.rs:130-152 (Q4_0 arm; the other 7 arms are the same shape)
unsafe fn mm_rows(job: &MmJob, r0: usize, r1: usize) {
    let od = job.od;          // outputs (weight rows)
    let id = job.id;          // input dim
    let nt = job.nt;          // tokens
    match job.ttype {
        TensorType::Q4_0 => {
            let nb = id / 32;          // blocks per row
            let ws = nb * Q4B;         // weight row stride: 28 × 18 B for 0.5B
            let rowb = nb * Q8B;       // activation row stride: 28 × 34 B
            for o in r0..r1 {          // this worker's rows
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q4_0_q8_0(wrow, xrow);
                }
            }
        }
        ...
```

Read the loop order carefully — it is the parallelism story: **the outer loop is over output rows `o`, and each row is written by exactly one worker**. That is why the comment at the top of the function can promise the multi-threaded result is *bit-identical* to single-threaded: no output element is touched twice, so there is no float summation order to disagree about.

The other arms only change the stride constants and the kernel name: `Q8_0` has `ws == rowb` (both 34 B per block); the K-quant arms compute strides in 256-value units (`nk = id/256`, weight row `nk × 144 / 176 / 210` B, activation row `nk × 306` B) and call `dot_q4_k_q8_k` / `dot_q5_k_q8_k` / `dot_q6_k_q8_k`; `Q5_0`/`Q5_1` add their high-bit planes (22/24 B blocks). Also note `job.w.add(o * ws)`: the weight "matrix" is a byte pointer plus a stride — the loader never materialized an `f32` matrix (doc 03), and the kernel never asks for one.

**One AVX2 kernel, line by line.** `dot_q4_0_q8_0` first picks its engine (all the `dot_*` wrappers share this shape):

```rust
// src/quants.rs:37-58 (abridged to the dispatch)
pub fn dot_q4_0_q8_0(q4: &[u8], q8: &[u8]) -> f32 {
    let nb = q8.len() / Q8B;                       // 32-value blocks
    #[cfg(target_arch = "x86_64")]
    { if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
          return unsafe { dot_q4_0_q8_0_avx2(q4, q8, nb) }; } }
    #[cfg(target_arch = "aarch64")]
    { if neon_enabled() { return unsafe { dot_q4_0_q8_0_neon(q4, q8, nb) }; } }
    dot_q4_0_q8_0_scalar(q4, q8, nb)               // portable fallback
}
```

Runtime detection (`is_x86_feature_detected!`) is why one binary ships everywhere and picks AVX2 only where it exists; doc 06's `supports_op` story is about *ops*, this one is about *instructions*, and both follow the same capability-query philosophy.

The AVX2 kernel processes one 32-value block per iteration with two 256-bit registers:

```rust
// src/quants.rs:98-121 (core loop, comments mine)
unsafe fn dot_q4_0_q8_0_avx2(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        let xp = x.as_ptr().add(ib * Q4B);   // weight block: 2 B scale + 16 B nibbles
        let yp = y.as_ptr().add(ib * Q8B);   // activation block: 2 B scale + 32 × i8
        let xd = f16_to_f32_bits(*xp.cast::<u16>());  // weight block scale
        let yd = f16_to_f32_bits(*yp.cast::<u16>());  // activation block scale
        let d = _mm256_set1_ps(xd * yd);     // scales fold in ONCE per block
        let tmp = _mm_loadu_si128(xp.add(2) as *const __m128i);   // 16 packed nibbles
        let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp); // high nibbles → hi lane
        let mut qx = _mm256_and_si256(bytes, _mm256_set1_epi8(0xF)); // keep 4 bits
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));  // unsigned 0..15 → signed −8..7
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);   // 32 × i8
        let ax = _mm256_sign_epi8(qx, qx);   // |qx|          (unsigned-ify)
        let sy = _mm256_sign_epi8(qy, qx);   // qy × sign(qx) (move w's sign onto acts)
        let dot = _mm256_maddubs_epi16(ax, sy);  // 16 u8×i8 → 8 × i16 pairwise dots
        let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
        acc = _mm256_fmadd_ps(d, q, acc);    // acc += (xd·yd) · q
    }
    hsum_float_8(acc)                        // horizontal add of the 8 lanes
}
```

The moves worth understanding:

1. **Nibble unpacking without a lookup table.** A 4-bit weight is stored as two nibbles per byte (low = element `2i`, high = element `2i+1`). One `_mm256_and_si256(0xF)` recovers the low halves; `_mm_srli_epi16(4)` on a 128-bit half plus `_mm256_set_m128i` re-packs the high halves into the upper lane — 32 centered nibbles in one register, no per-byte work.
2. **The sign trick.** `maddubs` multiplies *unsigned* × *signed* bytes. Weights are centered to −8..7 (signed), so the kernel takes their absolute value (`sign_epi8(qx, qx)`) and instead flips the activations' signs by the weights' signs (`sign_epi8(qy, qx)`). Same product, and now the fast unsigned×signed instruction applies.
3. **Scales outside the integer loop.** `d = xd·yd` is one scalar per block; the integer chain (`maddubs` → `madd` → widen) produces the exact integer dot for the block, and one `fmadd` folds it into the float accumulator. The integer math is *exact*; only the block quantization itself approximates.
4. **The horizontal sum** is its own small art — one lane value out of eight:

```rust
// src/quants.rs:424-429
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    _mm_cvtss_f32(_mm_add_ss(x128, _mm_movehdup_ps(x128)))
}
```

**The scalar kernel is the reference semantics.** Before admiring the SIMD, read the portable version — it is the mathematical definition every fast path must reproduce:

```rust
// src/quants.rs:123-145
fn dot_q4_0_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q4B..];
        let yb = &y[ib * Q8B..];
        let dx = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])); // weight scale
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]])); // activation scale
        let mut si = 0i32;
        for j in 0..16 {
            let v0 = (xb[2 + j] & 0x0F) as i8 - 8;        // low nibble, centered
            let v1 = (xb[2 + j] >> 4) as i8 - 8;          // high nibble, centered
            si += (v0 as i32) * (yb[2 + j] as i8 as i32);
            si += (v1 as i32) * (yb[2 + j + 16] as i8 as i32);
        }
        s += si as f32 * dx * dy;                          // exact int dot × scales
    }
    s
}
```

This is also the honest worked example. Take `dx = dy = 1.0` for clarity and a weight byte `xb[2] = 0x21` (binary `0010_0001`): low nibble `1 − 8 = −7`, high nibble `2 − 8 = −6` — two dequantized weights `−7·dx` and `−6·dx` from one byte. The i32 accumulator never rounds: the only approximation in the whole kernel happened when the values were quantized.

**The Q8_0 kernel is the same skeleton, minus unpacking.** With both sides already int8, the loop shrinks to load-scale-multiply-accumulate:

```rust
// src/quants.rs:166-187 (core, abridged)
unsafe fn dot_q8_0_q8_0_avx2(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        let xd = f16_to_f32_bits(*x.as_ptr().add(ib * Q8B).cast::<u16>());
        let yd = f16_to_f32_bits(*y.as_ptr().add(ib * Q8B).cast::<u16>());
        let d = _mm256_set1_ps(xd * yd);            // the two block scales
        let qx = _mm256_loadu_si256(x.as_ptr().add(ib * Q8B + 2).cast::<__m256i>());
        let qy = _mm256_loadu_si256(y.as_ptr().add(ib * Q8B + 2).cast::<__m256i>());
        let ax = _mm256_sign_epi8(qx, qx);          // |qx|, signs moved onto qy
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_maddubs_epi16(ax, sy);     // 16 u8×i8 → 8 × i16 dots
        let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    hsum_float_8(acc)
}
```

Why the `sign_epi8` dance again when Q8_0 values are already signed? Because `maddubs` insists on unsigned × signed, and the *weight* side must be the unsigned one — so the same abs-and-transfer trick from the Q4_0 kernel reappears. Once you have seen it twice, every minfer dot kernel reads as a variation on one theme: **unpack to centered int8 → exact integer dots → one float fold per block**.

This kernel is also the entire `lm_head` story: the output projection matmul (vocab ≈ 151k rows of Q8_0 for Qwen3) is this loop 151,936 times per token — doc 05's `n_out` tail-row optimization exists precisely to cut how many of those rows run.

**The NEON counterpart: one instruction, 16 MACs.** On aarch64 (Apple Silicon, doc 14's home turf) the equivalent of the unpack-and-multiply chain is a single instruction — `SDOT`, issued through inline assembly because Rust's `std::arch` exposed no stable intrinsic at the time:

```rust
// src/quants.rs:623-633
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn sdot_vec(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",   // acc += 4-way i8 dot of 16-byte lanes
        acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
        options(nomem, nostack),
    );
    acc
}
```

Each `sdot` takes two 16-byte int8 vectors and adds **sixteen** multiply-accumulates into four i32 lanes. The NEON `dot_q4_0_q8_0` (`src/quants.rs:636`) unpacks nibbles with NEON shuffles and drives `sdot_vec` per 16 bytes — the aarch64 answer to `maddubs`. `MINFER_NO_NEON=1` disables the whole NEON layer (`neon_enabled()`) and drops to scalar, which is how the optimization campaign A/Bs the SIMD paths.

**The activation quantizers.** The Q8_0 one, entry first:

```rust
// src/quants.rs:310-318 (entry; _buf variant at :389 writes into a caller buffer)
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<u8> {
    let k = x.len();
    debug_assert!(k % 32 == 0);
    let nb = k / 32;
    let mut y = vec![0u8; nb * Q8B];
    quantize_row_q8_0_to(x, &mut y);
    y
}
```

Per 32-value block: find `amax = max|x|`, store `d = amax / 127` as `f16` (2 bytes), store each `x[i]/d` rounded to nearest `i8` (32 bytes) → 34 bytes. The AVX2 variant (`quantize_avx2`, `src/quants.rs:321`) is worth reading because it shows the max-reduce and the rounding in registers:

```rust
// src/quants.rs:321-341 (core, abridged)
unsafe fn quantize_avx2(x: &[f32], y: &mut [u8], k: usize) {
    for i in 0..nb {
        let v0..v3 = /* four 8-float loads: the 32-value block */;
        let ma = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sb, v0), _mm256_andnot_ps(sb, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sb, v2), _mm256_andnot_ps(sb, v3)));
        // sb = -0.0; andnot(sb, v) = |v| — abs without an extra op
        let ms = /* horizontal max of ma */;
        let d = ms / 127.0f32;
        y[yo]..y[yo+1] = f16(d).to_le_bytes();       // the block scale
        let id = if ms != 0.0 { 127.0f32 / ms } else { 0.0f32 };
        let i0 = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(v0, _mm256_set1_ps(id)), _MM_ROUND_NEAREST as i32));
        ...
```

Two details to notice: the **absolute value is free** (`_mm256_andnot_ps` with `-0.0` clears the sign bit), and the zero-block guard (`ms != 0.0` → inverse 0) keeps an all-zero block from producing NaNs — a whole layer of `0.0/0.0` if skipped. The `debug_assert!(k % 32 == 0)` is where doc 02's alignment story pays off — every activation row is a whole number of blocks, so the quantizer never sees a partial block.

The K-quant activations quantizer (`quantize_row_q8_k_buf`, `src/quants.rs:798`) fills the 306-byte Q8_K blocks of §2.4. Its NEON worker (`:848`) shows every field:

```rust
// src/quants.rs:848-887 (core, abridged)
unsafe fn quantize_row_q8_k_buf_neon(row: &[f32], out: &mut [u8]) {
    for s in 0..n_super {
        let blk = &row[s * 256..(s + 1) * 256];
        let o = s * crate::block::Q8KB;                 // 306-byte stride
        let mut amax = 0.0f32;
        for g in 0..64 { /* amax over the 256 values, exact max reduction */ }
        let d = amax / 127.0f32;
        out[o]..out[o+1] = f16(d).to_le_bytes();        // ① block scale, f16
        for g in 0..16 {                                // 16 groups of 16 values
            let q8 = saturating_round(blk[g*16..g*16+16] * (1/d));   // ② int8 quants
            vst1_s8(out.as_mut_ptr().add(o + 2 + base) ..., q8);     //    at o+2..o+258
            bsums[g] = exact int sum of the SATURATED q8;            // ③ group sums
        }
        out[o + 258 + g] = 0;                           // ④ 16 reserved bytes, zeroed
        out[o + 274..o + 306] = bsums;                  //    16 × i16 group sums
    }
}
```

Three details carry weight: ② uses **saturating** narrowing (clamping to [−128, 127] exactly like the scalar `.clamp()`), ③ computes the `bsums` from the *saturated* values so the integer group sums are exact — the K-quant dot kernels use them for their correction term and any drift there would break parity — and ④ keeps the reserved field zeroed so the region reads deterministically. The AVX2/scalar paths write byte-identical layouts, which is what lets one kernel consume activations from any build.

**The payoff: a K-quant dot kernel, walked.** All of §2.4's structure (super-scales, 6-bit sub-scales, mins, `bsums`) exists to serve this loop — `dot_q4_k_q8_k_scalar` (`src/quants.rs:903-958`), the reference every K-quant fast path must match:

```rust
// src/quants.rs:903-958 (scalar, abridged but complete in structure)
fn dot_q4_k_q8_k_scalar(q4: &[u8], q8k: &[u8]) -> f32 {
    for i in 0..n_super {
        let d    = w_scale(i) * a_scale(i);          // super-scale × activation scale
        let dmin = w_dmin(i)  * a_scale(i);
        let (scales, mins) = block::unpack_q4k_scales(&q4b[4..16]);  // 12 B → 8+8 6-bit values
        // ① the MIN term: Σ mins[s] × (sum of quants in sub-block s)
        let mut mterm = 0i32;
        for s in 0..8 {
            let b0 = bsums(2 * s);  let b1 = bsums(2 * s + 1);   // from Q8_K's ③ above
            mterm += mins[s] as i32 * (b0 + b1);
        }
        sumf -= dmin * mterm as f32;
        // ② the VALUE term: nibble × int8 dots per sub-block, weighted by scales
        for j in 0..4 {
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for l in 0..32 {
                s_lo += (q4b[q4off + l] & 0x0F) as i32 * (q8b[q8off + l] as i8 as i32);
                s_hi += (q4b[q4off + l] >> 4) as i32 * (q8b[q8off + 32 + l] as i8 as i32);
            }
            sumi1 += s_lo * scales[2 * j] as i32;
            sumi2 += s_hi * scales[2 * j + 1] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}
```

The Q4_K value formula is `value = q · d_s − min_s` per sub-block `s` (a *non-centered* 4-bit scheme: instead of subtracting 8 like Q4_0, each sub-block carries its own min). Unrolling the math shows why the kernel has two terms:

```text
Σ_values (q·d_s − min_s)·a        per sub-block
= d_s · Σ(q·a)  −  min_s · Σa     ← the min term is a sum over ACTIVATIONS only
```

`Σ(q·a)` is the integer dot ②; `Σa` per sub-block is exactly what Q8_K's `bsums` carry — precomputed once at quantization time (doc 10's §3.2 ③) so the kernel never re-touches the activation bytes for the min correction ①. That is the entire reason the Q8_K format exists, and the reason doc 02's byte-size table and this kernel must agree on where `bsums` live (offset 274 in the 306-byte block — visible in both `quantize_row_q8_k_buf_neon` and `dot_q4_k_q8_k_scalar`).

**The persistent thread pool.** Decode runs ~250 matmuls per token (doc 13), each tiny (one token row through `od` rows of weights). Spawning threads per matmul measured **~170 µs** (kernel.rs's own comment) — against a per-token budget of a few milliseconds, that alone would be the bottleneck:

```text
// src/kernel.rs:274-281 (the dispatch every worker wakes for)
let job = *pool.job.lock().unwrap();
match job {
    PoolJob::MatMul(m) => {
        let (r0, r1) = chunk(pool.n + 1, my_idx, m.od);  // even row split
        unsafe { mm_rows(&m, r0, r1) };
    }
    PoolJob::ParFor(p) => { ... }                        // generic parallel-for
}
pool.done.fetch_add(1, Ordering::SeqCst);
```

The design (kernel.rs comment block, lines ~274–287):

- Workers are spawned **once**, lazily, by `get_pool` (`OnceLock`, process lifetime) and spin on an atomic `gen` counter — 8000 spin iterations, then `yield_now`, so idle workers cost nothing but wake in microseconds.
- The submitting thread publishes a `MmJob` under `job`, bumps `gen`, and waits on `done == n+1` (the main thread participates as the last worker — no wasted idle main).
- A `gate` `Mutex` serializes submissions: the comment (`kernel.rs:246-252`) records the real hazard — two concurrent callers (parallel tests, the multi-slot server) would clobber `job` and share `done`, letting one caller return before *its* range was computed while workers still read its stack-local context. That is a use-after-free, and the fix is one lock around submit→wait.
- `chunk(parts, idx, total)` splits `od` rows evenly; each row belongs to exactly one worker → bit-identical results at any thread count (§2.3's promise).
- Worker count: `set_cpu_threads` (CLI `--threads`, `kernel.rs:54`) must run before the first matmul because the pool is spawned lazily on first use; `cpu_threads()` (`:61`) reads the `CPU_THREADS` atomic where `0` means auto-detect.
- The same pool also serves `par_for` (`kernel.rs:362`) — which is exactly what attention reuses for per-head parallelism in doc 11.

**The embedding "matmul".** `embed_tokens` (`kernel.rs:389`) is not a matmul at all: for each token id it walks the quantized `token_embd` row block-by-block, dequantizing (scale × nibble for Q4_0/Q4_1, scale × byte for Q8_0) into the output row:

```rust
// src/kernel.rs:389-410 (Q4_0/Q8_0/Q4_1 arm, abridged)
pub fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
    for (ti, &id) in ids.iter().enumerate() {
        let idx = id as usize;                       // the token's row in the table
        for b in 0..nbp {                            // blocks in one embedding row
            let off = (idx * nbp + b) * bb;
            let d = fp16_to_f32(...);                // block scale
            ...                                      // dequantize 32 values → out row
        }
    }
}
```

This is the `GetRows` node of doc 05 made concrete — "the embedding table is a lookup" — and it exists here because CPU and GPU backends share the same row getter, keeping embeddings byte-for-byte identical across backends.

### 3.3 Design choices (why this shape and not another)

**Why not dequantize weights to f32 at load?** §2.1's arithmetic: 6× the RAM, 7× the per-multiply bandwidth, seconds of extra load time — and the GPU backends want the *quantized* bytes anyway (doc 14's shaders consume the same block layouts). The quantized bytes are not an intermediate representation; they are the storage format.

**Why Q8_0 (int8) for activations, not f32 or int4?** The inner loop is the hot code; int8×int8 has dedicated silicon on both ISAs (§2.2). f32 activations would forfeit `maddubs`/`SDOT`; int4 activations would double the quantization error exactly where values are least pre-calibrated (activations change every call; weights were tuned at conversion time). llama.cpp's CPU path makes the same choice, and matching it is what makes the bit-parity tests possible at all.

**Why do K-quant weights get a different activation format (Q8_K)?** The K-quant kernels' inner loop processes 256 values per super-block and needs the activation side in matching 256-value groups *with exact group sums* (`bsums`) for its correction term. A 34-byte Q8_0 stream would force the kernel to re-group and re-sum activations per call; the 306-byte Q8_K block carries everything precomputed (§2.4). The pairing is enforced in one place (`cpu_quant_matmul_f32`) so it cannot drift.

**Why per-row thread ownership instead of splitting the inner reduction?** Splitting a dot product across threads needs a reduction (adding partial sums in some order), and float addition is not associative — results would drift with thread count. One row per worker makes `--threads` a pure performance knob with zero numerical effect, which the whole verification methodology (greedy token equality across machines, doc 12's seeded gates) quietly depends on.

**Why is a `Vec<u8>` allocated for quantized activations on every matmul call?** It is sized per call (`nt × row_bytes`) because `nt` changes between prefill and decode. Doc 07's allocator owns the *node* buffers; this scratch lives one call deep and is invisible to the graph. (CPU_OPTIMIZATIONS.md's "batched QKV shared quantization" record — the prefill gain in GRAPH-REFACTOR-PLAN §12 — exists precisely because three sibling matmuls each re-quantizing the same `normed` row was visible in the profile.)

**Why are Q2_K / Q3_K / I-quants not supported?** Each extra format costs a hand-written AVX2 + NEON + scalar kernel triplet (×2 for the activation-format pairing) plus parity tests. The supported set — Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 (32-value) and Q4_K/Q5_K/Q6_K (256-value) — covers every GGUF quant llama.cpp recommends for quality at 4–8 bits; Q2_K/Q3_K trade real quality for size, and the I-quants' lookup-table design resists this kernel shape. `docs/SUPPORT-MATRIX.md` is the authoritative list.

### 3.4 Pitfalls & invariants

- **Block-size divisibility is a format contract, not a hint.** `id % 32 == 0` (and `id % 256 == 0` for K-quants) is guaranteed by GGUF conversion tools and asserted in the quantizer; a model that broke it would silently misindex without the assert.
- **The transposed-output bug that decode hid** (`vec_ops.rs:673-692`, the `mat_mul_f32` comment): an earlier version wrote `C[row*n + col]` — an `[m,n]` output — while every caller wanted `[nt][od]`. For decode (`nt == 1`) the two layouts coincide, so no decode-only test caught it; prefill output was transposed. Lesson the repo kept: **test with `nt > 1` or the token-major convention will bite.**
- **Row ownership = bit-identical parallelism.** Never "optimize" the pool into splitting a row's reduction; that trades away the determinism the verification gates rely on (§3.3).
- **The `gate` lock is load-bearing** (kernel.rs:246-252): removing it works in single-threaded tests and corrupts memory the first time two threads submit concurrently (the server's multi-slot path).
- **K-quant weights need Q8_K activations, 32-value weights need Q8_0** — crossing the pairing (e.g. feeding Q8_0 blocks to `dot_q4_k_q8_k`) misindexes the super-block scales. The `cpu_quant_matmul_f32` branch exists to make the pairing unstateable from the call site.
- **The activation-Q8_K layout is kernel-pair-defined** — 306 bytes as written by `quantize_row_q8_k_buf` (`quants.rs:790` comment), *not* the `BlockQ8_K` struct layout (`block.rs:173`); doc 02 flagged the same nuance on the on-disk side. When touching either side, re-verify the quantizer→kernel byte contract together.
- **Scales fold once per block, integers stay exact** — any refactor that converts intermediate integer dots to float mid-block changes the numerics and breaks parity with llama.cpp.

## 4. Observe & verify

- `cargo test quants::` / `cargo test kernel::` — per-kernel parity tests: every SIMD path is checked against the scalar reference you read in §3.2 (and, for the formats llama.cpp also ships, against llama.cpp-produced reference values).
- `cargo test --release` greedy token gates — end-to-end: doc 12's seeded/greedy gates (`-n 32 --greedy --seed 42`) would shift the instant any kernel changed one ulp of accumulated behavior.
- `MINFER_TIMING=1 ./target/release/minfer <model> "hi"` — splits per-token wall time into sampling vs forward (doc 09 §4); on CPU, forward *is* these kernels.
- `--threads N` — the pool's worker count; output must be identical for any N (that property is itself tested).
- `MINFER_NO_NEON=1` (aarch64) — forces scalar, the A/B lever for the NEON layer.

## 5. Cross-references

- [02 — GGUF load](02-gguf-load.md) §2.3/§3.2 — where the block layouts and byte-size formulas come from (and the on-disk Q8_K nuance this doc's §2.4 completes).
- [03 — Model dispatch and weights](03-model-dispatch-weights.md) — why `Tensor.data` is the raw GGUF bytes.
- [08 — The scheduler](08-scheduler-execute.md) §3.2 — the `MatMul` dispatch arm that lands here.
- [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) — the non-matmul half of the layer; reuses this doc's `par_for` pool for per-head parallelism.
- [13 — The decode loop](13-decode-loop-graph-reuse.md) — why decode is bandwidth-bound (250 matmuls/token, each streaming weights).
- [14](14-metal-backend.md) / [15](15-cuda-backend.md) — the same math with GPU execution models (f32 activations on Metal; int8 MMQ tensor-core GEMM on CUDA).
- `docs/CPU_OPTIMIZATIONS.md`, `docs/SUPPORT-MATRIX.md` — the optimization history and the authoritative quant-format matrix.

← [09 — Prefill: the first forward](09-prefill-forward-path.md) · [Index](./README.md) · [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) →
