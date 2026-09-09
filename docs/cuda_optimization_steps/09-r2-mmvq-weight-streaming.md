# 09 · R2 — MMVQ weight-streaming rework (LANDED)

> **Result**: 7B q4_k_m decode: tg128 42.2 → 45.1 tok/s (**+6.9%**), @2K 36.7 → 38.8 (+5.7%); the gap to llama.cpp (47.1 / 44.9) narrowed by about half in both cases.
> **Commit**: `6df3245`. **Date**: 2026-08-31.

## 1. Background — where things stood

At the end of Era A the engine's decode (nt==1) ran on the MMVQ path 8e landed (see chapter 06):
the activation vector is quantized to q8_0 before every matmul, weights stay in native GGUF
format without dequant, the kernel does 4-byte-packed integer dot products with `__dp4a`, and
the launch parameters copy llama.cpp's `MMVQ_PARAMETERS` table. That step lifted 7B q4_K decode
by +37%, but measured against llama.cpp in the same window it still trailed:

- tg128 (generation at a 128-token context): 42.2 vs 47.1 tok/s;
- @2K (a 2048-token context): 36.7 vs 44.9 tok/s.

The gap is ~10% (short context) to ~18% (long context). The @2K portion comes mainly from
split-attention (R4's story — chapter 10), while the tg128 gap falls almost entirely on the
matmul kernel itself: each decode step's wall time is basically "stream the ~4.5 GB of weights
through VRAM once", with attention and elementwise a mere rounding error.

The record's calibration of the 8e kernel: effective weight-stream rate ~147 GB/s versus llama's
same-class kernel at ~197 GB/s — the record summarized it as "8e runs at only about 60% of
llama's effective stream rate" (the per-kernel normalization basis for the two numbers was not
preserved in the summary; the ratio itself defers to the session record). GB10's DRAM roofline
is 273 GB/s (calibrated by r55's measurement), meaning minfer's decode kernels during their
active period consume barely half of DRAM peak — far from the bandwidth ceiling, showing the
bottleneck is not "the bytes cannot move" but "how the moving instructions are organized".

Where things would stick without this step: decode is the metric users feel most (generation
speed), and its theoretical ceiling is decided solely by the weight-stream rate. As long as the
MMVQ kernels' stream efficiency stays at ~150 GB/s, every later decode-side optimization (R4's
attention, the D series' quantization folding) is pressed down by this foundation. R2 chose to
raise the matmul kernels' stream efficiency before touching attention, because the attribution
was clear: the tg128 gap is 100% matmul, and the matmul gap is structural (load instructions per
byte) — not something parameter tuning can fix.

## 2. Principle — the GPU mechanism

**Weight-streaming-bound decode.** At nt==1 every matmul is a matrix-vector product `y = W·x`:
producing one output row requires reading that row of the weights once and dotting it with x,
which resides in registers/L1. Weight bytes are **used exactly once per token** — there is no
k-fold inner-product reuse (that is the prefill GEMM's dividend), so decode's ideal speed is
simply:

```
ideal tok/s = effective DRAM bandwidth ÷ weight bytes per token
```

7B q4_K_M's weights are ~4.5 GB; DRAM roofline 273 GB/s gives an ideal ceiling on the order of
~60 tok/s. Both engines measure far below it (llama 47.1, minfer 42.2), and **the difference
between them** can only come from how efficiently each kernel streams weight bytes into the
pipeline — how many instructions and memory transactions each byte costs.

**Effective GB/s** is the quantity defined on exactly this basis: `weight bytes streamed per
token × tok/s`. It translates "tok/s" into the kernel's physical workload, freeing the
minfer/llama comparison from implementation differences between model layers.

**The v1 (8e) kernel's two wastes.** A q4_K 256-element super-block contains 8 32-element
sub-blocks; adjacent sub pairs (even/odd) **share the same 32 B nibble chunk** — the even sub
takes the low 4 bits, the odd sub the high 4. v1's thread mapping was "one sub-block per
thread":

- Each sub's thread reads the whole 32 B chunk in via 8 4-byte loads, using only half the
  nibbles; **the sibling sub's other thread reads the same 32 B again** — every weight byte is
  touched by load instructions twice per row;
- The second read hits L1 (the same sector), so DRAM bytes do not double, **but the
  load-instruction count per byte doubles**. The kernel at that point is issue/latency-bound,
  not bandwidth-bound; the instruction stream is stuffed with redundant loads and long
  scoreboard latency cannot be effectively spread;
- q6_K is worse: the raw layout's block stride is 210 B (not a multiple of 4), so ql/qh can only
  be read 2 bytes at a time (llama.cpp `get_int_b2` style) — one 16 B ql piece takes 8 loads.

**v2's structure.** The thread mapping changes from "one sub per thread" to "one sub-PAIR (64
elements) per thread": the pair's two subs share the nibble bytes, the chunk is read once via 2
16-byte `uint4` loads and serves both subs' half-nibbles — exactly 1 load instruction per weight
byte per row (with the uint4 widening, per-pair weight loads drop from 16 4B loads to 2 16B
loads). On the q6_K side, 7e② had already padded the block stride to 224 B (14×16), making every
ql/qh piece in a block 16 B aligned, so uint4 loads are legal. q5_K's qh plane is a 32 B shared
by all 8 subs, reusable via bit indexing alone.

In one sentence: **the bytes were not wasted (L1 caught them) — what was wasted was the
instruction stream; v2 cuts the instruction stream back.**

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

One v2 kernel each for the three types (q4_K/q5_K/q6_K), coexisting with v1, selected by a
**dispatch gate**:

- **The gate condition is `id % 256 == 0`** (whole super-blocks). v2's pair mapping assumes id
  divisible by 256 — shapes with a partial tail (e.g. id=2176 = 8.5 super-blocks) do not map
  and must stay on v1. This is precisely the root of the later pitfall (§3.3).
- q6_K's v2 **additionally requires the padded 224 B stride**: the uint4 alignment was bought by
  the padded layout; the raw 210 B layout keeps v1.
- `MINFER_MMVQ_V1=1` forces v1 back — preserving the ability to A/B measure and
  regression-compare (a habit running through the whole campaign: every default switch keeps a
  "0"/"V1" opt-out).
- The grid shape is unchanged (`grid(od, nt)`, one 256-thread block per output row): v2 doubles
  each thread's work and halves thread coverage, but the reduce structure (warp shuffle +
  block tree) and llama's launch table are untouched — change the mapping, not the scheduling,
  so the A/B attributes to exactly one thing.

### 3.2 Key code

**Before — v1 q4_K: one sub per thread, the whole chunk read with only half used** (current tree
`src/cuda_kernels.cu`, `q4_k_q8_mmvq`):

```cuda
for (int u = threadIdx.x; u < nsub; u += 256) {   // u = 32-element sub-block
    const int blk_i = u >> 3, sub = u & 7;
    const uint8_t* blk = weights + (size_t)row * row_stride + blk_i * Q4KB;
    uint8_t s8, m8;
    get_scale_min_k4(sub, blk + 4, &s8, &m8);
    // sub-block nibbles: chunk (sub>>1) of 32B, lo nibbles for even sub,
    // hi for odd; element l of the sub-block ↔ byte l.
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
    const bool lo = (sub & 1) == 0;
    ...
    #pragma unroll
    for (int v = 0; v < 8; v++) {                 // 8×4B = the whole 32B chunk
        const uint32_t w = qw[v];
        const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
        const int xa = (int)xw[v];
        dot = __dp4a(n, xa, dot);
        sx  = __dp4a(0x01010101, xa, sx);
    }
    acc += d8 * ((float)s8 * (float)d * (float)dot
               - (float)m8 * (float)dm * (float)sx);
}
```

Each sub thread issues 8 4B loads; the same chunk is read again verbatim by the sibling sub's
thread — a 32 B chunk consumes 16 load instructions per row in total.

**After — v2 q4_K: one sub-PAIR per thread, the chunk read once for both subs** (commit
`6df3245`, also in the current tree, `q4_k_q8_mmvq_v2`):

```cuda
for (int u = threadIdx.x; u < npair; u += 256) {  // u = 64-element sub-PAIR
    const int kbx = u >> 2, c = u & 3;
    const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q4KB;
    const int s0 = 2 * c, s1 = 2 * c + 1;
    get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
    get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
    // one 32B nibble chunk: lo nibbles = sub s0's 32 elements, hi = sub s1's
    // (16B-aligned: 144·kbx + 16 + 32·c ≡ 0 mod 16)
    const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32);
    const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32 + 16);
    const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    ...
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        const uint32_t wv = ws[v];
        const int xa_v = (int)xa[v], xb_v = (int)xb[v];
        dota = __dp4a((int)(wv & 0x0F0F0F0F), xa_v, dota);        // sub s0
        sxa  = __dp4a(0x01010101, xa_v, sxa);
        dotb = __dp4a((int)((wv >> 4) & 0x0F0F0F0F), xb_v, dotb); // sub s1
        sxb  = __dp4a(0x01010101, xb_v, sxb);
    }
    acc += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
         + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
}
```

The same 32 B chunk: **2 16B loads, two `__dp4a` accumulators advancing in parallel**. The low
nibbles feed s0's dot directly, `(wv >> 4) & 0x0F0F0F0F` feeds s1 — two views of one datum, zero
repeated loads. The alignment comment is a hard guarantee: `144·kbx + 16 + 32·c` is always a
multiple of 16 against q4_K's 144 B block header.

**q6_K v2: the uint4 bought by the padded stride** (`q6_k_q8_mmvq_v2`, abridged):

```cuda
// v1 mapping with s = 2*pair + half: chunk = s>>3 = pair>>2,
// g = (s>>1)&3 = pair&3, is = s&1 = half (the pair's two subs share
// chunk/g; only the 16-byte is-half differs)
const int chunk = pair >> 2, g = pair & 3;
// padded 224B stride ⇒ every ql/qh piece is 16B aligned
const uint4 qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32);
const uint4 qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32 + 16);
const uint4 qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
const uint4 qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
```

Compare the v1 kernel's header-comment confession (the comment before the current tree's
`q6_k_q8_mmvq`): *"q6_K block strides are 210B raw / 224B padded (7e② repack) — both even but
not 4-aligned, so the weight side reads 2-byte halves (llama.cpp get_int_b2 style)"* — one 16 B
ql piece costs 8 2B loads; v2 does the same mapping with 1 uint4.

**q5_K v2: the qh plane shared via bit indexing** (`q5_k_q8_mmvq_v2`, abridged):

```cuda
// the qh plane is 32 bytes SHARED by all 8 sub-blocks (byte l holds
// one high bit per sub for element l) — every chunk reads the same
// bytes, only the bit index (s0/s1) differs
const uint4 h0 = *reinterpret_cast<const uint4*>(blk + 16);
const uint4 h1 = *reinterpret_cast<const uint4*>(blk + 16 + 16);
...
const uint32_t hia = (((qhv >> s0) & 0x01010101u) << 4);
const uint32_t hib = (((qhv >> s1) & 0x01010101u) << 4);
dota = __dp4a((int)((wv & 0x0F0F0F0F) | hia), xa_v, dota);
```

qh's byte layout is "byte l stores one high bit per sub for each of the 8 subs" — the same 32 B
serves all chunks, only the bit index changes with the sub. Shift + mask, 3 ALU instructions,
reinsert the high bit into the nibble, replacing v1's wrong shape of reading qh by chunk offset.

**The dispatch gate** (`src/cuda.rs`, the q4_K arm; shared as `mmvq_v2`):

```rust
fn mmvq_v2(id: usize) -> bool {
    id % 256 == 0 && !std::env::var("MINFER_MMVQ_V1").map_or(false, |v| v == "1")
}
```

The three type arms each pick between the v1/v2 launchers by this gate (the q6_K arm adds a
`blk_stride_padded` condition).

### 3.3 Pitfalls

**The gate condition hid the bugs for two weeks.** The first v2 passed the first suite round
carrying two bugs: q6_K's nibble-group computed wrong (written as `g = pair >> 1`, correct is `g
= pair & 3`), and q5_K's qh offset computed per chunk (correct is the shared 32 B indexed by
bit). Why it went uncaught: the parity/greedy shapes of the time used the non-square id=2176,
and `2176 % 256 = 128 ≠ 0` — **the dispatch gate sent it back to v1**. The v2 code was never
executed at all; the all-green tests were a fake green. Only after the id=2560 shape (10 whole
super-blocks, takes v2) was added did the engine-level greedy check immediately emit garbled
tokens, and the two bugs surfaced. The lesson was later written into the step-document rules:
**test shapes must cover every dispatch arm's gate condition, not just the kernel math**.

The remaining details went relatively smoothly: v1/v2 accumulate in a different order (within a
pair, the two subs each dp4a first, then add), but each output row's summation-tree structure is
unchanged, and greedy output is identical (v1 ≡ v2 token-for-token) — no need to touch the
tolerance gate.

## 4. Verification

- **Parity shape expansion**: the id=2560 shape was added to the parity sweep, specifically
  targeting the `id % 256 == 0` gate arm — defending against exactly §3.3's "gate routes the
  code around the test" fake green.
- **Engine-level greedy comparison**: the whole engine runs greedy generation, v1 ≡ v2 token by
  token — a gate one level above kernel unit tests, catching mapping/dispatch errors
  kernel-level tests miss (both of this step's bugs were caught by it).
- **Full suite 164 passing**: the regression safety net, confirming the change did not ripple
  into other types and paths.
- **Same-binary interleaved A/B**: tg128 and @2K each measured under both a quiet-GPU window and
  sglang contention, medians taken — defending against machine-state drift reading noise as
  gain (+6.9% is the quiet window; the contention window kept a +5–8% relative gain, same
  direction).

## 5. Results

| Metric (7B q4_k_m) | before (8e) | after (R2) | Δ | llama.cpp same window |
|---|---:|---:|---:|---:|
| tg128 decode | 42.2 | **45.1** | +6.9% | 47.1 (gap ~10% → ~4%) |
| @2K decode | 36.7 | **38.8** | +5.7% | 44.9 (gap ~18% → ~14%) |

Under the contention window (sglang on the same machine) the gain held at +5–8% relative. The
remaining 14% gap at @2K is mostly split-attention (handled by the next step, R4); the
tg128-side matmul gap narrowed to ~4%. At the kernel level, load instructions per weight byte
per row halved (re-reads eliminated) and q6_K's ql/qh went from 8×2B to 1×16B, lifting the
decode kernels' effective stream rate after v2 shipped — the record defers to whole-step tok/s
and the two-window consistency, and does not list a separate kernel GB/s after value.

This commit touches 6 files: `src/cuda_kernels.cu` +208 lines (the three v2 kernels + launcher),
`src/cuda.rs` dispatch wiring +131 lines, `src/graph/cuda_backend.rs` a 520-line refactor (v2
gate wiring), the rest documentation.

## 6. Lessons

1. **Test shapes must walk every dispatch arm's gate condition** — `id=2176` happened to land in
   the v1 gate, letting two v2 bugs pass all-green; adding one `id=2560` shape was worth more
   than ten more lines of unit tests.
2. **When the bottleneck is the instruction stream rather than the bytes, first check "how many
   load instructions per byte"**: L1 will hide re-read bytes, but it cannot hide redundant
   instructions in the issue stream.
3. **Data-path granularity (how much weight one thread covers) is a first-class design axis for
   MMVQ-class kernels**, of the same order as launch-table tuning; llama's parameter table
   gives the scheduling, but the bytes→threads mapping — the alignment and sharing
   relationships — you must derive yourself.
4. **Alignment is bought**: q6_K's uint4-ization depends on 7e②'s 224 B padded layout; the
   dispatch gate must check both premises — "id divisible" and "stride already padded".

---
← [08 · R1 int8 MMQ prefill GEMM](./08-r1-int8-mmq-prefill-gemm.md) · [Index](./README.md) · [10 →](./10-r4-split-attention-dim-parallel.md)
