# 41 · r38 — q6_K BT-style raw-byte mma kernel (LANDED)

> **Result**: 7B whole-prefill 1518.4 → 1561.9 tok/s (+2.87%, 3/3 pairs); matched-nt q6_K GEMM 368.9 → 221.8 µs/GMAC (1.66×).
> **Commit**: `75aabb9`. **Date**: 2026-09-05.

## 1. Background — where things stood

On 2026-09-05 the P6 MMQ campaign reached r38 — the starting point of Era D (q6_K / FA / prepass / promotion, r38–r60).

Before this, r28–r34 had already brought the q4_K MMQ line to convergence: r28 traded a raw-nibble B side plus lower smem for 2 blocks/SM, r29's kd-loop unroll cashed in the instruction-count reduction once those 2 blocks/SM were in place, r31 removed the bank conflict on scale reads, and r34 moved the A-side layout transform out of the kernel entirely (the quantize-transpose prepass, +9.72%). r37 then ran a post-parity whole-prefill attribution (master table row 51): whole-prefill 1521 tok/s, 2.15× vs llama.cpp — **but the same attribution table also showed that the q6_K GEMM alone cost 1094.7 ms, 51.2% of the entire prefill wall clock**, at 6.38×/GMAC unit throughput (i.e. each GMAC takes 6.38× as long as on the q4_K MMQ line, which sat at ~57.8 µs/GMAC at the time, inferred from 368.9/6.38).

In other words: the MMQ redesign made q4_K fast and left q6_K on a path slower than f16 (row 51's one-line conclusion: "MMQ made q4_K fast and left q6_K on a slower-than-f16 path — the next lever is a different kernel"). In the 7B q4_k_m model the q6_K tensors are `attn_v` and `ffn_down` (the r38 commit message verbatim: "q6_K GEMMs (attn_v + ffn_down, the r37 51.2% residual at 368.9 uS/GMAC)"); every prefill passes through them, and a 51.2% share of the wall clock makes them the next biggest lever.

It is worth a quick look at how the q4_K line converged — because r38's kernel shape is its template: r28's raw-nibble B-side kernel (1375.2 → 1410.4, +2.56%, squeezing smem down to 45,056 B to buy 2 blocks/SM); r29's kd-loop unroll (+2.80%, "the instruction cut r25 judged wall-inert became real at 2 blocks/SM"); r31's q-major sda repack (+1.07%, LDS.64 32→0); r34's quantize-transpose prepass (1364.2 → 1496.8, +9.72%, moving the entire A-side layout transform out of the kernel). The common skeleton of all four steps: **raw-byte weight streaming, low smem for high residency, index arithmetic kept out of the loop, A side preprocessed by a prepass**. r38's job was to prove this skeleton ports wholesale onto q6_K, whose layout is completely different.

Why "a different kernel" instead of tuning `mmq_nt<7,2>` further? The 6.38× per-GMAC gap is structural: the generic kernel runs every type through the same raw-byte-stream processing on the B side, and q6_K's cross-plane (ql+qh) bitfield reassembly cannot be pushed into that framework without per-element in-loop ALU; the ceiling of parameter tuning is far below moving the reassembly into staging once and for all. r37's attribution table prescribed exactly that: "different kernel".

Why was q6_K so slow? At that point it ran the pre-r28 generic int8-GEMM kernel `mmq_nt_kernel<7,2,0>`. That kernel copes reasonably with types like q4_K — contiguous nibbles, sparse scales — but q6_K's packing is hostile to a hot loop: a weight value has to be reassembled across two planes, `ql` (low 4 bits) and `qh` (high 2 bits), and a scale arrives every 16 elements — all of the staging phase's per-element bitfield work sits inside the hot loop. r37's conclusion was "the next lever is a different kernel": give q6_K a dedicated mma kernel modeled on the r28/r34 shape that worked (raw-byte streaming + an expanded B + the BT A-side shell) and drive the reassembly arithmetic out of the hot loop.

That is r38. First fix a layout misreading, then build the kernel, and finally take an occupancy beating.

## 2. Principle — the GPU mechanism

### 2.1 The real q6_K layout (this doc's core correction)

Start from the definition in `src/block.rs` on the current tree (GGUF/ggml convention: 210 B per block, 256 elements):

```rust
// src/block.rs — Q6_K — 6-bit super-block quantization, 256 elements
// 16 blocks of 16 elements each, effectively 6.5625 bits per weight
#[repr(C)]
pub struct BlockQ6_K {
    pub ql: [u8; 128],    // quants, lower 4 bits (QK_K/2)
    pub qh: [u8; 64],     // quants, upper 2 bits (QK_K/4)
    pub scales: [i8; 16], // scales, quantized with 8 bits (QK_K/16)
    pub d: Fp16,          // super-block scale
}
// size assertion: 2 + 16 + 128 + 64 = 210 B (Q6KB = 210)
```

Three key facts:

- **The sub-block granularity is 16, not 32.** `scales[16]` holds 16 int8 scales, one per 16 elements; `d` is the f16 common factor of the whole 256-element super-block. The dequantized value is `d · sc[sub] · (q − 32)`, with q the 6-bit code 0..63 (so the expanded value is centered on −32..31).
- **An element's low 4 bits live in `ql`, its high 2 bits in `qh`**, and the two planes interleave under different bitfield rules (see the `expand_q6_elem` closed form in §3.2).
- **The kernel's loop unit is still the 32-element chunk** (8 per super-block — the task language's QI6_K=8 count: QK_K/32 = 8; llama.cpp's own `QI6_K` macro normalizes the other way — 32 iterations × 8 elements each, the same 8×32 grid). The pre-r38 working model, "q6_K = 8 32-element sub-blocks" (isomorphic to q4_K's 8×32), is **wrong**: one 32-element chunk spans **two 16-element sub-blocks with different scales**, `sc[2c]` and `sc[2c+1]`.

This correction directly determines the mma structure: `mma.m16n8k32` consumes 32 k at a time, but its integer accumulator has no "scale every 16 k" breakpoint; a single scale would multiply one sub-block's contribution by the wrong factor. Hence **KSPLIT=2** (a new concept in this doc: split one 32-k chunk into two `mma.m16n8k16`, k=0..15 and k=16..31, each rescaled by its own 16-element sub-block's scale).

Walk one concrete element through `expand_q6_elem` (§3.2) to pin down the cross-plane interleaving — take **elem = 70** (super-block 0, the 4th 16-element sub-block, the 6th element of the 2nd 32-chunk):

| Quantity | Expression | Value | Meaning |
|---|---|---|---|
| `m` | `70 & 31` | 6 | offset within the 32-chunk |
| `it` | `70 >> 7` | 0 | first 128-element half |
| `n` | `70 & 127` | 70 | offset within the half |
| ql position | `it·64 + (n&63)` = 6, shift `(70>>6)·4` = 4 | high nibble of `ql[6]` | low 4 bits |
| qh position | `it·32 + m` = 6, shift `((70>>5)&3)·2` = 4 | bits 4..5 of `qh[6]` | high 2 bits |
| sub-block scale | `elem/16` = 4 | `sc[4]` | direct evidence of 16-element sub-block granularity |

Note that the ql index follows `n` (offset within the half) while the qh index follows `m` (offset within the chunk) — two interleavings with different strides are exactly the arithmetic reason q6_K cannot be treated as 8×32.

### 2.2 Rescale arithmetic and a rounding trap

q4_K's rescale has two terms (`d` and `dmin` both multiply the accumulator); q6_K has no dmin, so a **single-term rescale** `sum += da·dsc` suffices, where `da` is the A-side per-token-block quantization scale (the packed `d|ssum` word from the r34 prepass) and `dsc = d · sc[16-sub-block]` is the B-side per-16-sub-block scale.

One easy numerical trap: fusing the two 16-sub-block integer accumulators **before** multiplying by the scale (the fused form) is not the same rounding sequence as **two independent `+=`**. Measured: the fused form deviates 1.2e-3 from the CPU reference, just past the 1e-3 parity gate; two independent `+=` pass. Integer accumulators `clow`/`chigh` kept separate, f32 accumulation kept separate — that is the form that passes the gate.

### 2.3 Expanded B and the smem budget

New concept, **expanded-B (the expanded B plane)**: during staging, each super-block's ql+qh bitfield reassembly is computed ahead of time and written into smem as a one-byte-per-element plane of centered int8 (−32..31), so in the hot loop a B fragment is a plain int8 read — ql/qh reassembly, `−32` centering, and bitfield shifts all leave the hot loop (the q6_K version of the "index arithmetic stays out of the loop" lesson verified over and over in r21/r22/r31). The cost is smem: the raw nibble stream is 4 bits/element, the expanded int8 plane is 8 bits/element.

New concept, **KDR** (k-decode rate): how many 32-element chunks each kt iteration stages. KDR=8 = stage an entire 256-element super-block in one go; KDR=4 = half of one. Both the row width of the B expanded plane and every smem plane scale linearly with KDR.

The kernel tile is MMQ_NBI=64 tokens × MMQ_NBJ=128 od rows, 256 threads (8 warps, 16 od rows per warp). The launcher's smem arithmetic (r38 commit text verbatim):

```
smem = KDR * MMQ_NBI * 32   // qa8      (A's swizzled q8 plane)
     + KDR * MMQ_NBI * 4    // sda_q    (A, one packed d|ssum word per token)
     + MMQ_NBJ * KDR * 32   // qb_exp   (B expanded centered-int8 plane)
     + KDR * MMQ_NBJ * 8    // sds      (per chunk×row float2 dsc0/dsc1)
```

- **KDR=4**: 8192 + 1024 + 16384 + 4096 = **29,696 B** → 2 blocks/SM;
- **KDR=8**: every term doubles = **59,392 B** → 1 block/SM.

This is the occupancy cliff. Spread the warps out: 256 threads = 8 warps per block. GB10's per-SM warp ceiling is 48 (derivable from r40's measurements: 3 blocks × 8 = 24 theoretical warps correspond to 18.12 achieved warps/SM = 37.74%, and 18.12/0.3774 = 48). So:

- **KDR=8 (59,392 B) → 1 block/SM = 8 warps/SM = 16.7% theoretical occupancy**;
- **KDR=4 (29,696 B) → 2 blocks/SM = 16 warps/SM = 33.3%**.

This is exactly the q4_K line's pre-r28 disease (before r28 the wide kernel ran 1 block/SM; r28 bought 2 blocks with 45,056 B). **The depth-vs-occupancy trade-off (r5's lesson) re-appears on every new kernel**: KDR=8, staging a whole super-block per iteration, looks like it saves iteration overhead but actually pushes the whole kernel back to latency-bound — r38 paid to reconfirm this lesson with a same-day A/B.

### 2.4 The mma shape in summary

Per kd (one 32-chunk) per warp:

- A side: 4 groups of `ldmatrix.sync.aligned.m8n8.x4` (g=0..3, covering the 4 16-token sub-tiles of the 64 tokens); the A plane is the q8 pre-transposed by the r34 prepass — identical to the r34 BT shell, weight-type agnostic;
- B side: 2 n half-tiles (nh=0/1, 8 rows each) × 2 k half-tiles (`b[nh][0]` = k 0..15 → 16-sub-block 2c, `b[nh][1]` = k 16..31 → 16-sub-block 2c+1);
- mma: 4 m sub-tiles × 2 n half-tiles × 2 k half-tiles = 16 `mma.m16n8k16`, integer accumulators `clow[4][2][4]` / `chigh[4][2][4]`;
- epilogue: two independent `+=`, `acc += da·dsc0·clow` and `acc += da·dsc1·chigh`.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

1. **Fix the layout model first, then draw the mma structure.** In task order, kernel design started directly after r37's attribution named q6_K; but the true first step was checking `block_q6_K`'s sub-block granularity against the CPU reference — the existence of `sc[16]` alone kills the "8×32, single-scale" m16n8k32 plan and forces KSPLIT=2. Discovering this after building on the wrong model would mean reworking the entire fragment layout.
2. **Expand B rather than feed ldmatrix raw bits.** The q4_K NB kernel (r28) does raw-nibble smem plus small in-loop expansion; q6_K's bitfield reassembly (two planes, two bit widths) is far more expensive than q4_K's nibble split — in the hot loop that is 5+ ALU ops per element. Expanded to centered int8, the B-side hot loop is a straight read; the cost (smem doubling) is absorbed by choosing KDR=4.
3. **Reuse the r34 BT shell verbatim on the A side.** A is the activation side, weight-type agnostic: the prepass pre-transposes qa8/sda, and the kernel does bulk uint4 LDG→STS. This concentrates the new kernel's delta on the B side, minimizing the risk surface.
4. **KDR=4, not 8.** See the arithmetic in §2.3: 59,392 B at 1 block/SM is a measured regression (1097.8 tok/s); 29,696 B at 2 blocks/SM is what sustains the occupancy r28 bought.
5. **Entry gate and clean fallback.** The new kernel runs only when every row's `id` is a multiple of 256 (`(id/32) % 8 == 0`, i.e. whole-super-block aligned); if any launcher cap/parameter check fails it returns 0 and the caller falls back cleanly to the generic `mmq_nt<7,2>` — the GPU-safety rule "capability-gate failure takes an explicit fallback, never a silent downgrade" realized as the fallback path in the type-dispatch layer. Both `block_stride` forms are supported: the raw 210 B weight stream and 7e②'s 224 B padded repack (parity verified on both sides).

### 3.2 Key code

**The element-expansion closed form** (r38 commit `75aabb9`, added to `src/cuda_kernels.cu`; the current tree still carries the same function body). `elem` is the element number 0..255 within a super-block; `ql`/`qh` point at in-block offsets 0 and 128:

```cuda
__device__ __forceinline__ int expand_q6_elem(const uint8_t* ql, const uint8_t* qh, int elem) {
    int m  = elem & 31;                   // offset within the 32-chunk
    int it = elem >> 7;                   // 0/1: first/second 128-element half
    int n  = elem & 127;                  // offset within the half
    int ql_idx   = it * 64 + (n & 63);
    int ql_shift = (n >> 6) * 4;          // 0 or 4 (low/high nibble)
    int qh_idx   = it * 32 + m;
    int qh_shift = ((n >> 5) & 3) * 2;    // 0,2,4,6 (2-bit fields)
    int v = ((ql[ql_idx] >> ql_shift) & 0x0F)
          | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
    return v - 32;                        // centered to -32..31
}
```

Segment by segment: the low 4 bits come from the low/high nibble of `ql[it*64 + (n&63)]` (bit 6 of `n` decides which), the high 2 bits from a 2-bit field of `qh[it*32 + m]` (bits 5..4 of `n` pick which 2-bit field) — note that `qh`'s index follows `m` (offset within the chunk) while `ql`'s follows `n` (offset within the half): that is exactly where the "16-element sub-blocks, ql/qh interleaved at different strides" structure comes from. The `−32` centering puts the expanded values directly in the mma-friendly symmetric range.

**The B-expansion + dsc staging macro** (same commit; called once per kt iteration). The expansion part:

```cuda
/* ---- B: expand KDR*32-chunk super-block (half at KDR=4) ---- */
const int sb    = ((kt) * KDR) >> 3;          // super-block number this kt covers
const int cbase = ((kt) * KDR) & 7;           // chunk offset within that sb
for (int x = threadIdx.x; x < MMQ_NBJ * (KDR * 32); x += blockDim.x) {
    const int jj  = x / (KDR * 32), bec = x % (KDR * 32);
    const int j   = j0 + jj;                  // od row
    const int elem = cbase * 32 + bec;        // element number within the super-block
    int v = 0;
    if (j < od && sb < nsb) {
        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
            + (size_t)sb * bstride;
        v = expand_q6_elem(blk, blk + 128, elem);
    }
    qb_exp[(size_t)jj * (KDR * 32) + bec] = (uint8_t)v;
}
```

The dsc (= d·sc) pair — a new concept in this doc, **dsc**: one `float2(dsc0, dsc1)` per (chunk, row); they are the rescale factors of the two 16-sub-blocks of that 32-chunk:

```cuda
const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
    + (size_t)(c >> 3) * bstride;
const float d  = h2f(*(const uint16_t*)(blk + 208));   // f16 super-block factor (offset 208)
const int s0   = 2 * (c & 7);                          // 16-sub-block pair index
dsc0 = d * (float)(int8_t)blk[192 + s0];               // sc[16] starts at offset 192
dsc1 = d * (float)(int8_t)blk[192 + s0 + 1];
```

The two offsets `192/208` are the layout table from §2.1: `scales` occupies 192..207 and `d` occupies 208..209. At KDR=4 one kt touches only half of a super-block (cbase 0..3 or 4..7), hence the "half at KDR=4" comment.

**The A-side bulk copy** (the r34 BT shell; introduced in r38 and kept unchanged by r39 — the excerpt below is the untouched A portion of the r39 diff, word-identical to its r38 introduction): the A plane is the qa8/sda pre-transposed by the prepass, and staging is pure uint4 copying with zero arithmetic:

```cuda
/* ---- A: bulk LDG->STS of the pre-transposed qa8/sda (no math) ---- */
const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ;
for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16; off += blockDim.x)
    ((uint4*)(qa8))[off] = ((const uint4*)(qa8g + qbase))[off];
const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ;
for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16; off += blockDim.x)
    ((uint4*)(sda_q))[off] = ((const uint4*)(sdag + sbase))[off];
```

The per-token packed `d|ssum` (`sda_q`, one uint32) moves together with the A plane — the hot loop's rescale needs only that word plus the B-side dsc, and never touches raw activations again. The A side is of one piece with r34's q4_K BT kernel, which is exactly where "the new kernel's delta lives on the B side" lands.

**The hot loop's B-fragment read** (the entire payoff of expansion — no bitfield ops, no ldmatrix, two int reads):

```cuda
// B-frag: straight int8 read from the expanded plane. Each mma.k16
// uses b[nh][0] (k=0..15, sub 2c) or b[nh][1] (k=16..31, sub 2c+1).
#pragma unroll
for (int nh = 0; nh < 2; nh++) {
    const int jj = j0w + nh * 8 + (lane >> 2);
    const uint8_t* qs = qb_exp + (size_t)jj * (KDR * 32) + (size_t)kd * 32;
    b[nh][0] = *(const int*)(qs + (lane & 3) * 4);
    b[nh][1] = *(const int*)(qs + 16 + (lane & 3) * 4);
}
```

The A side takes its fragments with 4 groups of `ldmatrix.sync.aligned.m8n8.x4` from the swizzled qa8 plane (byte-identical in origin to the r34 BT kernel), after which 16 `mma.m16n8k16` fill the two integer accumulator sets `clow`/`chigh`.

**The launcher's smem arithmetic and fallback** (r38 commit, `launch_mmq_raw_nb_bt_q6k_nt`):

```cuda
constexpr int KDR = 4;
const int smem = KDR * MMQ_NBI * 32   // qa8
               + KDR * MMQ_NBI * 4    // sda_q (one uint32 per token)
               + MMQ_NBJ * KDR * 32   // qb_exp (centered int8, half super-block)
               + KDR * MMQ_NBJ * 8;   // sds (float2 dsc pair = 8B)
dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ);
cudaFuncSetAttribute(&mmq_raw_nb_bt_q6k_kernel<KDR>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
...
mmq_raw_nb_bt_q6k_kernel<KDR><<<grid, 256, smem, stream>>>(...);
// any cudaFuncSetAttribute/launch failure: return 0 -> the caller falls back cleanly to mmq_nt<7,2>
```

At commit time the dispatch gate was `type_id == 7 && MINFER_MMQ_Q6K_NB=1 && (id/32)%8 == 0` (opt-in). The same spot on the current tree (`src/cuda.rs`, around lines 3170-3238) shows the gate's evolved form: r60 flipped `MINFER_MMQ_Q6K_NB`/`MINFER_MMQ_RAW` to default-on (opt out with "0"), and r53/r56 added `w_exp`/`w_dsc` pre-expanded-plane pointers to the launcher (null on miss, falling back to this doc's in-kernel expansion path) — this doc only concerns r38's null/no-plane form. The gate skeleton follows (current-tree code, comments mark the r38 difference):

```rust
if type_id == 7                              // Q6_K
    && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")   // r38: opt-in, requires =1
    && Self::mmq_gate_on("MINFER_MMQ_RAW")      // r38: opt-in, requires =1
    && (id / 32) % 8 == 0                       // whole-super-block aligned
{
    let nchunk = (id / 32) as i32;
    ... // launcher returning 0 falls back cleanly to the generic mmq_nt<7,2>
}
```

### 3.3 Pitfalls

1. **The wrong layout model is the costliest trap.** The "q6_K = 8 32-element sub-blocks" model explains a lot of phenomena convincingly (8 chunks, 210 B, 256 elements all check out) — only `scales[16]`'s 16 does not. An m16n8k32 single-scale plan designed on the wrong model blows up parity at ~1e0 magnitude; checking the layout element by element against the CPU reference is what exposed the 16×16 truth.
2. **Fused rescale rounding overshoots the gate.** If the two 16-sub-block contributions are fused into one expression before multiplying by the scale, the deviation from the CPU reference is 1.2e-3, just past the 1e-3 gate; splitting into two independent `+=` is clean. Lesson: the association order of integer accumulators and f32 accumulation is part of the parity contract.
3. **The KDR=8 smem cliff.** Whole-super-block staging (59,392 B) is not a "more depth, fewer iterations" micro-optimization — it drops 2 blocks/SM straight back to 1 block/SM, and whole-prefill collapses from the ~1518–1533 band to 1097.8 tok/s. The ghost of the r7–r8 era ("wide KD=8 first measured 2124 = phantom (silent smem-cap failure)") returned from the other direction: this time there was no silent failure — the launch legally succeeded and performance collapsed. A legal cap ≠ a cap worth using.
4. **Unit-test the expansion mapping before integration.** The element→super-block mapping (the index arithmetic of `expand_q6_elem`) ran as a standalone verifier with 0 mismatches before integration (a verbatim application of r28's "validate layout maps standalone before integration" lesson).

## 4. Verification

- **Layout verifier**: the element→super-block mapping of `expand_q6_elem` ran standalone over all elements, 0 mismatches (defends against the kind of structural error in §3.3#1 sneaking into the kernel).
- **ptxas resource audit**: KDR=4 compiles to 85 regs / 0 spill (defends against accidental register pressure breaking the 2 blocks/SM occupancy premise).
- **Parity gate**: logits deviation vs the baseline path below the 1e-3 scale, run once for each of the raw 210 B and padded 224 B `block_stride` forms (defends against numerical regressions in bitfield reassembly/rescaling, and against one of the two weight byte streams going untested — the two layouts differ in offset arithmetic: `bstride` only changes the row pitch and the in-block 192/208 offsets are shared, but read-side overrun behavior differs, so both must pass).
- **greedy-32 byte-identical**: the greedy 32-token output stream matches the pre-change binary byte for byte (defends against "parity numbers pass but argmax flips on a knife edge").
- **Interleaved A/B measurement**: same window, same binary, 3/3 pairs positive (defends against fake deltas manufactured by machine-state drift).

## 5. Results

| Metric | before → after | Note |
|---|---|---|
| whole-prefill (7B, same-window A/B median) | 1518.4 → 1561.9 tok/s (**+2.87%**, 3/3) | master row 52 |
| matched-nt q6_K GEMM | 368.9 → 221.8 µs/GMAC (**1.66×**) | below the ≥2× project bar |
| KDR=8 variant | regressed to 1097.8 tok/s | 59,392 B → 1 block/SM, vetoed |
| ptxas | 85 regs / 0 spill (KDR=4) | occupancy premise holds |
| vs llama.cpp (3325-eq anchor) | 2.13× (r37 was 2.15×) | the relative value falls as the engine itself gets faster |

Three readings:

1. **1.66× < 2× landed anyway**, because the ≥2× bar set at project time meant "same league as the q4_K MMQ line", and the old q6_K path (368.9 µs/GMAC, 6.38×/GMAC) was rotten enough that even 1.66× cut the unit cost by nearly 40%. A strictly-positive change has no reason to stay outside the gate. The master table records this as "LANDED, < 2x".
2. **The occupancy lever is decisive**: the same kernel, KDR=8 → 1097.8 (1 block/SM), KDR=4 → +2.87% (2 blocks/SM). The difference between shallow and deep staging is not tuning; it is the difference between 16.7% and 33.3% theoretical occupancy.
3. **The attn_v kernel is still latency-bound (compute only 16.7%)** — the hook this doc leaves behind: the kernel got faster but is still waiting; r39's pipelining and r41's load width both start from this number.

**q6_K line postscript** (this doc is the first step of a four-step arc; all numbers from master table rows 52-55):

| Step | q6_K kernel | whole-prefill |
|---|---|---|
| r37 baseline | 368.9 µs/GMAC (6.38×/GMAC) | 1521 tok/s |
| **r38 (this doc)** | 221.8 µs/GMAC (1.66×) | 1561.9 (+2.87%) |
| r39 KDR=2 double buffer | attn_v −19.7% | 1777.5 (+13.3%) |
| r40 third resident block | kernel −23% | 2015.6 (+13.0%) |
| r41 B-expand uint4 widen | 0.654 ms (−61.5%) | 2605.2 (+30.7%) |

This doc's 1.66× looks like the smallest win, but it established the kernel skeleton the next three steps share — without that skeleton, pipelining, residency, and load width have nowhere to land.

## 6. Lessons

1. **Check the layout against the CPU reference before designing the mma structure** — a single field, `scales[16]`, vetoed the entire 8×32 plan; a layout error reworked at the fragment-layout layer costs everything.
2. **Depth vs occupancy goes up for auction again on every new kernel** (r5's lesson, Nth rerun): for a parameter like KDR — "how many k to stage per iteration" — compute smem → blocks/SM first and decide from that, never from intuition.
3. **Land strictly-positive changes even below the project bar**: the bar is a guess made at project time; the measured cost of the alternative is the fact. "LANDED, < 2x" is a perfectly legitimate state.
4. **The verifier's place is before integration**: a 0-mismatch standalone mapping verifier institutionalizes r28's lesson — a layout bug costs one minute in the verifier, an afternoon in the parity matrix.

---
← 40 · [Index](./README.md) · [42 · r39 q6_K KDR=2 double-buffer](42-r39-q6k-kdr2-double-buffer.md) →
