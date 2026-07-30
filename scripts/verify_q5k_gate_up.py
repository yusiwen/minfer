#!/usr/bin/env python3
"""Verify minfer's Q5_1 gate/up projections + SwiGLU + Q5_K ffn_down.

Results (2025-07-30):
  - gate/up/swiglu: cos=1.00000002 vs minfer ✓ (internally consistent)
  - ffn_down: cos=1.00000008 vs minfer ✓ (internally consistent)
  - llama ffn_norm → gate: cos=0.99995 (nearly identical, ffn_norm matches)
  - BUT llama ffn_out norm=787 vs minfer fd=1170 (1.49× gap)

Root cause: ffn_norm element values differ slightly between llama and minfer
(e.g., ffn_norm[0]: llama=0.40978 vs minfer=0.40121) despite overall cos≥0.999.
This per-element ~2% difference compounds through Q5_1 gate/up → SwiGLU → Q5_K
ffn_down, amplified by the Q5_K weight matrix, producing the 1.49× fd norm gap.

The per-element ffn_norm difference traces back to the attn_out (cos=0.99975
but individual elements differ by ~2%). This is likely from Q5_1 attention
projections (Q/K/V/O) using Q8_0 quantization, where the rounding behavior
(Rust .round() vs C roundf()) introduces ~1ULP per-element error that
accumulates through 24 attention heads and 4 projection matrices.
"""

import numpy as np
import struct
import os
import sys

GGUF_PATH = os.path.expanduser(
    "~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
    "qwen2.5-0.5b-instruct-q5_k_m.gguf"
)

DUMP_DIR = "/tmp/q5km_l2b"  # minfer dump directory

# ---- fp16 helpers ----
def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1f
    mant = h & 0x3ff
    if exp == 0:
        return (-1)**sign * 2**(-14) * (mant / 1024.0)
    return (-1)**sign * 2**(exp - 15) * (1.0 + mant / 1024.0)

# ---- GGUF parser ----
def parse_gguf_tensor(data, tensor_name):
    """Find tensor by name and return (ne, type, abs_offset)."""
    idx = data.find(tensor_name.encode())
    if idx < 0:
        return None
    nl = struct.unpack('<Q', data[idx - 8:idx])[0]
    p = idx + nl
    nd = struct.unpack('<I', data[p:p + 4])[0]
    p += 4
    ne = [struct.unpack('<Q', data[p + i * 8:p + i * 8 + 8])[0] for i in range(nd)]
    p += nd * 8
    tt = struct.unpack('<I', data[p:p + 4])[0]
    p += 4
    toff = struct.unpack('<Q', data[p:p + 8])[0]
    return ne, tt, toff


def get_data_section_start(data):
    """Parse GGUF header to find data section start."""
    p = 0
    magic = struct.unpack('<I', data[p:p + 4])[0]
    p += 4  # version
    version = struct.unpack('<I', data[p:p + 4])[0]
    p += 4
    tc = struct.unpack('<Q', data[p:p + 8])[0]
    p += 8
    kc = struct.unpack('<Q', data[p:p + 8])[0]
    p += 8

    sm = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8, 7: 1}
    for _ in range(kc):
        kl = struct.unpack('<Q', data[p:p + 8])[0]
        p += 8
        p += kl
        vt = struct.unpack('<I', data[p:p + 4])[0]
        p += 4
        if vt == 8:
            sl = struct.unpack('<Q', data[p:p + 8])[0]
            p += 8 + sl
        elif vt == 9:
            at = struct.unpack('<I', data[p:p + 4])[0]
            p += 4
            an = struct.unpack('<Q', data[p:p + 8])[0]
            p += 8
            if at == 8:
                for _ in range(an):
                    sl = struct.unpack('<Q', data[p:p + 8])[0]
                    p += 8 + sl
            else:
                p += an * sm.get(at, 4)
        else:
            p += sm.get(vt, 4)

    align = 32
    p = (p + align - 1) // align * align
    for _ in range(tc):
        nl = struct.unpack('<Q', data[p:p + 8])[0]
        p += 8
        p += nl
        nd = struct.unpack('<I', data[p:p + 4])[0]
        p += 4
        p += nd * 8
        p += 4
        p += 8
    return (p + align - 1) // align * align


# ---- Q8_0 quantization (matching minfer's quantize_row_q8_0_buf) ----
def quantize_q8_0(x):
    """Quantize f32 vector to Q8_0 bytes. Returns (q8_bytes, d_q8_list)."""
    nb = len(x) // 32
    q8 = bytearray(nb * 34)
    d_list = []
    for b in range(nb):
        xx = x[b * 32:(b + 1) * 32]
        amax = float(np.max(np.abs(xx)))
        d = amax / 127.0 if amax > 0 else 0.0
        id_val = 1.0 / d if d > 0 else 0.0
        df = np.float16(d)
        bits = int(df.view(np.uint16))
        q8[b * 34] = bits & 0xFF
        q8[b * 34 + 1] = (bits >> 8) & 0xFF
        for j in range(32):
            qi = int(np.clip(np.round(xx[j] * id_val), -128, 127))
            q8[b * 34 + 2 + j] = qi & 0xFF
        d_list.append(d)
    return bytes(q8), d_list


# ---- Q5_1 × Q8_0 dot product (matching minfer's dot_q5_1_q8_0_scalar) ----
def dot_q5_1_q8_0(q5_row, q8_row):
    """Compute dot product of Q5_1 weight row with Q8_0 activation.
    q5_row: bytes, nb_blocks * 24 bytes
    q8_row: bytes, nb_blocks * 34 bytes
    Returns float dot product.
    Weight formula: val = d_q5 * unsigned_5bit + m_q5
    """
    nb = len(q5_row) // 24
    result = 0.0
    for b in range(nb):
        q5b = q5_row[b * 24:(b + 1) * 24]
        q8b = q8_row[b * 34:(b + 1) * 34]
        d_q5 = f16_to_f32(struct.unpack('<H', q5b[0:2])[0])
        m_q5 = f16_to_f32(struct.unpack('<H', q5b[2:4])[0])
        d_q8 = f16_to_f32(struct.unpack('<H', q8b[0:2])[0])
        qh = struct.unpack('<I', q5b[4:8])[0]
        qs = q5b[8:24]
        q8qs = q8b[2:34]

        sum_sub = 0
        sum_q8 = 0
        for j in range(16):
            u_lo = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4)
            u_hi = ((qs[j] >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4)
            q8_lo = int.from_bytes(q8qs[j:j + 1], 'little', signed=True)
            q8_hi = int.from_bytes(q8qs[j + 16:j + 17], 'little', signed=True)
            sum_sub += u_lo * q8_lo + u_hi * q8_hi
            sum_q8 += q8_lo + q8_hi
        # Q5_1: weight = d * unsigned + m
        result += d_q8 * (d_q5 * sum_sub + m_q5 * sum_q8)
    return result


# ---- Main ----
def main():
    print("=== Q5_K_M Gate / Up / SwiGLU Verification ===\n")

    # ---- Load GGUF ---- 
    print(f"Loading GGUF: {GGUF_PATH}")
    with open(GGUF_PATH, 'rb') as f:
        data = f.read()

    # Known absolute file offsets (ctx.offset = 5947744)
    gate_abs = 390437216
    up_abs = 393705824

    # Load ffn_norm weight (blk.2.ffn_norm.weight, F32, 896 floats)
    # From GGUF: toff=391116864? Let me find it
    fn_idx = data.find(b'blk.2.ffn_norm.weight')
    if fn_idx < 0:
        print("ERROR: cannot find ffn_norm weight")
        return 1
    nl = struct.unpack('<Q', data[fn_idx - 8:fn_idx])[0]
    p = fn_idx + nl
    nd = struct.unpack('<I', data[p:p + 4])[0]; p += 4
    p += nd * 8  # ne
    p += 4  # type
    toff_fn = struct.unpack('<Q', data[p:p + 8])[0]
    fn_abs = 5947744 + toff_fn
    fn_weight = np.frombuffer(
        data[fn_abs:fn_abs + 896 * 4],
        dtype=np.float32
    ).astype(np.float64)
    print(f"  ffn_norm weight loaded, first 4: {fn_weight[:4]}")

    # Load minfer dumps
    print(f"\nLoading minfer dumps from: {DUMP_DIR}")
    attn_out = np.fromfile(
        f"{DUMP_DIR}/minfer_dump_layer2_attn_out.f32",
        dtype=np.float32).reshape(-1, 896).astype(np.float64)
    mf_bg = np.fromfile(
        f"{DUMP_DIR}/minfer_dump_layer2_bg.f32",
        dtype=np.float32).reshape(-1, 4864).astype(np.float64)
    mf_bf = np.fromfile(
        f"{DUMP_DIR}/minfer_dump_layer2_bf.f32",
        dtype=np.float32).reshape(-1, 4864).astype(np.float64)
    mf_swiglu = np.fromfile(
        f"{DUMP_DIR}/minfer_dump_layer2_swiglu.f32",
        dtype=np.float32).reshape(-1, 4864).astype(np.float64)

    print(f"  attn_out: {attn_out.shape}")
    print(f"  bg (gate): {mf_bg.shape}")
    print(f"  bf (up):   {mf_bf.shape}")
    print(f"  swiglu:    {mf_swiglu.shape}")

    # Compute ffn_norm = RMSNorm(attn_out, ffn_norm_weight)
    eps = 1e-6
    t = 0  # token index
    x = attn_out[t].copy()
    rms = np.sqrt(np.mean(x ** 2) + eps)
    ffn_norm_py = x / rms * fn_weight
    print(f"\n  Token 0 ffn_norm [py]: norm={np.linalg.norm(ffn_norm_py):.4f}")

    # Quantize ffn_norm to Q8_0
    q8_bytes, d_q8_list = quantize_q8_0(ffn_norm_py)
    print(f"  Q8_0 quantization: {len(q8_bytes)} bytes, d_q8[0]={d_q8_list[0]:.6f}")

    # Load gate weight row 0
    idim = 896  # input dim
    odim = 4864  # output dim
    nb_blocks = idim // 32  # 28
    row_bytes = nb_blocks * 24  # 672

    # Compute gate[0] and up[0] (row 0 of each)
    print(f"\n  Computing gate[0] (row 0, 896→1 dot):")
    gate_0 = dot_q5_1_q8_0(data[gate_abs:gate_abs + row_bytes], q8_bytes)
    print(f"    Python: {gate_0:.8f}")
    print(f"    Minfer: {mf_bg[t, 0]:.8f}")
    ratio = gate_0 / mf_bg[t, 0] if mf_bg[t, 0] != 0 else float('inf')
    print(f"    Ratio: {ratio:.10f}")

    up_0 = dot_q5_1_q8_0(data[up_abs:up_abs + row_bytes], q8_bytes)
    print(f"\n  Computing up[0] (row 0, 896→1 dot):")
    print(f"    Python: {up_0:.8f}")
    print(f"    Minfer: {mf_bf[t, 0]:.8f}")
    ratio = up_0 / mf_bf[t, 0] if mf_bf[t, 0] != 0 else float('inf')
    print(f"    Ratio: {ratio:.10f}")

    # Full gate computation (all 4864 output rows)
    print(f"\n  Computing full gate vector (896→4864, {odim} rows)...")
    gate_full = np.zeros(odim, dtype=np.float32)
    up_full = np.zeros(odim, dtype=np.float32)
    for o in range(odim):
        gate_full[o] = dot_q5_1_q8_0(
            data[gate_abs + o * row_bytes:gate_abs + (o + 1) * row_bytes],
            q8_bytes
        )
        up_full[o] = dot_q5_1_q8_0(
            data[up_abs + o * row_bytes:up_abs + (o + 1) * row_bytes],
            q8_bytes
        )

    # Compare with minfer
    cos_g = float(np.dot(gate_full, mf_bg[t]) / (
        np.linalg.norm(gate_full) * np.linalg.norm(mf_bg[t]) + 1e-30
    ))
    cos_u = float(np.dot(up_full, mf_bf[t]) / (
        np.linalg.norm(up_full) * np.linalg.norm(mf_bf[t]) + 1e-30
    ))
    print(f"    Gate cos (py vs minfer): {cos_g:.10f}")
    print(f"    Up   cos (py vs minfer): {cos_u:.10f}")
    print(f"    Gate norm: py={np.linalg.norm(gate_full):.2f}  mf={np.linalg.norm(mf_bg[t]):.2f}")
    print(f"    Up   norm: py={np.linalg.norm(up_full):.2f}  mf={np.linalg.norm(mf_bf[t]):.2f}")

    # SwiGLU: silu(gate) * up
    silu_g = gate_full / (1.0 + np.exp(-gate_full))
    swiglu_py = silu_g * up_full
    cos_s = float(np.dot(swiglu_py, mf_swiglu[t]) / (
        np.linalg.norm(swiglu_py) * np.linalg.norm(mf_swiglu[t]) + 1e-30
    ))
    print(f"\n    SwiGLU cos (py vs minfer): {cos_s:.10f}")
    print(f"    SwiGLU norm: py={np.linalg.norm(swiglu_py):.2f}  mf={np.linalg.norm(mf_swiglu[t]):.2f}")

    # Summary
    print(f"\n=== Summary ===")
    all_ok = cos_g > 0.9999 and cos_u > 0.9999 and cos_s > 0.9999
    print(f"  Gate:      {'✓' if cos_g > 0.9999 else '✗'} (cos={cos_g:.8f})")
    print(f"  Up:        {'✓' if cos_u > 0.9999 else '✗'} (cos={cos_u:.8f})")
    print(f"  SwiGLU:    {'✓' if cos_s > 0.9999 else '✗'} (cos={cos_s:.8f})")
    print(f"  {'ALL PASS' if all_ok else 'DEVIATIONS FOUND'}")

    # ---- Cross-check: compute gate/up from llama's ffn_norm dump ----
    print(f"\n=== Cross-check: use llama's ffn_norm as input ===")
    ll_ffn_norm = np.fromfile(
        "/tmp/llama_layer2_ffn_norm.f32",
        dtype=np.float32).reshape(-1, 896).astype(np.float64)

    # Quantize llama ffn_norm to Q8_0 (same as minfer)
    q8_ll, d_list_ll = quantize_q8_0(ll_ffn_norm[t])
    print(f"  llama ffn_norm Q8_0: d_q8[0]={d_list_ll[0]:.6f} (minfer: {d_q8_list[0]:.6f})")

    # Compute gate using llama ffn_norm + same Q5_1 weights
    gate_from_ll = np.zeros(odim, dtype=np.float32)
    up_from_ll = np.zeros(odim, dtype=np.float32)
    for o in range(odim):
        gate_from_ll[o] = dot_q5_1_q8_0(
            data[gate_abs + o * row_bytes:gate_abs + (o + 1) * row_bytes],
            q8_ll
        )
        up_from_ll[o] = dot_q5_1_q8_0(
            data[up_abs + o * row_bytes:up_abs + (o + 1) * row_bytes],
            q8_ll
        )

    cos_g2 = float(np.dot(gate_full, gate_from_ll) / (
        np.linalg.norm(gate_full) * np.linalg.norm(gate_from_ll) + 1e-30
    ))
    print(f"  Gate cos (minfer_ffn vs llama_ffn as input): {cos_g2:.8f}")
    print(f"  Gate norms: minfer_ffn={np.linalg.norm(gate_full):.2f}  llama_ffn={np.linalg.norm(gate_from_ll):.2f}")
    if cos_g2 < 0.9999:
        print(f"  NOTE: same Q5_1 weights + same Q8_0 quantization, but different ffn_norm input")
        print(f"        → gate outputs differ, which explains swiglu/ffn_out divergence")

    # ---- Verify Q5_K ffn_down ----
    print(f"\n=== Verify Q5_K ffn_down (blk.2.ffn_down.weight) ===")
    idx_fd = data.find(b'blk.2.ffn_down.weight')
    nl = struct.unpack('<Q', data[idx_fd - 8:idx_fd])[0]
    p = idx_fd + nl
    nd = struct.unpack('<I', data[p:p + 4])[0]; p += 4
    p += nd * 8  # ne
    p += 4  # type
    toff_fd = struct.unpack('<Q', data[p:p + 8])[0]
    fd_abs = 5947744 + toff_fd
    print(f"  ffn_down abs_off={fd_abs}")

    # Q5_K: 176 bytes per 256 elements, 19 super-blocks per row (4864/256=19)
    # Dequant swiglu to Q8_0 first (for Q5_K × Q8_0 dot)
    fd_idim = 4864  # input dim for ffn_down
    q8_sw, _ = quantize_q8_0(swiglu_py)  # quantize swiglu

    # Q5_K deinterleave + dot formula
    def dot_q5_k_q8_0(q5_row, q8_row):
        nb_super = len(q5_row) // 176
        result = 0.0
        for s in range(nb_super):
            q5b = q5_row[s*176:(s+1)*176]
            q8b = q8_row[s*8*34:(s+1)*8*34]
            d = f16_to_f32(struct.unpack('<H', q5b[0:2])[0])
            dmin = f16_to_f32(struct.unpack('<H', q5b[2:4])[0])
            sc_arr = q5b[4:16]
            sc = [0]*8; mn = [0]*8
            for j in range(4):
                sc[j] = sc_arr[j] & 0x3F; mn[j] = sc_arr[j+4] & 0x3F
            for j in range(4, 8):
                sc[j] = (sc_arr[j+4] & 0xF) | ((sc_arr[j-4] >> 6) << 4)
                mn[j] = (sc_arr[j+4] >> 4) | ((sc_arr[j] >> 6) << 4)
            qh = q5b[16:48]; qs = q5b[48:176]
            nb2 = [0]*256
            for ci in range(4):
                chunk = qs[ci*32:ci*32+32]
                for l in range(32):
                    nb2[(2*ci)*32+l] = chunk[l] & 0x0F
                    nb2[(2*ci+1)*32+l] = chunk[l] >> 4
            for sub in range(8):
                dl = d * sc[sub]; ml = dmin * mn[sub]
                q8blk = q8b[sub*34:(sub+1)*34]
                d_q8 = f16_to_f32(struct.unpack('<H', q8blk[0:2])[0])
                q8qs = q8blk[2:34]
                sum_sub = 0; sum_q8 = 0
                for k in range(32):
                    hbit = (qh[sub*4 + k//8] >> (k % 8)) & 1
                    uval = nb2[sub*32+k] | (hbit << 4)
                    q8v = int.from_bytes(q8qs[k:k+1], 'little', signed=True)
                    sum_sub += uval * q8v; sum_q8 += q8v
                # Q5_K: weight = dl * (unsigned - 16) - ml
                result += d_q8 * (dl * sum_sub + (-dl * 16 - ml) * sum_q8)
        return result

    # Compute ffn_down output element 0
    row_bytes_fd = 19 * 176  # 3344
    fd_0 = dot_q5_k_q8_0(data[fd_abs:fd_abs + row_bytes_fd], q8_sw)
    mf_fd = np.fromfile(f"{DUMP_DIR}/minfer_dump_layer2_fd.f32", dtype=np.float32).reshape(-1, 896).astype(np.float64)
    print(f"  ffn_down[0]: Python={fd_0:.8f}  Minfer={mf_fd[t, 0]:.8f}")
    if fd_0 != 0:
        print(f"  Ratio: {mf_fd[t, 0]/fd_0:.10f}")

    # Full ffn_down (896 output rows)
    print(f"  Computing full ffn_down (896 rows)...")
    fd_py = np.zeros(896, dtype=np.float32)
    for o in range(896):
        fd_py[o] = dot_q5_k_q8_0(
            data[fd_abs + o * row_bytes_fd:fd_abs + (o + 1) * row_bytes_fd],
            q8_sw
        )
    cos_fd = float(np.dot(fd_py, mf_fd[t]) / (
        np.linalg.norm(fd_py) * np.linalg.norm(mf_fd[t]) + 1e-30
    ))
    print(f"  ffn_down cos (py vs minfer): {cos_fd:.10f}")
    print(f"  ffn_down norm: py={np.linalg.norm(fd_py):.2f}  mf={np.linalg.norm(mf_fd[t]):.2f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
