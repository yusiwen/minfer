"""Verify GQA attention output for first token (nkv=1 → output = V).

For single-token KV cache, softmax of 1 element = 1.0, so
attention output o[head] = V[head // gqa] (repeated across GQA group).

Usage:
    uv run python -m scripts.verify_attention --model <model> \\
      --compare /tmp/minfer_dump_layer0_ba.f32 --n-head 14 --n-kv 2 --n-dims 64
"""

import argparse
import struct
import numpy as np
from gguf import GGUFReader, GGML_QUANT_SIZES
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize


def read_tensor_bytes(model_path, data_offset, nbytes):
    with open(model_path, 'rb') as f:
        f.seek(data_offset)
        return f.read(nbytes)


def main():
    parser = argparse.ArgumentParser(description="Verify GQA attention output")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--compare", default="/tmp/minfer_dump_layer0_ba.f32",
                        help="Path to ba dump file")
    parser.add_argument("--n-head", type=int, default=14)
    parser.add_argument("--n-kv", type=int, default=2)
    parser.add_argument("--n-dims", type=int, default=64)
    parser.add_argument("--token-id", type=int, default=151644)
    args = parser.parse_args()

    nh = args.n_head
    nk = args.n_kv
    hd = args.n_dims
    gqa = nh // nk
    nkt = nk * hd

    reader = GGUFReader(args.model)

    # Compute attention output = V for single token
    # Read embedding → RMSNorm → WV matmul → add bias → RoPE → V
    embd_off = None; embd_type = None; hidden_dim = 896
    for t in reader.tensors:
        if 'token_embd' in t.name:
            embd_off = t.data_offset; embd_type = t.tensor_type; hidden_dim = t.shape[0]; break

    qt = GGMLQuantizationType(embd_type); bs, ts = GGML_QUANT_SIZES[qt]
    nbp = hidden_dim // 32; rbytes = nbp * ts

    with open(args.model, 'rb') as f:
        f.seek(embd_off + args.token_id * rbytes)
        emb = dequantize(np.frombuffer(f.read(rbytes), dtype=np.uint8), qt)

    # Read attn_norm + WV weight
    norm_data = wv_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_norm.weight':
            with open(args.model, 'rb') as f: f.seek(t.data_offset); norm_data = np.frombuffer(f.read(hidden_dim*4), dtype=np.float32)
        if t.name == 'blk.0.attn_v.weight': wv_tensor = t

    scale = 1.0 / np.sqrt(np.mean(emb**2) + 1e-6)
    act = emb * scale * norm_data.astype(np.float64)

    wvt = GGMLQuantizationType(wv_tensor.tensor_type); wbs, wts = GGML_QUANT_SIZES[wvt]
    wvnb = int(wv_tensor.shape[0]) // wbs; wvrow = wvnb * wts
    with open(args.model, 'rb') as f:
        f.seek(wv_tensor.data_offset); wvraw = f.read(wvrow * nkt)

    v = np.zeros(nkt, dtype=np.float64)
    for o in range(nkt):
        w = dequantize(np.frombuffer(wvraw[o*wvrow:o*wvrow+wvrow], dtype=np.uint8), wvt)
        v[o] = np.dot(w, act)

    # Add bias if present
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.bias':
            with open(args.model, 'rb') as f: f.seek(t.data_offset); vb = np.frombuffer(f.read(nkt*4), dtype=np.float32)
            v += vb.astype(np.float64)

    # Apply RoPE to V (same as BK for this token)
    freq_base = 1000000.0
    for f in reader.fields.values():
        if f.name == 'qwen2.rope.freq_base':
            freq_base = struct.unpack('<f', f.parts[-1].tobytes())[0]; break

    p = 0  # first token position
    v_rope = v.copy()
    for hk in range(nk):
        base = hk * hd; half = hd // 2
        for i in range(half):
            freq = 1.0 / np.power(freq_base, (2.0*i)/hd)
            theta = p * freq
            cs, sn = np.cos(theta), np.sin(theta)
            x0 = v[base + i]; x1 = v[base + i + half]
            v_rope[base + i] = x0 * cs - x1 * sn
            v_rope[base + i + half] = x0 * sn + x1 * cs

    # For single token: attention output = V repeated across GQA heads
    ref = np.zeros(nh * hd, dtype=np.float64)
    for h in range(nh):
        hk = h // gqa
        src = hk * hd
        dst = h * hd
        ref[dst:dst+hd] = v_rope[src:src+hd]

    mf = np.fromfile(args.compare, dtype=np.float32).astype(np.float64)
    mf_token0 = mf[:nh*hd]

    cos = float(np.dot(ref, mf_token0) / (np.linalg.norm(ref)*np.linalg.norm(mf_token0)+1e-30))
    print(f"GQA attention (token 0, nkv=1): cos={cos:.10f}  {'✓' if cos>0.9999 else '✗ MISMATCH'}")
    print(f"Ref RMS={np.sqrt(np.mean(ref**2)):.4f}  minfer RMS={np.sqrt(np.mean(mf_token0**2)):.4f}")
    for h in range(min(4, nh)):
        b = h * hd
        hcos = float(np.dot(ref[b:b+hd], mf_token0[b:b+hd])/(np.linalg.norm(ref[b:b+hd])*np.linalg.norm(mf_token0[b:b+hd])+1e-30))
        print(f"  head {h}: cos={hcos:.10f}")


if __name__ == "__main__":
    main()
