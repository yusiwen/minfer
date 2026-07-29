"""Verify RMSNorm output for a GGUF model's attention norm layer.

Uses gguf.quants.dequantize() — validated against llama.cpp C reference.

Usage:
    uv run python -m scripts.verify_rmsnorm --model <path.gguf> --layer 0
"""

import argparse
import os
import sys
import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize
from gguf import GGML_QUANT_SIZES


def read_tensor_bytes(model_path, data_offset, nbytes):
    with open(model_path, 'rb') as f:
        f.seek(data_offset)
        return f.read(nbytes)


def main():
    parser = argparse.ArgumentParser(description="Verify RMSNorm output")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index for attn_norm (default: 0)")
    parser.add_argument("--token-id", type=int, default=0,
                        help="Token ID (default: 0 = first vocab entry)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to minfer dump file for cross-validation (e.g. /tmp/minfer_dump_layer0_bn.f32)")
    args = parser.parse_args()

    from gguf import GGUFReader
    reader = GGUFReader(args.model)

    # Find embedding
    embd_off = None
    embd_type = None
    hidden_dim = None
    for t in reader.tensors:
        if 'token_embd' in t.name:
            embd_off = t.data_offset
            embd_type = t.tensor_type
            hidden_dim = t.shape[0]
            break

    if embd_off is None:
        print("ERROR: token_embd.weight not found")
        sys.exit(1)

    # Find attn_norm
    norm_name = f"blk.{args.layer}.attn_norm.weight"
    norm_data = None
    for t in reader.tensors:
        if t.name == norm_name:
            norm_raw = read_tensor_bytes(args.model, t.data_offset, t.shape[0] * 4)
            norm_data = np.frombuffer(norm_raw, dtype=np.float32)
            break

    if norm_data is None:
        print(f"ERROR: {norm_name} not found")
        sys.exit(1)

    hidden_dim = hidden_dim or len(norm_data)
    nb = hidden_dim // 32
    qtype = GGMLQuantizationType(embd_type)
    block_size, type_size = GGML_QUANT_SIZES[qtype]

    print(f"Model: {args.model}")
    print(f"Layer: {args.layer} ({norm_name})")
    print(f"Hidden dim: {hidden_dim}  embd type: {qtype.name}")
    print(f"Token ID: {args.token_id}")
    print()

    # Read one row of embedding, dequantize with verified dequantize()
    row_bytes = nb * type_size
    offset = embd_off + args.token_id * row_bytes
    raw = read_tensor_bytes(args.model, offset, row_bytes)
    emb = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype)

    # RMSNorm
    emb_rms = float(np.sqrt(np.mean(emb ** 2)))
    scale = 1.0 / np.sqrt(float(np.mean(emb ** 2)) + 1e-6)
    bn = emb * scale * norm_data.astype(np.float64)

    bn_rms = float(np.sqrt(np.mean(bn ** 2)))
    print(f"Embedding RMS: {emb_rms:.6f}")
    print(f"RMSNorm scale: {scale:.6f}")
    print(f"RMSNorm output RMS: {bn_rms:.6f}")
    print(f"norm_w[0]={norm_data[0]:.6f}  norm_w[1]={norm_data[1]:.6f}")
    print()
    print("First 8 RMSNorm output values:")
    for i in range(min(8, len(bn))):
        print(f"  bn[{i}] = {bn[i]:+.8e}")

    # Cross-validation against minfer dump
    if args.compare and __import__('os').path.exists(args.compare):
        mf = np.fromfile(args.compare, dtype=np.float32)
        nt = len(mf) // hidden_dim
        m_bn = mf[:hidden_dim].astype(np.float64)  # first token
        cos = float(np.dot(bn, m_bn) / (np.linalg.norm(bn) * np.linalg.norm(m_bn) + 1e-30))
        rms_r = float(np.sqrt(np.mean(m_bn ** 2))) / bn_rms if bn_rms > 0 else 0
        print(f"\nCross-validation ({__import__('os').path.basename(args.compare)}, {nt} tokens):")
        print(f"  cosine = {cos:.10f}  {'✓' if cos > 0.9999 else '✗ MISMATCH'}")
        print(f"  minfer[0] = {m_bn[0]:+.8e}")
        print(f"  python[0] = {bn[0]:+.8e}")


if __name__ == "__main__":
    main()
