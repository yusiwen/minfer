"""Verify matmul output: compute dot product of weight row 0 with activation,
compare against minfer dump.

Uses gguf.quants.dequantize() — validated against llama.cpp C reference.

Usage:
    uv run python -m scripts.verify_matmul --model <model> \\
      --layer 0 --token-id 151644 \\
      --weight blk.0.attn_q.weight \\
      --compare /tmp/minfer_dump_layer0_bq.f32
"""

import argparse
import struct
import sys
import numpy as np
from gguf import GGUFReader, GGML_QUANT_SIZES
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize


def read_tensor_bytes(model_path, data_offset, nbytes):
    with open(model_path, 'rb') as f:
        f.seek(data_offset)
        return f.read(nbytes)


def main():
    parser = argparse.ArgumentParser(description="Verify matmul output")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--token-id", type=int, default=0,
                        help="Token ID for embedding lookup")
    parser.add_argument("--weight", type=str, required=True,
                        help="Weight tensor name (e.g. blk.0.attn_q.weight)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to minfer dump file for cross-validation")
    parser.add_argument("--n-values", type=int, default=32,
                        help="Number of output values to compare")
    args = parser.parse_args()

    reader = GGUFReader(args.model)

    # Read norm weight (attn_norm for Q/K/V matmuls, ffn_norm for FFN)
    norm_name = f"blk.{args.layer}.attn_norm.weight"
    if "ffn" in args.weight:
        norm_name = f"blk.{args.layer}.ffn_norm.weight"
    norm_data = None
    for t in reader.tensors:
        if t.name == norm_name:
            norm_raw = read_tensor_bytes(args.model, t.data_offset, t.shape[0] * 4)
            norm_data = np.frombuffer(norm_raw, dtype=np.float32)
            break
    if norm_data is None:
        print(f"ERROR: {norm_name} not found")
        sys.exit(1)

    # Read embedding, dequantize
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

    hidden_dim = hidden_dim or len(norm_data)
    qtype = GGMLQuantizationType(embd_type)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    nb = hidden_dim // 32

    # Compute activation: RMSNorm(embedding)
    row_bytes = nb * type_size
    offset = embd_off + args.token_id * row_bytes
    raw = read_tensor_bytes(args.model, offset, row_bytes)
    emb = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype)
    scale = 1.0 / np.sqrt(float(np.mean(emb ** 2)) + 1e-6)
    activation = emb * scale * norm_data.astype(np.float64)

    # Read weight tensor
    weight_tensor = None
    for t in reader.tensors:
        if t.name == args.weight:
            weight_tensor = t
            break
    if weight_tensor is None:
        print(f"ERROR: {args.weight} not found")
        sys.exit(1)

    wtype = GGMLQuantizationType(weight_tensor.tensor_type)
    w_block_size, w_type_size = GGML_QUANT_SIZES[wtype]
    inner_dim = int(weight_tensor.shape[0])
    w_nb = inner_dim // w_block_size
    w_row_bytes = w_nb * w_type_size

    print(f"Weight: {args.weight}")
    print(f"  shape={weight_tensor.shape}  type={wtype.name}")
    print(f"  inner_dim={inner_dim}  blocks={w_nb}  row_bytes={w_row_bytes}")
    print(f"Activation: token_id={args.token_id}  dims={len(activation)}")
    print()

    # Check for bias tensor
    bias_suffix = args.weight.replace(".weight", ".bias")
    bias = None
    for t in reader.tensors:
        if t.name == bias_suffix:
            with open(args.model, 'rb') as f:
                f.seek(t.data_offset)
                bias = np.frombuffer(f.read(t.shape[0] * 4), dtype=np.float32)
            break

    # Read all rows up to n_values for comparison
    n_vals = min(args.n_values, inner_dim)
    w_all_raw = read_tensor_bytes(args.model, weight_tensor.data_offset,
                                   w_row_bytes * n_vals)
    ref = np.zeros(n_vals, dtype=np.float64)
    for o in range(n_vals):
        off = o * w_row_bytes
        w_row = dequantize(np.frombuffer(w_all_raw[off:off + w_row_bytes], dtype=np.uint8), wtype)
        ref[o] = np.dot(w_row, activation)
        if bias is not None:
            ref[o] += bias[o]

    print(f"Python reference (first {n_vals} values):")
    for i in range(min(8, n_vals)):
        print(f"  [{i}] {ref[i]:+.8e}")
    print(f"  RMS={np.sqrt(np.mean(ref**2)):.6f}")

    if args.compare:
        mf = np.fromfile(args.compare, dtype=np.float32).astype(np.float64)
        nt_elems = len(mf) // inner_dim if inner_dim > 0 else 1
        mf_row = mf[:n_vals].astype(np.float64)
        cos = float(np.dot(ref, mf_row) / (np.linalg.norm(ref) * np.linalg.norm(mf_row) + 1e-30))
        print(f"\nCross-validation ({args.compare}, {nt_elems} rows):")
        print(f"  cosine = {cos:.10f}  {'✓' if cos > 0.9999 else '✗ MISMATCH'}")
        if cos < 0.9999:
            for i in range(min(8, n_vals)):
                print(f"  [{i}] ref={ref[i]:+.8e}  minfer={mf_row[i]:+.8e}")


if __name__ == "__main__":
    main()
