"""Verify token embedding dequant for a GGUF model.

Uses gguf.quants.dequantize() — the same Python implementation validated
against llama.cpp's C reference in gguf-py/tests/test_quants.py.
Auto-detects quantization type from the model's token_embd tensor.

Usage:
    uv run python -m scripts.verify_embed --model <path.gguf>
    uv run python -m scripts.verify_embed --model <path.gguf> --token-id 0
"""

import argparse
import sys
import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize
from gguf import GGML_QUANT_SIZES


def read_tensor_data(model_path, data_offset, nbytes):
    """Read raw tensor data from GGUF file."""
    with open(model_path, 'rb') as f:
        f.seek(data_offset)
        return f.read(nbytes)


def main():
    parser = argparse.ArgumentParser(description="Verify token embedding dequant")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--token-id", type=int, default=0,
                        help="Token ID to dequant (default: 0 = first vocab entry)")
    args = parser.parse_args()

    from gguf import GGUFReader
    reader = GGUFReader(args.model)

    # Find token_embd tensor
    embd_off = None
    embd_shape = None
    hidden_dim = None
    embd_type = None
    for t in reader.tensors:
        if 'token_embd' in t.name:
            embd_off = t.data_offset
            embd_shape = t.shape
            hidden_dim = t.shape[0]
            embd_type = t.tensor_type
            break

    if embd_off is None:
        print("ERROR: token_embd.weight not found in model")
        sys.exit(1)

    hidden_dim = hidden_dim or 896
    nb = hidden_dim // 32  # blocks per row

    qtype = GGMLQuantizationType(embd_type)
    block_size, type_size = GGML_QUANT_SIZES[qtype]

    print(f"Model: {args.model}")
    print(f"Embedding shape: {embd_shape}  type: {qtype.name}")
    print(f"Hidden dim: {hidden_dim}  blocks per row: {nb}")
    print(f"Token ID: {args.token_id}")
    print()

    # Read one row of embedding data
    row_bytes = nb * type_size
    offset = embd_off + args.token_id * row_bytes
    raw = read_tensor_data(args.model, offset, row_bytes)

    # Dequantize using verified gguf.quants.dequantize
    data = np.frombuffer(raw, dtype=np.uint8)
    vals = dequantize(data, qtype)

    rms = float(np.sqrt(np.mean(vals ** 2)))
    print(f"Dequantized row: {len(vals)} elements")
    print(f"RMS: {rms:.6f}")
    print(f"Min={vals.min():.6f}  Max={vals.max():.6f}")
    print(f"Mean: {vals.mean():.6f}  Std: {vals.std():.6f}")
    print()
    print("First 8 values:")
    for i in range(min(8, len(vals))):
        print(f"  [{i}] {vals[i]:+.8e}")

    # Print first block raw bytes for debugging
    print()
    print("Block 0 raw data:")
    blk = raw[:type_size]
    print(f"  hex: {blk.hex()}")


if __name__ == "__main__":
    main()
