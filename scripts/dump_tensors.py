"""Dump GGUF tensor layout — offset, byte size, shape, and type.

Output format aligns with llama.cpp's gguf_dump.py for easy diff comparison.

Usage:
    uv run python -m scripts.dump_tensors --model <model.gguf>
    uv run python -m scripts.dump_tensors --model <model.gguf> --summary
"""

import argparse
import sys
from gguf import GGUFReader, GGML_QUANT_SIZES
from gguf.constants import GGMLQuantizationType


def main():
    parser = argparse.ArgumentParser(description="Dump GGUF tensor layout")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--summary", action="store_true",
                        help="Show only summary (total tensors, data section size)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter tensor names (substring match)")
    args = parser.parse_args()

    reader = GGUFReader(args.model)

    # Detect data section offset
    data_offset = reader.data_offset if hasattr(reader, 'data_offset') else 0

    if args.summary:
        total_bytes = sum(t.n_bytes for t in reader.tensors)
        print(f"tensor_count={len(reader.tensors)}")
        print(f"data_offset={data_offset}")
        print(f"total_n_bytes={total_bytes}")
        return

    for t in reader.tensors:
        if args.filter and args.filter not in t.name:
            continue
        qtype = GGMLQuantizationType(t.tensor_type)
        shape_str = f"[{','.join(str(s) for s in t.shape)}]"
        print(f"offset={t.data_offset} n_bytes={t.n_bytes} "
              f"shape={shape_str} type={qtype.name} name={t.name}")


if __name__ == "__main__":
    main()
