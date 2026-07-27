"""Generate llama.cpp per-layer reference hidden states for minfer comparison.

Uses the "truncated model" approach: for each layer N, creates a fake GGUF
with block_count=N (zero-copy, metadata only), loads it in llama-cpp-python,
prefills the prompt, and extracts hidden states via llama_get_embeddings().

Usage:
    uv run python -m scripts.dump_llama_ref --model <gguf> --prompt "Hello"
    uv run python -m scripts.dump_llama_ref --model <gguf> --prompt "Hello" --output ./ref

Output (per layer):
    {output_dir}/layer{N}_hidden_states.npy   last-token hidden state
    {output_dir}/logits_prefill.npy           final logits (full model only)
    {output_dir}/token_ids.npy                input token IDs
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama, llama_cpp

from .lib import create_truncated_model


def main():
    parser = argparse.ArgumentParser(description="Dump llama.cpp per-layer reference data")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", default="Hello", help="Prompt text (bare, no chat template)")
    parser.add_argument("--output", default="./llama_ref", help="Output directory")
    parser.add_argument("--layers", type=int, default=0,
                        help="Number of layers (0 = auto-detect from GGUF)")
    parser.add_argument("--no-bos", action="store_true", help="Don't add BOS token")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ── Phase 1: Read model metadata ─────────────────────────
    print("=== Phase 1: Reading model metadata ===")
    reader = GGUFReader(model_path)

    arch = "qwen2"
    if "general.architecture" in reader.fields:
        raw = reader.fields["general.architecture"].parts[-1]
        arch = raw.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
    print(f"  Architecture: {arch}")

    # Detect total layers
    total_layers = args.layers
    if total_layers == 0:
        for name in ("block_count", f"{arch}.block_count"):
            if name in reader.fields:
                total_layers = int(reader.fields[name].parts[-1].tobytes().decode())
                break
    if total_layers == 0:
        # Fallback: count blk.N.* tensors
        layers = set()
        for t in reader.tensors:
            if "blk." in t.name:
                try:
                    layers.add(int(t.name.split(".")[1]))
                except (ValueError, IndexError):
                    pass
        total_layers = max(layers) + 1 if layers else 24
    print(f"  Total layers: {total_layers}")

    # Detect hidden_dim from token_embd
    hidden_dim = 896
    for t in reader.tensors:
        if "token_embd" in t.name:
            hidden_dim = t.shape[0]
            break
    print(f"  Hidden dim: {hidden_dim}")

    # ── Phase 2: Tokenize prompt ────────────────────────────
    print(f"\n=== Phase 2: Tokenizing prompt: {args.prompt!r} ===")
    full_llm = Llama(
        model_path=model_path, n_ctx=512, n_gpu_layers=0,
        embedding=True, verbose=False,
    )
    token_ids = full_llm.tokenize(
        args.prompt.encode("utf-8"), add_bos=not args.no_bos, special=True,
    )
    input_length = len(token_ids)
    print(f"  Token IDs: {token_ids}")
    print(f"  Input length: {input_length}")
    np.save(os.path.join(args.output, "token_ids.npy"), np.array(token_ids, dtype=np.int64))
    full_llm.close()
    del full_llm

    # ── Phase 3: Per-layer dump ──────────────────────────────
    print(f"\n=== Phase 3: Dumping layers 1..{total_layers} ===")
    temp_dir = tempfile.mkdtemp(prefix="minfer_dump_")

    for layer_target in range(1, total_layers + 1):
        t0 = time.time()
        fake_path = os.path.join(temp_dir, f"truncated_{layer_target}.gguf")

        n_tensors = create_truncated_model(model_path, fake_path, layer_target, arch)
        t1 = time.time()

        llm = Llama(
            model_path=fake_path, n_ctx=512, n_gpu_layers=0,
            embedding=True, verbose=False,
        )
        t2 = time.time()

        llm.eval(token_ids)
        t3 = time.time()

        emb_ptr = llama_cpp.llama_get_embeddings(llm._ctx.ctx)
        if not emb_ptr:
            print(f"  ❌ Layer {layer_target}: llama_get_embeddings returned NULL")
            llm.close()
            os.remove(fake_path)
            continue

        total_elems = input_length * hidden_dim
        hidden = np.array(
            emb_ptr[:total_elems], dtype=np.float32
        ).reshape(input_length, hidden_dim)

        last_hidden = hidden[-1, :]
        out_path = os.path.join(args.output, f"layer{layer_target}_hidden_states.npy")
        np.save(out_path, last_hidden)

        # Save full-model logits
        if layer_target == total_layers:
            logits_ptr = llm._ctx.get_logits()
            logits = np.array(logits_ptr[:llm.n_vocab()], dtype=np.float32)
            np.save(os.path.join(args.output, "logits_prefill.npy"), logits)

        llm.close()
        os.remove(fake_path)

        t4 = time.time()
        h_rms = float(np.sqrt(np.mean(last_hidden**2)))
        print(f"  Layer {layer_target:2d}/{total_layers} | "
              f"total={t4-t0:.1f}s | RMS={h_rms:.4f}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n=== Done ===")
    print(f"  Output: {os.path.abspath(args.output)}")
    files = sorted(os.listdir(args.output))
    print(f"  Files ({len(files)}): {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")


if __name__ == "__main__":
    main()
