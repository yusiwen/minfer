"""Generate llama.cpp reference (full model, single pass) for minfer comparison.

Usage:
    uv run python -m scripts.dump_llama_ref --model <gguf> --prompt "Hello"
    uv run python -m scripts.dump_llama_ref --model <gguf> --prompt "Hello" --chat

Output:
    {output_dir}/full_hidden_states.npy     last-token hidden state (after all layers)
    {output_dir}/logits_prefill.npy         final logits
    {output_dir}/token_ids.npy              input token IDs
    {output_dir}/prompt.txt                 rendered prompt text
"""

import argparse
import os
import struct
import sys
import time
import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama, llama_cpp


def main():
    parser = argparse.ArgumentParser(description="Dump llama.cpp reference data (full model)")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--chat", action="store_true",
                        help="Wrap with chat_template (matches minfer)")
    parser.add_argument("--output", default="./llama_ref", help="Output directory")
    parser.add_argument("--no-bos", action="store_true", help="Don't add BOS token")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ── Phase 1: Read metadata ───────────────────────────────
    print("=== Phase 1: Model metadata ===")
    reader = GGUFReader(model_path)
    arch = "qwen2"
    if "general.architecture" in reader.fields:
        raw_bytes = reader.fields["general.architecture"].parts[-1].tobytes()
        arch = raw_bytes.decode("utf-8", errors="ignore").strip("\x00").strip()

    total_layers = 0
    for name in ("block_count", f"{arch}.block_count"):
        if name in reader.fields:
            raw = reader.fields[name].parts[-1].tobytes()
            total_layers = struct.unpack("<I", raw)[0]
            break
    print(f"  Architecture: {arch}, Layers: {total_layers}")

    hidden_dim = 896
    for t in reader.tensors:
        if "token_embd" in t.name:
            hidden_dim = t.shape[0]
            break
    print(f"  Hidden dim: {hidden_dim}")

    # ── Phase 1b: Chat template ──────────────────────────────
    actual_prompt = args.prompt
    if args.chat:
        tmpl_raw = None
        for f in reader.fields.values():
            if f.name == "tokenizer.chat_template":
                tmpl_raw = f.parts[-1].tobytes().decode("utf-8", errors="replace").strip("\x00")
                break
        if tmpl_raw:
            from jinja2 import Environment
            env = Environment()
            env.add_extension("jinja2.ext.do")
            tmpl = env.from_string(tmpl_raw)
            messages = [{"role": "user", "content": args.prompt}]
            actual_prompt = tmpl.render(messages=messages, add_generation_prompt=True).strip()
            print(f"  Chat template: {len(actual_prompt)} chars")
        else:
            print("  [WARN] --chat set but no template found")

    with open(os.path.join(args.output, "prompt.txt"), "w") as f:
        f.write(actual_prompt)

    # ── Phase 2: Tokenize + infer ────────────────────────────
    print(f"\n=== Phase 2: Inference ===")
    full_llm = Llama(
        model_path=model_path, n_ctx=512, n_gpu_layers=0,
        embedding=True, verbose=False,
    )
    # minfer's render_template always prepends BOS text. Match that.
    add_bos = not args.no_bos and (not args.chat)
    if args.chat:
        # Read BOS token text from GGUF
        bos_token_id = 151643
        for f in reader.fields.values():
            if f.name == "tokenizer.ggml.bos_token_id":
                bos_token_id = struct.unpack("<I", f.parts[-1].tobytes())[0]
                break
        # Get BOS token text by tokenizing a placeholder then decoding
        bos_text_piece = full_llm.detokenize([bos_token_id])
        bos_text = bos_text_piece.decode("utf-8", errors="replace") if isinstance(bos_text_piece, bytes) else bos_text_piece
        if bos_text and bos_text != "[PAD151643]":
            actual_prompt = bos_text + actual_prompt
            # Update prompt.txt
            with open(os.path.join(args.output, "prompt.txt"), "w") as f:
                f.write(actual_prompt)
    token_ids = full_llm.tokenize(
        actual_prompt.encode("utf-8"), add_bos=add_bos, special=True,
    )
    input_length = len(token_ids)
    print(f"  Prompt: {actual_prompt[:80]}...")
    print(f"  Tokens: {input_length}")

    t0 = time.time()
    full_llm.eval(token_ids)
    t1 = time.time()
    print(f"  Prefill: {t1-t0:.2f}s")

    # ── Phase 3: Extract hidden state ────────────────────────
    print(f"\n=== Phase 3: Extract ===")
    emb_ptr = llama_cpp.llama_get_embeddings(full_llm._ctx.ctx)
    if not emb_ptr:
        print("ERROR: llama_get_embeddings returned NULL")
        full_llm.close()
        sys.exit(1)

    total_elems = input_length * hidden_dim
    hidden = np.array(emb_ptr[:total_elems], dtype=np.float32).reshape(input_length, hidden_dim)
    last_hidden = hidden[-1, :]

    out_path = os.path.join(args.output, "full_hidden_states.npy")
    np.save(out_path, last_hidden)
    rms = float(np.sqrt(np.mean(last_hidden**2)))
    print(f"  Hidden state: RMS={rms:.4f}")

    # ── Phase 4: Extract logits ──────────────────────────────
    logits_ptr = full_llm._ctx.get_logits()
    logits = np.array(logits_ptr[:full_llm.n_vocab()], dtype=np.float32)
    logits_path = os.path.join(args.output, "logits_prefill.npy")
    np.save(logits_path, logits)
    top = int(np.argmax(logits))
    print(f"  Logits: top token={top}")

    # Save token IDs
    np.save(os.path.join(args.output, "token_ids.npy"), np.array(token_ids, dtype=np.int64))

    full_llm.close()

    print(f"\n=== Done ===")
    print(f"  Output: {os.path.abspath(args.output)}")
    for f in sorted(os.listdir(args.output)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
