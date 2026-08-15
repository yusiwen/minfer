"""Compare minfer dumps vs llama.cpp reference.

Supports both per-layer (if files exist) and full-model (single dump) comparison.

Usage:
    uv run python -m scripts.compare_layers --llama-dir ./llama_ref --minfer-dir /tmp --hidden-dim 896
"""

import argparse
import glob
import os
import sys
import numpy as np

from .lib import load_f32, cosine_sim


def main():
    parser = argparse.ArgumentParser(description="Compare minfer vs llama.cpp output")
    parser.add_argument("--llama-dir", default="/tmp/llama_ref",
                        help="Directory with llama reference .npy files")
    parser.add_argument("--minfer-dir", default="/tmp",
                        help="Directory with minfer dump .f32 files")
    parser.add_argument("--hidden-dim", type=int, default=896,
                        help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=24,
                        help="Total layers (for per-layer comparison)")
    args = parser.parse_args()

    # ── Prompt validation ─────────────────────────────────────
    llama_prompt_path = os.path.join(args.llama_dir, "prompt.txt")
    minfer_prompt_path = os.path.join(args.minfer_dir, "minfer_dump_prompt.txt")
    lp_exists = os.path.exists(llama_prompt_path)
    mp_exists = os.path.exists(minfer_prompt_path)
    if lp_exists and mp_exists:
        l_text = open(llama_prompt_path).read().strip()
        m_text = open(minfer_prompt_path).read().strip()
        if l_text != m_text:
            print("=" * 60)
            print("PROMPT MISMATCH — comparison aborted")
            print("=" * 60)
            print(f"  Llama  prompt ({len(l_text)} chars): {l_text[:200]}...")
            print(f"  Minfer prompt ({len(m_text)} chars): {m_text[:200]}...")
            sys.exit(1)
        print(f"Prompt match: ✓ ({len(l_text)} chars)")
    elif lp_exists or mp_exists:
        missing = "minfer" if lp_exists else "llama"
        print(f"[WARN] Only {missing} prompt file found — skipping validation")

    found_any = False

    # ── Per-layer comparison (if files exist) ────────────────
    llama_hidden = {}
    for n in range(1, args.layers + 1):
        path = os.path.join(args.llama_dir, f"layer{n}_hidden_states.npy")
        if os.path.exists(path):
            llama_hidden[n] = np.load(path)

    minfer_files = sorted(glob.glob(
        os.path.join(args.minfer_dir, "minfer_dump_layer*_out.f32")
    ))
    minfer_layers = {}
    for f in minfer_files:
        basename = os.path.basename(f)
        parts = basename.replace("minfer_dump_layer", "").replace("_out.f32", "").split("_")
        layer_idx = int(parts[0])
        data = load_f32(f)
        n_tokens = len(data) // args.hidden_dim
        last_token = data[-args.hidden_dim:] if n_tokens > 0 else data
        minfer_layers[layer_idx] = last_token

    if llama_hidden and minfer_layers:
        found_any = True
        print(f"\n=== Per-layer comparison ===")
        print(f"{'Layer':<8} {'minfer RMS':<12} {'llama RMS':<12} {'ratio':<10} {'cos':<12}")
        print("-" * 60)
        first_divergence = None

        for llama_n in sorted(llama_hidden.keys()):
            minfer_idx = llama_n - 1
            if minfer_idx not in minfer_layers:
                continue

            l_h = llama_hidden[llama_n]
            m_h = minfer_layers[minfer_idx]
            l_rms = float(np.sqrt(np.mean(l_h**2)))
            m_rms = float(np.sqrt(np.mean(m_h**2)))
            rms_ratio = m_rms / l_rms if l_rms > 0 else 0.0
            cos = cosine_sim(m_h, l_h)

            marker = ""
            if cos < 0.999:
                if first_divergence is None:
                    first_divergence = minfer_idx
                    marker = "  ← FIRST DIVERGENCE"
                else:
                    marker = "  ← DIVERGED"

            print(f"  {minfer_idx:<6} {m_rms:<12.4f} {l_rms:<12.4f} "
                  f"{rms_ratio:<10.4f} {cos:<12.6f}{marker}")

        print(f"\n--- Per-layer summary ---")
        if first_divergence is not None:
            print(f"  ❌ First divergence at minfer layer {first_divergence}")
        else:
            print(f"  ✅ All layers match (cos >= 0.999)")

    # ── Full-model comparison ─────────────────────────────────
    full_hidden_path = os.path.join(args.llama_dir, "full_hidden_states.npy")
    if os.path.exists(full_hidden_path):
        l_hidden = np.load(full_hidden_path)
        # Find the corresponding minfer dump (last layer output)
        last_layer_out = os.path.join(args.minfer_dir, f"minfer_dump_layer{args.layers-1}_out.f32")
        if os.path.exists(last_layer_out):
            m_data = load_f32(last_layer_out)
            m_last = m_data[-args.hidden_dim:] if len(m_data) > args.hidden_dim else m_data
            cos_full = cosine_sim(m_last, l_hidden)
            l_rms = float(np.sqrt(np.mean(l_hidden**2)))
            m_rms = float(np.sqrt(np.mean(m_last**2)))
            found_any = True
            if not llama_hidden:
                print(f"\n=== Full-model comparison ===")
                print(f"  minfer RMS={m_rms:.4f}, llama RMS={l_rms:.4f}")
                print(f"  Cosine: {cos_full:.6f} {'✓' if cos_full > 0.999 else '✗'}")

    # ── Logits comparison ─────────────────────────────────────
    llama_logits = os.path.join(args.llama_dir, "logits_prefill.npy")
    minfer_logits = os.path.join(args.minfer_dir, "minfer_dump_logits.f32")
    if os.path.exists(llama_logits) and os.path.exists(minfer_logits):
        l_logits = np.load(llama_logits).astype(np.float64)
        m_logits = load_f32(minfer_logits).astype(np.float64)
        if len(m_logits) == len(l_logits):
            found_any = True
            cos_logits = cosine_sim(m_logits, l_logits)
            l_top = int(np.argmax(l_logits))
            m_top = int(np.argmax(m_logits))
            print(f"\n=== Logits ===")
            print(f"  Cosine:       {cos_logits:.6f}")
            print(f"  Llama top:    {l_top}")
            print(f"  Minfer top:   {m_top}")
            print(f"  Match:        {'✓' if l_top == m_top else '✗'}")
        else:
            print(f"\nLogits shape mismatch: minfer={len(m_logits)} llama={len(l_logits)}")

    if not found_any:
        print("\nERROR: No data found for comparison!")
        sys.exit(1)


if __name__ == "__main__":
    main()
