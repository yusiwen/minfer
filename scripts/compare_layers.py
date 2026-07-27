"""Compare minfer layer dumps vs llama.cpp reference — per-layer cosine / RMS / error analysis.

Usage:
    uv run python -m scripts.compare_layers --llama-dir ./llama_ref --minfer-dir /tmp --hidden-dim 896
    uv run python -m scripts.compare_layers --llama-dir ./llama_ref --minfer-dir /tmp --hidden-dim 896 --layers 24

Assumptions:
    llama_ref/layer{N}_hidden_states.npy   (last-token hidden state after N layers, N=1..L)
    {minfer_dir}/minfer_dump_layer{N}_out.f32  (all-tokens hidden state, last token extracted, N=0..L-1)
"""

import argparse
import glob
import os
import sys
import numpy as np

from .lib import load_f32, cosine_sim


def main():
    parser = argparse.ArgumentParser(description="Compare minfer vs llama.cpp layer outputs")
    parser.add_argument("--llama-dir", default="./llama_ref",
                        help="Directory with llama reference .npy files")
    parser.add_argument("--minfer-dir", default="/tmp",
                        help="Directory with minfer dump .f32 files")
    parser.add_argument("--hidden-dim", type=int, default=896,
                        help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=24,
                        help="Total number of layers")
    args = parser.parse_args()

    # ── Load llama reference ─────────────────────────────────
    llama_hidden = {}
    for n in range(1, args.layers + 1):
        path = os.path.join(args.llama_dir, f"layer{n}_hidden_states.npy")
        if os.path.exists(path):
            llama_hidden[n] = np.load(path)

    if not llama_hidden:
        print("ERROR: No llama reference files found!")
        print(f"  Expected: {args.llama_dir}/layer{{N}}_hidden_states.npy")
        print(f"  Run: uv run scripts/dump_llama_ref.py --model <gguf> --prompt Hello")
        sys.exit(1)

    # ── Load minfer dumps ────────────────────────────────────
    minfer_files = sorted(glob.glob(
        os.path.join(args.minfer_dir, "minfer_dump_layer*_out.f32")
    ))
    minfer_layers = {}
    for f in minfer_files:
        # Parse "minfer_dump_layer{N}_out.f32"
        basename = os.path.basename(f)
        parts = basename.replace("minfer_dump_layer", "").replace("_out.f32", "").split("_")
        layer_idx = int(parts[0])
        data = load_f32(f)
        n_tokens = len(data) // args.hidden_dim
        last_token = data[-args.hidden_dim:] if n_tokens > 0 else data
        minfer_layers[layer_idx] = last_token

    if not minfer_layers:
        print("ERROR: No minfer dump files found!")
        print(f"  Expected: {args.minfer_dir}/minfer_dump_layer{{N}}_out.f32")
        sys.exit(1)

    print(f"Llama ref layers: {sorted(llama_hidden.keys())}")
    print(f"Minfer layers:    {sorted(minfer_layers.keys())}")
    print()

    # ── Layer-by-layer comparison ────────────────────────────
    print(f"{'Layer':<8} {'minfer RMS':<12} {'llama RMS':<12} {'ratio':<10} {'cos':<12}")
    print("-" * 60)

    found_any = False
    first_divergence = None

    for llama_n in sorted(llama_hidden.keys()):
        minfer_idx = llama_n - 1  # llama layer N = minfer layer N-1
        if minfer_idx not in minfer_layers:
            continue

        found_any = True
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

    if not found_any:
        print("\nNo matching layers found!")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n--- Summary ---")
    if first_divergence is not None:
        print(f"  ❌ First divergence at minfer layer {first_divergence}")
    else:
        print(f"  ✅ All layers match (cos >= 0.999)")

    # ── Logits comparison (if available) ─────────────────────
    llama_logits = os.path.join(args.llama_dir, "logits_prefill.npy")
    minfer_logits = os.path.join(args.minfer_dir, "minfer_dump_logits.f32")
    if os.path.exists(llama_logits) and os.path.exists(minfer_logits):
        l_logits = np.load(llama_logits).astype(np.float64)
        m_logits = load_f32(minfer_logits).astype(np.float64)
        if len(m_logits) == len(l_logits):
            cos_logits = cosine_sim(m_logits, l_logits)
            l_top = int(np.argmax(l_logits))
            m_top = int(np.argmax(m_logits))
            print(f"\nLogits comparison:")
            print(f"  Cosine:       {cos_logits:.6f}")
            print(f"  Llama top:    {l_top}  (score={l_logits[l_top]:.2f})")
            print(f"  Minfer top:   {m_top}  (score={m_logits[m_top]:.2f})")
        else:
            print(f"\nLogits shape mismatch: minfer={len(m_logits)} llama={len(l_logits)}")


if __name__ == "__main__":
    main()
