"""Verify RoPE rotation: compare bq before/after RoPE with Python computation.

Usage:
    uv run python -m scripts.verify_rope --before /tmp/minfer_dump_layer0_bq.f32 \\
      --after /tmp/minfer_dump_layer0_bq_rope.f32 --n-head 14 --n-dims 64 --pos 0
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify RoPE rotation")
    parser.add_argument("--before", required=True,
                        help="Path to bq dump before RoPE")
    parser.add_argument("--after", required=True,
                        help="Path to bq dump after RoPE")
    parser.add_argument("--n-head", type=int, default=14,
                        help="Number of attention heads")
    parser.add_argument("--n-dims", type=int, default=64,
                        help="Per-head dimension")
    parser.add_argument("--freq-base", type=float, default=1000000.0,
                        help="RoPE theta base")
    parser.add_argument("--freq-scale", type=float, default=1.0,
                        help="RoPE frequency scale")
    parser.add_argument("--position", type=int, default=0,
                        help="Token position (0-based in sequence)")
    parser.add_argument("--n-tokens", type=int, default=30,
                        help="Number of tokens in input")
    args = parser.parse_args()

    before = np.fromfile(args.before, dtype=np.float64)
    after = np.fromfile(args.after, dtype=np.float64)

    n_head = args.n_head
    n_dims = args.n_dims
    half = n_dims // 2
    nt = args.n_tokens

    # Compute freqs once for all heads at this position
    p = args.position
    freqs = np.zeros(half, dtype=np.float64)
    for i in range(half):
        freqs[i] = args.freq_scale / np.power(args.freq_base, (2.0 * i) / n_dims)
    theta = p * freqs
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Apply RoPE to before and compare with after
    ref = before.copy()
    for h in range(n_head):
        base = h * n_dims
        for i in range(half):
            x0 = before[base + i]
            x1 = before[base + i + half]
            ref[base + i] = x0 * cos_th[i] - x1 * sin_th[i]
            ref[base + i + half] = x0 * sin_th[i] + x1 * cos_th[i]

    nq = n_head * n_dims  # 896 for Qwen2.5-0.5B
    # Compare per-token: token 0 starts at offset 0
    ref_token0 = ref[:nq]
    mf_token0 = after[:nq]

    cos = float(np.dot(ref_token0, mf_token0) /
                (np.linalg.norm(ref_token0) * np.linalg.norm(mf_token0) + 1e-30))

    print(f"Position: {p}  n_head={n_head}  n_dims={n_dims}")
    print(f"freq_base={args.freq_base}  freq_scale={args.freq_scale}")
    print(f"cosine (token 0): {cos:.10f}  {'✓' if cos > 0.9999 else '✗ MISMATCH'}")
    rms_r = float(np.sqrt(np.mean(ref_token0**2)))
    rms_m = float(np.sqrt(np.mean(mf_token0**2)))
    print(f"RMS ref={rms_r:.4f}  minfer={rms_m:.4f}  ratio={rms_m/rms_r:.4f}")

    # Head-by-head comparison
    print()
    print("Per-head comparison (first 4 values per head):")
    for h in range(min(4, n_head)):
        base = h * n_dims
        rms_h = float(np.sqrt(np.mean(ref[base:base+n_dims]**2)))
        print(f"  Head {h}: ref[{base}]={ref[base]:+.6e}  minfer[{base}]={mf_token0[base]:+.6e}  head_RMS={rms_h:.4f}")


if __name__ == "__main__":
    main()
