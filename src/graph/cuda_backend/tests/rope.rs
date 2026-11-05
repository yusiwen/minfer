//! #735: the CUDA rope serves both pairings.
//!
//! The kernel used to be NEOX-only, so a Llama file (which rotates pairs of
//! consecutive head values) had every rope node assigned to the CPU. This gate
//! compares the **kernel** against a scalar oracle written out here — not against
//! the CPU path, so a shared fault cannot hide (GATE-CONTRACT rule 1) — for both
//! layouts, and pins the identity arm (position 0 rotates nothing in either).

use super::*;

/// The scalar oracle: the kernel's formula, written independently.
fn scalar_rope(
    x: &[f32],
    n_head: usize,
    hd: usize,
    nt: usize,
    positions: &[usize],
    style: RopeStyle,
    freq_base: f32,
    freq_scale: f32,
) -> Vec<f32> {
    let mut y = x.to_vec();
    let half = hd / 2;
    for t in 0..nt {
        for h in 0..n_head {
            let base = (t * n_head + h) * hd;
            for d in 0..half {
                let freq = freq_scale / freq_base.powf((2.0 * d as f32) / hd as f32);
                let theta = positions[t] as f32 * freq;
                let (cs, sn) = (theta.cos(), theta.sin());
                let (i0, i1) = match style {
                    RopeStyle::Interleaved => (base + 2 * d, base + 2 * d + 1),
                    RopeStyle::NonInterleaved => (base + d, base + d + half),
                };
                let (x0, x1) = (y[i0], y[i1]);
                y[i0] = x0 * cs - x1 * sn;
                y[i1] = x0 * sn + x1 * cs;
            }
        }
    }
    y
}

#[test]
fn cuda_rope_matches_the_scalar_reference_in_both_layouts() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (n_head, hd, nt) = (2usize, 4usize, 3usize);
    let (freq_base, freq_scale) = (10000.0f32, 1.0f32);
    // Token 0 sits at position 0: the identity arm, in both layouts.
    let positions: Vec<usize> = vec![0, 3, 7];
    let x: Vec<f32> = (0..nt * n_head * hd)
        .map(|i| (i as f32) * 0.125 - 1.0)
        .collect();
    let xb0 = cb.alloc_buffer(nt * n_head * hd);
    cb.write_host(xb0, &x).unwrap();

    for style in [RopeStyle::NonInterleaved, RopeStyle::Interleaved] {
        let mut b = GraphBuilder::new();
        let xi = b.input("x", [n_head * hd, nt, 1, 1], DType::F32);
        let p = b.input("positions", [nt, 1, 1, 1], DType::I32);
        let r = b.rope(
            xi,
            p,
            style,
            RoPEMeta {
                freq_base,
                freq_scale,
                n_head,
                hd,
                freqs: None,
            },
        );
        b.output(r);
        let g = b.build();

        let out = cb.alloc_buffer(nt * n_head * hd);
        let posb = cb.alloc_buffer(nt);
        let posf: Vec<f32> = positions
            .iter()
            .map(|&p| f32::from_bits(p as u32))
            .collect();
        cb.write_host(posb, &posf).unwrap();
        cb.exec_ids(&g.nodes[r], &[xb0, posb], out, None).unwrap();

        let got = cb.copy_to_host(out).unwrap();
        let want = scalar_rope(&x, n_head, hd, nt, &positions, style, freq_base, freq_scale);
        assert_close(&format!("rope {style:?}"), &got, &want, 1e-5);

        // The identity arm: token 0 (position 0) must be untouched, so a layout
        // that rotates the wrong pair cannot pass on a flat average.
        let per_token = n_head * hd;
        assert_eq!(
            &got[..per_token],
            &x[..per_token],
            "{style:?}: position 0 must be the identity"
        );
    }
}
