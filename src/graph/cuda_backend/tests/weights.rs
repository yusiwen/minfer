//! Weight registration: norm lengths, the q4_K `W_dsc` plane and the dense decode mirrors.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// #169, the CUDA half: a norm weight whose registered length is not the
/// f32 the `rms_norm` kernel indexes (`d*4` bytes) must be refused **before**
/// the launch, because the kernel reads `d*4` bytes regardless and an f16
/// norm (2 B/element) would be read past its end.
///
/// The two arms differ only in that property: the same graph, the same
/// `d = 64`, both names registered, both dims valid float4 multiples — so
/// the **f32** arm (the control) can only pass and the **f16** arm can only
/// fail through the size check. Before #169 the registry lookup alone
/// admitted the f16 arm, which is what makes this a value gate rather than
/// a relation: it asserts the refusal's own text names both lengths.
#[test]
fn cuda_norm_weight_size_is_part_of_the_invariant() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (d, eps) = (64usize, 1e-5f32);
    let run = |cb: &mut CudaBackend, name: &str, t: TensorType, bytes: Vec<u8>| {
        cb.state.register_weight(name, &bytes);
        let mut wt = Tensor::from_data(t, &[d as i64, 1, 1, 1], bytes);
        wt.name = name.to_string();
        let mut b = GraphBuilder::new();
        let x = b.input("x", [d, 1, 1, 1], DType::F32);
        let rn = b.rms_norm(x, Some(&wt), eps);
        b.output(rn);
        let g = b.build();
        let xb = cb.alloc_buffer(d);
        cb.write_host(xb, &vec![1.0f32; d]).unwrap();
        let ob = cb.alloc_buffer(d);
        cb.exec_ids(&g.nodes[rn], &[xb], ob, None)
    };

    // Control arm: f32, exactly `d*4` bytes registered — must execute.
    let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
    let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    run(&mut cb, "n169_f32", TensorType::F32, wbytes)
        .expect("an f32 norm weight of d*4 bytes must execute");

    // Property arm: the same norm weight at 2 B/element. Registered (so the
    // "not registered" refusal cannot be the one that fires) and d is a
    // valid float4 dim, so only the size check can refuse it.
    let f16bytes: Vec<u8> = vec![0u8; d * 2];
    let err = run(&mut cb, "n169_f16", TensorType::F16, f16bytes)
        .expect_err("an f16 norm weight must be refused before the launch");
    assert!(err.contains("n169_f16"), "{err}");
    assert!(
        err.contains(&format!("{} B", d * 2)),
        "the refusal must name the registered length ({} B): {err}",
        d * 2
    );
    assert!(
        err.contains(&format!("{} B", d * 4)),
        "the refusal must name the length the kernel reads ({} B): {err}",
        d * 4
    );
    assert!(err.contains("f16-norm"), "{err}");
}
#[test]
fn cuda_q6k_exp_dense_byte_exact() {
    // r53 gate 1: the pre-expanded dense W_exp plane must be byte-identical
    // to an independent scalar mirror of the device expand_q6_elem over the
    // whole tensor (the r44 readback gate, 0 mismatches) — checked on the
    // HOST expander and on the DEVICE upload (pinned readback).
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let row_len = nbe * 210;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        // padded repack (the register_weight_q6k_padded layout)
        let mut padded = vec![0u8; od * nbe * 224];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * 210;
                let dst = r * nbe * 224 + ib * 224;
                padded[dst..dst + 210].copy_from_slice(&raw[src..src + 210]);
            }
        }
        // independent scalar mirror, straight from the device formula
        let mut want = vec![0u8; od * id];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 224;
                let blk = &padded[base..base + 210];
                let (ql, rest) = blk.split_at(128);
                let qh = &rest[..64];
                for e in 0..256usize {
                    let m = e & 31;
                    let it = e >> 7;
                    let n = e & 127;
                    let ql_idx = it * 64 + (n & 63);
                    let ql_shift = (n >> 6) * 4;
                    let qh_idx = it * 32 + m;
                    let qh_shift = ((n >> 5) & 3) * 2;
                    let v = ((ql[ql_idx] >> ql_shift) & 0x0F)
                        | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
                    want[j * id + sb * 256 + e] = (v as i32 - 32) as u8;
                }
            }
        }
        // host-side production expander vs the mirror
        let host = crate::cuda::CudaState::expand_q6k_dense(&padded, od, id);
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q6k_dense vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r53exp{od}x{id}");
        state.register_weight_q6k_padded(&name, &raw, od, id);
        state.register_weight_q6k_exp(&name, &padded, od, id);
        let exp_name = format!("{name}__exp{od}x{id}");
        let p = state.get_weight_ptr(&exp_name).expect("W_exp registered");
        let mut got = vec![0u8; od * id];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_exp vs mirror ({od}x{id})");
    }
}
#[test]
fn cuda_q6k_dsc_dense_byte_exact() {
    // r56 (Session E item 2b) gate 1: the precomputed dsc f32-pair plane
    // must be byte-identical to an independent scalar mirror of the
    // kernel's in-loop dsc computation (d = f16(blk+208); dsc = d *
    // (int8)blk[192 + 2*(c&7) + {0,1}]) — checked on the HOST expander and
    // on the DEVICE upload (pinned readback), over shapes covering several
    // super-blocks per row and od values that exercise the chunk-major
    // [c*od + j] layout.
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0xC0FF_EE12_3456_789A;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let nchunk = id / 32;
        let row_len = nbe * 210;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        let mut padded = vec![0u8; od * nbe * 224];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * 210;
                let dst = r * nbe * 224 + ib * 224;
                padded[dst..dst + 210].copy_from_slice(&raw[src..src + 210]);
            }
        }
        // independent scalar mirror straight from the kernel formula
        let mut want = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 224;
                let blk = &padded[base..base + 210];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
                for cc in 0..8usize {
                    let sc0 = blk[192 + 2 * cc] as i8 as f32;
                    let sc1 = blk[192 + 2 * cc + 1] as i8 as f32;
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    want[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
                    want[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
                }
            }
        }
        // host-side production expander vs the mirror
        let host = crate::cuda::CudaState::expand_q6k_dsc(&padded, od, id);
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q6k_dsc vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r56dsc{od}x{id}");
        state.register_weight_q6k_padded(&name, &raw, od, id);
        state.register_weight_q6k_dsc(&name, &padded, od, id);
        let dsc_name = format!("{name}__dsc{od}x{id}");
        let p = state.get_weight_ptr(&dsc_name).expect("W_dsc registered");
        let mut got = vec![0u8; nchunk * od * 8];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_dsc vs mirror ({od}x{id})");
    }
}
#[test]
fn cuda_q4k_dsc_dense_byte_exact() {
    // r59 (Session F item 1) gate 1: the precomputed q4_K dsc f32-pair
    // plane must be byte-identical to an independent scalar mirror of the
    // kernel's in-loop SDS decode (d = f16(blk), dmin = f16(blk+2),
    // (sc, m) = get_scale_min_k4(c&7, blk+4), pair = (d*sc, -(dmin*m))).
    // Checked on the HOST expander and on the DEVICE upload (pinned
    // readback), over shapes covering several super-blocks per row and od
    // values that exercise the chunk-major [c*od + j] layout. Q4_K needs
    // no padding: the raw 144-byte block stride is already 16-B aligned.
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0xC0FF_EE12_3456_789B;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let nchunk = id / 32;
        let row_len = nbe * 144;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        // independent scalar mirror straight from the kernel formula
        let mut want = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 144;
                let blk = &raw[base..base + 144];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
                let q = &blk[4..16]; // 12 packed 6-bit scales+mins
                for cc in 0..8usize {
                    let (sc, m) = if cc < 4 {
                        (q[cc] & 63, q[cc + 4] & 63)
                    } else {
                        (
                            (q[cc + 4] & 0xF) | ((q[cc - 4] >> 6) << 4),
                            (q[cc + 4] >> 4) | ((q[cc] >> 6) << 4),
                        )
                    };
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    want[idx..idx + 4].copy_from_slice(&(d * (sc as f32)).to_bits().to_le_bytes());
                    want[idx + 4..idx + 8]
                        .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
                }
            }
        }
        // host-side production expander vs the mirror
        let host =
            crate::cuda::CudaState::expand_q4k_dsc(&raw, od, id).expect("q4_K payload length");
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q4k_dsc vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r59dsc{od}x{id}");
        state.register_weight(&name, &raw);
        state.register_weight_q4k_dsc(&name, &raw, od, id);
        let dsc_name = format!("{name}__q4dsc{od}x{id}");
        let p = state.get_weight_ptr(&dsc_name).expect("W_dsc registered");
        let mut got = vec![0u8; nchunk * od * 8];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_dsc vs mirror ({od}x{id})");
    }
}
/// #165: a payload that is not a q4_K payload registers **nothing** — no
/// `__q4dsc` device weight and no `q4k_dsc` map entry — while the q4_K payload
/// does. The positive control comes first and its registration is asserted by the
/// same registry queries the refusal uses, so a green run really observes the plane
/// (a query that is blind to planes could not see the control either).
#[test]
fn cuda_q4dsc_plane_is_q4k_only() {
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Qwen3-0.6B `ffn_down` geometry — the shape #165 names.
    let (od, id) = (1024usize, 3072usize);
    let q4k_len = id / 256 * 144 * od;
    let planes = |s: &crate::cuda::CudaState| -> (usize, usize) {
        let p = s.q4dsc_planes();
        (p.len(), p.iter().map(|(_, b)| b).sum())
    };
    let (base_n, base_b) = planes(state);

    // positive control: a real q4_K payload DOES register
    let ok_name = format!("f165q4k{od}x{id}");
    let raw = vec![0u8; q4k_len];
    state.register_weight(&ok_name, &raw);
    state.register_weight_q4k_dsc(&ok_name, &raw, od, id);
    let ok_dsc = format!("{ok_name}__q4dsc{od}x{id}");
    assert!(
        state.get_weight_ptr(&ok_dsc).is_some(),
        "positive control: the q4_K payload must register {ok_dsc}"
    );
    let (n_after_ok, b_after_ok) = planes(state);
    assert_eq!(
        (n_after_ok, b_after_ok),
        (base_n + 1, base_b + (id / 32) * od * 8),
        "the registry query must see exactly the control's plane"
    );

    // a q8_0-length payload (34 B / 32 elements) is LONGER than q4_K's: refused
    let q80_name = format!("f165q80{od}x{id}");
    let q80 = vec![0u8; od * (id / 32) * 34];
    state.register_weight(&q80_name, &q80);
    state.register_weight_q4k_dsc(&q80_name, &q80, od, id);
    assert!(
        state
            .get_weight_ptr(&format!("{q80_name}__q4dsc{od}x{id}"))
            .is_none(),
        "a q8_0 payload must not register a __q4dsc plane"
    );

    // a shorter payload (a future smaller-ratio type) is refused too, instead of
    // being read past the tensor
    let short_name = format!("f165short{od}x{id}");
    let short = vec![0u8; q4k_len - 144];
    state.register_weight(&short_name, &short);
    state.register_weight_q4k_dsc(&short_name, &short, od, id);
    assert!(
        state
            .get_weight_ptr(&format!("{short_name}__q4dsc{od}x{id}"))
            .is_none(),
        "a short payload must not register a __q4dsc plane"
    );

    // the two refusals added exactly zero planes
    assert_eq!(
        planes(state),
        (n_after_ok, b_after_ok),
        "only the q4_K control may add a plane"
    );
}
/// #165 acceptance: loading a real model on CUDA registers a `W_dsc` plane for
/// **exactly** the admissible q4_K weights and nothing else. The default cached model
/// is the 0.5B q4_0, whose `ffn_down` `[4864, 896]` passed the old geometry gate: the
/// type gate is the only thing that can refuse it (q4_0's bytes/element equals
/// q4_K's), so before the fix this asserted 24 planes / 26 148 864 B. Point
/// `MINFER_BATCH_TEST_MODEL` at a qwen2 q8_0 GGUF to measure the q8_0 model the
/// ticket names (expected: 0 either way, qwen3's loader never registered the plane).
/// `#[ignore]`: it needs a cached GGUF (the real-model set).
#[test]
#[ignore]
fn cuda_real_model_registers_q4dsc_planes_only_for_q4k() {
    use crate::gguf::GgmlType;
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let path = match std::env::var("MINFER_BATCH_TEST_MODEL") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let mut p = std::path::PathBuf::from(std::env::var("HOME").unwrap());
            p.push(
                ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/\
                 qwen2.5-0.5b-instruct-q4_0.gguf",
            );
            p
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} not cached", path.display());
        return;
    }
    // Hold the model-load lock across load + query (a parallel load of another
    // architecture registers same-named tensors and swaps the registry underneath).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    // Expected set, from the GGUF index with the loader's own rule: a 2-D q4_K
    // tensor (ne[0] = in = id, ne[1] = out = od) with `id % 256 == 0` and `od` even.
    let mut want: Vec<String> = Vec::new();
    for part in &gguf.parts {
        for ti in &part.ctx.info {
            if ti.type_ != GgmlType::Q4_K {
                continue;
            }
            let (id, od) = (ti.ne[0] as usize, ti.ne[1] as usize);
            if id == 0 || id % 256 != 0 || od == 0 || od % 2 != 0 {
                continue;
            }
            want.push(format!("{}__q4dsc{od}x{id}", ti.name));
        }
    }
    want.sort();
    let _model = crate::models::load_model(&gguf).expect("load model");
    let planes = state.q4dsc_planes();
    let mut got: Vec<String> = planes.iter().map(|(n, _)| n.clone()).collect();
    got.sort();
    let bytes: usize = planes.iter().map(|(_, b)| b).sum();
    eprintln!(
        "q4dsc planes for {}: {} expected (q4_K), {} registered, {} bytes",
        path.display(),
        want.len(),
        got.len(),
        bytes
    );
    assert_eq!(
        got, want,
        "the W_dsc plane set must be exactly the model's admissible q4_K weights"
    );
}
/// #208 (the CUDA half): the bf16 weight matmul reproduces the **exact**
/// `f32::from_bits(bits << 16)` reference, bitwise.
///
/// The weights and activations are small exact integers (representable in bf16
/// and f32), so every partial product is an integer and the dot is far below
/// 2^24 — the result is independent of accumulation order, so the kernel must
/// match the reference bitwise rather than within a tolerance. The reference
/// decodes each stored bf16 word with the CPU's own exact decode
/// (`crate::block::bf16_to_f32` == `f32::from_bits(bits << 16)`), so a wrong
/// decode (reading the words as `__half`, a byte-order slip, a row/index error)
/// moves the values far outside that and the gate is red.
///
/// Both launcher arms are covered: `id % 8 == 0` selects `bf16_f32_matmul_vec`,
/// and `id % 8 != 0` selects `bf16_f32_matmul_scalar`. The shapes stay inside
/// the kernel's `NR0 = 4` row grouping and below the 256-element `CHK` chunk, so
/// the vec arm really runs its in-chunk loop.
#[test]
fn cuda_bf16_matmul_matches_the_exact_shift_reference() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // (od, id, nt): one vec shape (id % 8 == 0) and one scalar shape, both with
    // od not a multiple of NR0*NSG = 8 so the tail-row guard runs too.
    for (od, id_, nt) in [(10usize, 64usize, 3usize), (5usize, 60usize, 2usize)] {
        // Small exact integers in [-4, 4] / [-3, 3].
        let wv: Vec<f32> = (0..od * id_)
            .map(|k| ((k * 7 + 3) % 9) as f32 - 4.0)
            .collect();
        let xv: Vec<f32> = (0..nt * id_)
            .map(|k| ((k * 5 + 2) % 7) as f32 - 3.0)
            .collect();
        // bf16 store: RNE into the top 16 bits (the writer's rule, #142).
        let wbytes: Vec<u8> = wv
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect();

        let name = format!("f208_bf16_matmul_{od}x{id_}_{nt}");
        cb.state.register_weight(&name, &wbytes);
        assert!(
            cb.state.has_weight_of_size(&name, wbytes.len()),
            "the bf16 weight must be registered at its raw 2 B/element length"
        );
        let wptr = cb
            .state
            .get_weight_ptr(&name)
            .expect("registered weight ptr");

        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xv).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        // The production dispatch path: pointer + TensorType, no graph node.
        let xp = cb.ptr_of(xb).expect("activation buffer pointer");
        let op = cb.ptr_of(ob).expect("output buffer pointer");
        cb.state
            .matmul_f32_ptr_layout(wptr, TensorType::BF16, xp, op, od, id_, nt, false)
            .expect("the bf16 matmul kernel must launch");
        let got = cb.copy_to_host(ob).unwrap();

        for t in 0..nt {
            for r in 0..od {
                // Exact reference: decode the stored word, dot in the kernel's
                // own accumulation class (integer sums are order-independent).
                let mut acc = 0f32;
                for i in 0..id_ {
                    let bits = u16::from_le_bytes([
                        wbytes[2 * (r * id_ + i)],
                        wbytes[2 * (r * id_ + i) + 1],
                    ]);
                    acc += crate::block::bf16_to_f32(bits) * xv[t * id_ + i];
                }
                assert_eq!(
                    got[t * od + r].to_bits(),
                    acc.to_bits(),
                    "bf16 matmul [{t},{r}] (od={od} id={id_} nt={nt}): got {} want {}",
                    got[t * od + r],
                    acc
                );
            }
        }
    }
}

/// #208 (the CUDA half): the bf16 embedding gather decodes the requested rows
/// **exactly** (`f32::from_bits(bits << 16)`), and the ids cross a non-trivial
/// row boundary so a stride/index error cannot pass by symmetry.
///
/// The words are hand-built bit patterns, not values round-tripped through an
/// f32 cast: the reference is the same exact shift the CPU's `Op::GetRows` bf16
/// arm performs, so equality is bitwise. A `__half` misread, a 1-element
/// off-by-one row stride, or an id read as a plain f32 all move values outside
/// that.
#[test]
fn cuda_bf16_embed_gather_matches_the_exact_shift_reference() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (n_vocab, n_embd, nt) = (7usize, 40usize, 4usize);
    let ids = [3u32, 6, 0, 5];
    // Hand-built integer bf16 words: bits = value << 7 gives exactly `value`
    // after the `<< 16` decode (value * 2^16 / 2^7 ... see the reference below),
    // so the expected f32 is computable on the host without a float cast.
    let words: Vec<u16> = (0..n_vocab * n_embd)
        .map(|k| (((k * 11 + 5) % 13) as u16) << 7)
        .collect();
    let wbytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let name = "f208_bf16_embed";
    cb.state.register_weight(name, &wbytes);
    let wptr = cb
        .state
        .get_weight_ptr(name)
        .expect("registered weight ptr");

    // ids as I32-as-f32 bit patterns (graph rule §4).
    let id_bits: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();
    let ib = cb.alloc_buffer(nt);
    cb.write_host(ib, &id_bits).unwrap();
    let ob = cb.alloc_buffer(n_embd * nt);
    let ip = cb.ptr_of(ib).expect("ids buffer pointer");
    let op = cb.ptr_of(ob).expect("output buffer pointer");
    cb.state
        .embed_rows_on_gpu(TensorType::BF16, wptr, ip, op, n_embd, nt, false)
        .expect("the bf16 embed gather must launch");
    let got = cb.copy_to_host(ob).unwrap();

    for (t, &id) in ids.iter().enumerate() {
        for i in 0..n_embd {
            let want = crate::block::bf16_to_f32(words[id as usize * n_embd + i]);
            assert_eq!(
                got[t * n_embd + i].to_bits(),
                want.to_bits(),
                "bf16 embed [{t},{i}] (id={id}): got {} want {want}",
                got[t * n_embd + i]
            );
        }
    }
}
