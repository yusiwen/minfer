//! `#[cfg(test)] mod tests` for `src/graph/kvformat.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn a_cell_is_a_whole_number_of_words() {
    // 128 elements -> 4 blocks -> 136 payload bytes -> 34 words (136 = 4*34).
    assert_eq!(KvFormat::Q8_0.row_elems(128), 34);
    // 64 -> 2 blocks -> 68 bytes -> 17 words.
    assert_eq!(KvFormat::Q8_0.row_elems(64), 17);
    // 96 -> 3 blocks -> 102 bytes -> 26 words (104 bytes, 2 bytes of padding).
    assert_eq!(KvFormat::Q8_0.row_elems(96), 26);
    assert_eq!(KvFormat::Q8_0.row_bytes(96), 104);
    for nkt in [32usize, 64, 96, 128, 256, 1024] {
        let words = KvFormat::Q8_0.row_elems(nkt);
        assert!(words * WORD_BYTES >= KvFormat::Q8_0.payload_bytes(nkt));
        assert!(words * WORD_BYTES < KvFormat::Q8_0.payload_bytes(nkt) + WORD_BYTES);
    }
    // F32 and F16 keep one word per element: the region stays f32-shaped.
    for f in [KvFormat::F32, KvFormat::F16] {
        assert_eq!(f.row_elems(128), 128);
        assert!(!f.is_packed());
    }
}

#[test]
fn a_packed_region_is_at_least_three_times_smaller_than_f32() {
    // 1024 elements/cell, one f32 word each vs 32 blocks * 34 bytes = 1088 B.
    let f32_bytes = KvFormat::F32.row_bytes(1024);
    let q8_bytes = KvFormat::Q8_0.row_bytes(1024);
    assert_eq!(f32_bytes, 4096);
    assert_eq!(q8_bytes, 1088);
    assert!(
        q8_bytes * 3 <= f32_bytes,
        "Q8_0 must be at least 3x smaller than f32 ({q8_bytes} vs {f32_bytes})"
    );
}

#[test]
fn a_width_the_format_cannot_express_is_refused() {
    assert!(KvFormat::Q8_0.check_width(96).is_ok());
    let err = KvFormat::Q8_0.check_width(80).unwrap_err();
    assert!(err.contains("multiple of 32"), "{err}");
    assert!(KvFormat::Q8_0.check_width(0).is_err());
    // F32/F16 have no such constraint.
    assert!(KvFormat::F32.check_width(80).is_ok());
    assert!(KvFormat::F16.check_width(1).is_ok());
}

#[test]
fn the_cache_type_gate_is_strict_and_device_aware() {
    // Unset with a small model: f32 everywhere (the 7B-class auto policy is dims-keyed).
    assert_eq!(resolve(Device::Cpu, None, 24, 128).unwrap(), KvFormat::F32);
    assert_eq!(resolve(Device::Cuda, None, 24, 128).unwrap(), KvFormat::F32);
    assert_eq!(
        resolve(Device::Metal, None, 24, 128).unwrap(),
        KvFormat::F32
    );
    assert_eq!(
        resolve(Device::Cpu, Some(""), 24, 128).unwrap(),
        KvFormat::F32
    );
    // The three spellings.
    assert_eq!(
        resolve(Device::Cpu, Some("f32"), 24, 512).unwrap(),
        KvFormat::F32
    );
    assert_eq!(
        resolve(Device::Cpu, Some("q8_0"), 24, 512).unwrap(),
        KvFormat::Q8_0
    );
    assert_eq!(
        resolve(Device::Cuda, Some("f16"), 24, 512).unwrap(),
        KvFormat::F16
    );
    assert_eq!(
        resolve(Device::Metal, Some("f16"), 24, 512).unwrap(),
        KvFormat::F16
    );
    // f16 on CPU is the documented "CPU stays f32", not an error: the env var is
    // usually set for the GPU run a process may also do.
    assert_eq!(
        resolve(Device::Cpu, Some("f16"), 24, 512).unwrap(),
        KvFormat::F32
    );
    // #153: the unset 7B-class auto policy is the *engine's* format now, so the
    // device kernels and the region width read one answer. An explicit spelling
    // always wins over the dims.
    assert_eq!(
        resolve(Device::Cuda, None, 28, 1024).unwrap(),
        KvFormat::F16,
        "28x1024 = 28672 >= {AUTO_F16_MIN_KV_ELEMS}: the 7B class auto-selects f16"
    );
    assert_eq!(
        resolve(Device::Metal, None, 28, 1024).unwrap(),
        KvFormat::F16,
        "the auto policy is device-wide, not CUDA-only"
    );
    assert_eq!(
        resolve(Device::Cuda, None, 24, 341).unwrap(),
        KvFormat::F32,
        "24x341 = 8184 < {AUTO_F16_MIN_KV_ELEMS}: the 0.5B class stays f32"
    );
    assert_eq!(
        resolve(Device::Cpu, None, 28, 1024).unwrap(),
        KvFormat::F32,
        "the CPU never auto-selects f16"
    );
    assert_eq!(
        resolve(Device::Cuda, Some("f32"), 28, 1024).unwrap(),
        KvFormat::F32,
        "an explicit f32 overrides the auto policy"
    );
    // A packed format is accepted exactly where the kernels exist. Since #310
    // (mechanism A's native packed decode plus mechanism B's f32 staging) Metal
    // reads a packed region too, so `q8_0` resolves on Metal, CUDA (C4 S2b) and
    // the CPU; only a build that compiles no CUDA — or a non-macOS build with no
    // Metal backend — refuses it there. The answer comes from the registry, not
    // from a list here.
    #[cfg(target_os = "macos")]
    assert_eq!(
        resolve(Device::Metal, Some("q8_0"), 24, 512).unwrap(),
        KvFormat::Q8_0,
        "#310 gave Metal a packed read"
    );
    #[cfg(not(target_os = "macos"))]
    {
        let err = resolve(Device::Metal, Some("q8_0"), 24, 512).unwrap_err();
        assert!(err.contains("q8_0"), "{err}");
        assert!(err.contains(Device::Metal.name()), "{err}");
    }
    assert_eq!(
        resolve(Device::Cpu, Some("q8_0"), 24, 512).unwrap(),
        KvFormat::Q8_0,
        "the CPU reads packed cells (C4 S1+S2)"
    );
    #[cfg(feature = "cuda")]
    assert_eq!(
        resolve(Device::Cuda, Some("q8_0"), 24, 512).unwrap(),
        KvFormat::Q8_0,
        "C4 S2b gave CUDA a packed read"
    );
    #[cfg(not(feature = "cuda"))]
    {
        let err = resolve(Device::Cuda, Some("q8_0"), 24, 512).unwrap_err();
        assert!(err.contains("q8_0"), "{err}");
        assert!(err.contains(Device::Cuda.name()), "{err}");
    }
    // A typo is refused on every device (CUDA used to read it as f32).
    for dev in [Device::Cpu, Device::Cuda, Device::Metal] {
        let err = resolve(dev, Some("q8"), 24, 512).unwrap_err();
        assert!(err.contains("not a KV cache type"), "{err}");
    }
    assert!(
        resolve(Device::Cpu, Some("Q8_0"), 24, 512).is_err(),
        "case matters"
    );
}

/// #153: the auto policy is a pure function of the device and the dims, so two
/// engines in one process can resolve different formats without a global.
#[test]
fn the_auto_policy_is_a_pure_function_of_the_dims() {
    assert_eq!(auto_device_format(Device::Cuda, 28, 1024), KvFormat::F16);
    assert_eq!(auto_device_format(Device::Cuda, 24, 341), KvFormat::F32);
    assert_eq!(auto_device_format(Device::Cpu, 80, 1024), KvFormat::F32);
    assert_eq!(auto_device_format(Device::Metal, 28, 1024), KvFormat::F16);
}

#[test]
fn a_packed_cell_round_trips_within_the_q8_0_block_error() {
    let nkt = 128usize;
    let src: Vec<f32> = (0..nkt)
        .map(|i| ((i as f32) * 0.37).sin() * 2.5 + (i % 7) as f32 * 0.11)
        .collect();
    let mut cell = vec![0.0f32; KvFormat::Q8_0.row_elems(nkt)];
    pack_q8_0_cell(&mut cell, nkt, &src);
    let mut back = vec![0.0f32; nkt];
    unpack_q8_0_cells(&cell, nkt, 0, 1, &mut back);
    // Q8_0 keeps a per-block scale, so the error is half a step of that block's
    // own range (`|x - d*q| <= d/2`) in exact arithmetic, plus the stored scale's
    // own rounding amplified by the quantized magnitude: the scale is an f16
    // (half an ulp is `2^-12 * d`), and `|q| <= 127`, so the second term is at
    // most `127 * 2^-12 * d ~ 0.031 d`. 0.55 d covers both and still fails loudly
    // on a wrong layout, which is off by whole steps.
    for i in 0..nkt {
        let d = block_step(&src[(i / Q8_0_BLOCK) * Q8_0_BLOCK..]);
        let bound = d * 0.55 + 1e-5;
        assert!(
            (src[i] - back[i]).abs() <= bound,
            "element {i}: {} vs {} (bound {bound})",
            src[i],
            back[i]
        );
    }
}

/// The step (`d`) Q8_0 gives the block `block` starts.
fn block_step(block: &[f32]) -> f32 {
    let am = block[..Q8_0_BLOCK]
        .iter()
        .fold(0.0f32, |m, x| m.max(x.abs()));
    am / 127.0
}
