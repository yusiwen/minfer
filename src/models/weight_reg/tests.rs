//! `#[cfg(test)] mod tests` for `src/models/weight_reg.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// #167 acceptance: an f16 weight is registered **raw** (the F16 arm exists at all),
/// and it clears the NB-BT-only flag because no NB-BT producer can feed its
/// f32-activation GEMM. This is the decision half of the qwen3 f16 gap: before the
/// shared rule, the qwen3 loader's `else if` chain ended at F32 and answered `None`.
#[test]
fn f16_weights_are_registered_raw_and_clear_the_nb_bt_flag() {
    // A Qwen3-0.6B f16 `ffn_gate` is [in=1024, out=3072] → id=ne[0]=1024, od=ne[1]=3072.
    let (id, od) = (1024usize, 3072usize);
    assert_eq!(
        cuda_weight_reg(TensorType::F16, 2, od * id * 2, od, id, true),
        CudaWeightReg::Raw {
            q80_p32: false,
            clear_nb_bt_only: true,
            q4k_dsc: false,
        },
        "an f16 weight must register raw and clear the NB-BT-only flag"
    );
    // Positive control: the same call for a type no kernel reads answers None, so the
    // assertion above is not `Raw` for every input.
    assert_eq!(
        cuda_weight_reg(TensorType::I8, 2, 0, od, id, true),
        CudaWeightReg::None
    );
}

/// #167: the q4_K dsc plane is decided by the shared
/// [`crate::q4k_dsc::q4k_dsc_plane_admitted`] rule **and** the two r59 gates — each
/// half independently refused, so the plane gate cannot pass for the wrong reason.
#[test]
fn q4k_weights_register_the_dsc_plane_only_under_every_gate() {
    let (id, od) = (1024usize, 3072usize);
    let bytes = crate::q4k_dsc::q4k_dsc_payload_bytes(od, id).expect("whole super-blocks");
    let dsc = |t, rank, b, od, id, gates| match cuda_weight_reg(t, rank, b, od, id, gates) {
        CudaWeightReg::Raw { q4k_dsc, .. } => q4k_dsc,
        other => panic!("expected Raw, got {other:?}"),
    };
    // positive: q4_K + exact payload + even od + the gates on
    assert!(dsc(TensorType::Q4_K, 2, bytes, od, id, true));
    // each gate refused on its own
    assert!(
        !dsc(TensorType::Q4_K, 2, bytes, od, id, false),
        "r59 gates off"
    );
    assert!(
        !dsc(TensorType::Q4_K, 2, bytes, od + 1, id, true),
        "an odd od has no row pair to stage"
    );
    // the type gate: q4_0 has q4_K's exact bytes-per-element ratio, so only the type
    // gate can refuse a same-length q4_0 payload
    assert_eq!(
        od * (id / 32) * 18,
        bytes,
        "q4_0 length == q4_K length here"
    );
    assert!(
        !dsc(TensorType::Q4_0, 2, bytes, od, id, true),
        "the type gate is the only thing that can refuse q4_0"
    );
    // the payload gate: a q8_0-length payload is refused even at the right type
    let q80 = od * (id / 32) * 34;
    assert!(q80 > bytes);
    assert!(!dsc(TensorType::Q4_K, 2, q80, od, id, true));
    // F16 must never reach the plane even if its length happened to match
    assert!(!dsc(TensorType::F16, 2, bytes, od, id, true));
}

/// The quantized arm keeps the pre-#167 dispatch exactly: q6_K is the padded repack,
/// q8_0 adds the p32 split planes, every other quant is raw and clears the flag, and
/// q4_K is the one type that does **not** clear it.
#[test]
fn the_quantized_arm_matches_the_pre_167_dispatch() {
    let (id, od) = (1024usize, 3072usize);
    assert_eq!(
        cuda_weight_reg(TensorType::Q6_K, 2, 0, od, id, true),
        CudaWeightReg::Q6kPadded
    );
    for t in [
        TensorType::Q4_0,
        TensorType::Q4_1,
        TensorType::Q5_0,
        TensorType::Q5_1,
        TensorType::Q5_K,
        TensorType::Q8_0,
    ] {
        let got = cuda_weight_reg(t, 2, 0, od, id, false);
        let want = CudaWeightReg::Raw {
            q80_p32: t == TensorType::Q8_0,
            clear_nb_bt_only: true,
            q4k_dsc: false,
        };
        assert_eq!(got, want, "{t:?}");
    }
    assert_eq!(
        cuda_weight_reg(TensorType::Q4_K, 2, 0, od, id, false),
        CudaWeightReg::Raw {
            q80_p32: false,
            clear_nb_bt_only: false,
            q4k_dsc: false,
        },
        "q4_K is NB-BT-consumable: it must not clear the flag"
    );
}

/// #167 audit: the 1-D f32 norms/biases are registered but must **not** clear the
/// NB-BT-only flag (they are not matmul weights), while a 2-D f32 weight must. The
/// pre-extraction `shape.len() == 2` was always false on a `[i64; 4]`; this pins the
/// corrected rank rule at both ranks so a regression to "never" or "always" fails.
#[test]
fn only_a_2d_f32_weight_clears_the_nb_bt_flag() {
    let clear = |rank| match cuda_weight_reg(TensorType::F32, rank, 0, 1024, 1024, true) {
        CudaWeightReg::Raw {
            clear_nb_bt_only, ..
        } => clear_nb_bt_only,
        other => panic!("expected Raw, got {other:?}"),
    };
    assert!(!clear(1), "a 1-D norm/bias is not a matmul weight");
    assert!(clear(2), "a 2-D f32 weight is a matmul weight");
}

/// The pure rule answers by type, not by accident: the three engine-relevant answers
/// for one geometry are three different variants.
#[test]
fn the_rule_distinguishes_the_three_registration_kinds() {
    let (id, od) = (1024usize, 3072usize);
    assert_eq!(
        cuda_weight_reg(TensorType::Q6_K, 2, 0, od, id, true),
        CudaWeightReg::Q6kPadded
    );
    assert!(matches!(
        cuda_weight_reg(TensorType::Q8_0, 2, 0, od, id, true),
        CudaWeightReg::Raw { .. }
    ));
    assert_eq!(
        cuda_weight_reg(TensorType::I8, 2, 0, od, id, true),
        CudaWeightReg::None
    );
}
