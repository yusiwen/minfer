//! The one CUDA per-tensor weight-registration rule shared by the qwen2 and qwen3
//! loaders (issue #167).
//!
//! Before #167 each loader carried its own copy of this dispatch, and the two copies had
//! already drifted twice: the qwen3 copy had neither the q4_K `W_dsc` plane registration
//! (r59, the rule #165 hardened into [`crate::q4k_dsc::q4k_dsc_plane_admitted`]) nor the
//! `TensorType::F16` branch (#141). A q4_K Qwen3 weight therefore fell back to
//! `mmq_raw_nb_bt`'s in-kernel scalar dsc decode, and an f16 Qwen3 model was dropped to
//! the CPU by the all-or-nothing `weights_on_cuda` gate even on a build with the f16
//! kernels. Both loaders now call [`register_cuda_weight`], so the type coverage has one
//! authority and cannot diverge by edit.
//!
//! The **decision** ([`cuda_weight_reg`]) is pure on purpose — no `CudaState`, no
//! environment. CI's CUDA job only compile-checks the device modules (a hosted runner has
//! no GPU), so a predicate that lived behind `#[cfg(feature = "cuda")]` would never be
//! *run* in CI; keeping it pure means the CPU job executes its tests. [`register_cuda_weight`]
//! is the thin `cuda`-gated half that carries the decision out.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::tensor::TensorType;

/// What the CUDA loader does with one weight tensor — the result of [`cuda_weight_reg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaWeightReg {
    /// No CUDA kernel dispatches this weight's type: leave the registry untouched. The
    /// loader still counts the tensor's bytes for the offload report (pre-#167 behaviour,
    /// kept), and the model-level `weights_on_cuda` type gate then refuses it, so the plan
    /// drops to CPU with the loader's loud line.
    None,
    /// Register the bytes as they are (`CudaState::register_weight`), plus any extras
    /// below. The plain raw registration is what every non-padded kernel reads.
    Raw {
        /// q8_0: also register the p32 split planes the decode MMVQ reads (doc 104).
        q80_p32: bool,
        /// This weight is not NB-BT-consumable, so the r60 mode-2 skip-write producer
        /// upstream is unsound for it: clear the process-wide flag (see
        /// `CudaState::clear_mmq_nb_bt_only`). Any such weight in any loaded model
        /// degrades the mix to mode 1.
        clear_nb_bt_only: bool,
        /// q4_K: also register the r59 `W_dsc` f32-pair plane, under
        /// [`crate::q4k_dsc::q4k_dsc_plane_admitted`] and the r59 dispatch gates.
        q4k_dsc: bool,
    },
    /// q6_K: register the padded 224-byte repack so the matmul kernel can use aligned
    /// uint4 weight loads (7e②).
    Q6kPadded,
}

/// The shared CUDA weight-registration decision.
///
/// * `ttype` — the tensor's GGUF type.
/// * `rank` — its number of stored dimensions (`ggml_n_dims`): trailing 1s dropped, at
///   least 1. Only the F32 arm reads it (a 2-D f32 tensor is a matmul weight; a 1-D one is
///   a norm/bias).
/// * `payload_bytes` — the sliced payload length, for the q4_K dsc payload contract.
/// * `od`, `id` — the GGUF geometry: `ne[1]` (output rows) and `ne[0]` (input width).
/// * `dsc_gates_on` — the r59 dispatch gates (`MINFER_MMQ_RAW_NB` + `MINFER_MMQ_A_TRANSPOSE`
///   + `MINFER_MMQ_Q4K_DSC != "0"`). Passed in, not read here, so this stays environment-free.
///
/// The quantized set mirrors the loader `matches!` it replaces; F16 and F32 are separate
/// arms because neither is a block-quant payload and neither may reach the dsc plane.
pub(crate) fn cuda_weight_reg(
    ttype: TensorType,
    rank: usize,
    payload_bytes: usize,
    od: usize,
    id: usize,
    dsc_gates_on: bool,
) -> CudaWeightReg {
    let quantized = matches!(
        ttype,
        TensorType::Q4_0
            | TensorType::Q4_1
            | TensorType::Q4_K
            | TensorType::Q5_0
            | TensorType::Q5_1
            | TensorType::Q5_K
            | TensorType::Q6_K
            | TensorType::Q8_0
    );
    if quantized {
        if ttype == TensorType::Q6_K {
            return CudaWeightReg::Q6kPadded;
        }
        return CudaWeightReg::Raw {
            q80_p32: ttype == TensorType::Q8_0,
            clear_nb_bt_only: !matches!(ttype, TensorType::Q4_K | TensorType::Q6_K),
            // `od % 2 == 0` is the r59 row-pair cp.async staging requirement, not a
            // payload property, which is why it is here and not in the payload contract.
            // `q4k_dsc_plane_admitted` carries the type and the exact-length payload gate.
            q4k_dsc: dsc_gates_on
                && od % 2 == 0
                && crate::q4k_dsc::q4k_dsc_plane_admitted(ttype, payload_bytes, od, id),
        };
    }
    if ttype == TensorType::F16 {
        // #141: f16 weights register raw (2 B/element) and the device matmul kernel
        // converts in-register — no f32 copy, so the memory the f16 file exists to save
        // is actually saved. Deliberately not in the quantized set above: that arm's dsc
        // plane gate would build a plane out of f16 bytes no kernel can read.
        return CudaWeightReg::Raw {
            q80_p32: false,
            // r60: f16 is not NB-BT-consumable — its GEMM reads the f32 activations, so
            // a mode-2 skip-write producer upstream would feed it a dead buffer.
            clear_nb_bt_only: true,
            q4k_dsc: false,
        };
    }
    if ttype == TensorType::F32 {
        return CudaWeightReg::Raw {
            q80_p32: false,
            // r60: a 2-D F32 weight is a matmul weight (norms/biases are 1-D) — its GEMM
            // reads the f32 A directly, so a mode-2 skip-write producer upstream would
            // feed it a dead buffer.
            //
            // #167 audit: the pre-extraction code spelled this `tensor.shape.len() == 2`,
            // but `Tensor::shape` is a `[i64; 4]`, so the test was *always false* in both
            // loaders and the flag was never cleared for a 2-D f32 weight. The intent is
            // restored here by taking the real rank. It can only disable the mode-2 MMQ
            // optimization (never change a result), and no model in the gate set has a
            // 2-D f32 matmul weight — but the dead test is not carried forward.
            clear_nb_bt_only: rank == 2,
            q4k_dsc: false,
        };
    }
    if ttype == TensorType::BF16 {
        // #208 (the CUDA half): bf16 registers raw, exactly like f16 above — the
        // 2 B/element words stay on the device and `bf16_f32_matmul_vec` /
        // `_scalar` / `embed_rows_bf16` promote in-register with the exact
        // `f32::from_bits(bits << 16)` decode. No f32 copy at registration, so the
        // memory a bf16 file exists to save is actually saved. Like f16 it is
        // deliberately outside the quantized `matches!` above (that arm's dsc-plane
        // gate has no type check and would expand bf16 bytes into a plane no kernel
        // reads) and it clears the NB-BT-only flag (its GEMM reads the f32
        // activations, so a mode-2 skip-write producer upstream would feed it a dead
        // buffer).
        //
        // **Metal is the other half and is deliberately NOT here.** The Metal
        // registration arm lives inline in each loader (`matches!(ttype, F32 | F16)`)
        // and #208's Metal half is a separate, later delegation, so a bf16 GGUF on a
        // Metal build still drops to the CPU loudly. Flipping it needs
        // `kernel_bf16_f32_matmul` + `kernel_get_rows_bf16` on that side.
        return CudaWeightReg::Raw {
            q80_p32: false,
            clear_nb_bt_only: true,
            q4k_dsc: false,
        };
    }
    CudaWeightReg::None
}

/// Carry out [`cuda_weight_reg`] against the process-wide CUDA registry — the one function
/// both loaders call for every weight tensor whose block the E5 offload plan puts on the
/// device.
///
/// `shape` is the `[i64; 4]` GGUF shape the loader built; `rank` is its stored dimension
/// count. Registration order (raw, then extras, then the flag) has no cross-effect: the
/// extras read the raw entry by name and the flag is an atomic. `NoCuda` is not a case
/// here — the caller only calls with a live `CudaState`.
#[cfg(feature = "cuda")]
pub(crate) fn register_cuda_weight(
    cuda: &crate::cuda::CudaState,
    reg_name: &str,
    ttype: TensorType,
    data: &[u8],
    shape: &[i64; 4],
    rank: usize,
) {
    // #171: the weight registrar is a failure-injection chokepoint
    // (`MINFER_TEST_CALL_FAIL=register_weight`) and an observable site. A `()`
    // return leaves a panic as the only failure channel, so it uses the panic
    // form.
    crate::testfail::note_checked("register_weight");
    crate::testfail::guard_panic("register_weight");
    // Issue #188: registration no longer takes the #185 device-entry guard.
    // `CudaState::register_weight` queues the H2D copy on the *context* stream and
    // waits on that stream (never a blocking `cudaMemcpy` on the legacy default
    // stream), the activation scratches are per stream, and the capture mode is
    // thread-local, so a registration may run while another engine captures on its
    // own stream. `register_cuda_weight`'s panic channel above is unchanged.
    let (id, od) = (shape[0] as usize, shape[1] as usize);
    let dsc_gates_on = crate::cuda::CudaState::mmq_gate_on("MINFER_MMQ_RAW_NB")
        && crate::cuda::CudaState::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE")
        && std::env::var("MINFER_MMQ_Q4K_DSC").as_deref() != Ok("0");
    match cuda_weight_reg(ttype, rank, data.len(), od, id, dsc_gates_on) {
        CudaWeightReg::None => {}
        CudaWeightReg::Q6kPadded => {
            cuda.register_weight_q6k_padded(reg_name, data, od, id);
        }
        CudaWeightReg::Raw {
            q80_p32,
            clear_nb_bt_only,
            q4k_dsc,
        } => {
            cuda.register_weight(reg_name, data);
            if q80_p32 {
                cuda.register_weight_q80_p32(reg_name, data, od, id);
            }
            if q4k_dsc {
                cuda.register_weight_q4k_dsc(reg_name, data, od, id);
            }
            if clear_nb_bt_only {
                cuda.clear_mmq_nb_bt_only();
            }
        }
    }
}

#[cfg(test)]
mod tests;
