// int8 MMQ prefill GEMM (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // R1: int8 MMQ prefill GEMM — q8_0-quantized activations (pad40 blocks
    // with the per-block int sum at offset 36) × raw quantized weights, tiled
    // mma.m16n8k32/m16n8k16 (s8) with per-k-block scale rescale. type_id as
    // in launch_dequant_f16; q6_stride = 210 raw / 224 padded (Q6_K only).
    // Requires id % 32 == 0 and sm_80+ (int8 mma; sm_75 falls back).
    // #147: 1 = launched and accepted, 0 = refused (named at the site).
    pub(crate) fn launch_mmq_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        q6_stride: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // P6: raw-byte staging MMQ (q4_K, whole 256-k super-blocks).
    pub(crate) fn launch_mmq_raw_wide_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 Direction-A: NB raw-nibble MMQ (q4_K, KD=8 native, 64x128, 2 blk/SM).
    // Returns 1 when it ran (KD=8), 0 on clean fallback (KD!=8 / smem / regs).
    pub(crate) fn launch_mmq_raw_nb_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // r59: `w_dsc` selects the DSC=true instantiation (registration-time
    // W_dsc f32-pair plane; null = in-kernel scalar decode, DSC=false).
    pub(crate) fn launch_mmq_raw_nb_bt_nt(
        type_id: i32,
        w: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
        cpart: *mut f32,
        ksplit: i32,
    ) -> i32;
    // P6 r38: q6_K BT kernel — the same bulk-LDG->STS A staging as r34, but the
    // B weight is EXPANDED to centered int8 (256 B/row) in staging and the mma is
    // m16n8k16 (KSPLIT=2) with a per-16-sub dsc rescale. r53: `w_exp` selects
    // the pre-expanded-B cp.async instantiation (null = r41 in-kernel expand).
    // Returns 1 on KD=8, 0 on clean fallback.
    pub(crate) fn launch_mmq_raw_nb_bt_q6k_nt(
        type_id: i32,
        w: *const u8,
        w_exp: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        bstride: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
        cpart: *mut f32,
        ksplit: i32,
    ) -> i32;
    // #147: 1 = launched and accepted, 0 = refused (the opt-in or the launch
    // failed and was named at the site); the caller turns a 0 into an `Err`.
    pub(crate) fn launch_mmq_raw_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
}

impl CudaState {
    /// R1: int8 MMQ prefill GEMM — quantize activations to q8_0 (pad40
    /// blocks; the quantize kernel also emits the per-block int sum used by
    /// the min-term correction), then one tiled mma.m16n8k32 (s8) launch per
    /// weight (see mmq_nt_kernel). Caller guarantees nt >= 16, id % 32 == 0,
    /// and a supported quant type. The q8 scratch follows the same
    /// grow-on-demand lifecycle as the f16 path's buf_f16_x: the 3-run
    /// capture protocol sizes it before the capture window opens.
    /// doc 92 auto-ksplit, parameterized (T2, plan §6.1): the M-starve gate
    /// stays `nt <= 64` (ntb == 1 — a single M tile row) and the resident-
    /// block target becomes `max(256, 2*SM)`. On GB10 (48 SMs) that is
    /// exactly the calibrated 256 — zero behavior change — while larger SM
    /// counts scale the target proportionally. `MINFER_MMQ_KSPLIT_TARGET`
    /// still overrides everything (explicit human choice beats the formula).
    fn auto_ksplit(&self, nt: usize, nbt_y: usize, nktile: usize) -> usize {
        if nt > 64 || nbt_y == 0 || nktile <= 1 {
            return 1;
        }
        let target: usize = std::env::var("MINFER_MMQ_KSPLIT_TARGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (256usize).max(2 * self.sm_count as usize));
        let want = (target + nbt_y - 1) / nbt_y;
        want.clamp(2, nktile)
    }

    pub fn prefill_mmq(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
        ksplit_req: i32,
    ) -> Result<(), String> {
        let type_id = match ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            other => return Err(format!("cuda: prefill MMQ got unsupported type {other:?}")),
        };
        // r49: the native pad40 buffer is sized upfront so an OOM surfaces here
        // (before any GEMM launch) instead of as a null deref in the q4_K
        // fallback's final `launch_mmq_raw_nt`. The fallbacks re-derive it via
        // `mmq_quantize_native` (which dedups); this call only guards OOM.
        if Self::get_or_grow(&self.buf_q8_prefill, nt * (id / 32) * 40).is_null() {
            return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
        }
        // Q6_K reads whichever layout the weight was registered with
        // (224-byte padded 7e② repack or raw 210); all others are raw.
        let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
            224
        } else {
            210
        };
        let stream = self.stream();
        // P6 r38: q6_K on a raw-byte BT-style kernel (expanded centered-int8 B,
        // m16n8k16 KSPLIT=2). Same A prepass as r34; the B path is q6_K-specific.
        // Gated on MINFER_MMQ_Q6K_NB + MINFER_MMQ_RAW (r60: default-on,
        // "0" opts out; the q6k path always uses the transposed-A raster).
        // On any cap/mismatch the launcher returns 0 -> clean fall through
        // to the generic mmq_nt<7,2>.
        if type_id == 7
            && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
            && Self::mmq_gate_on("MINFER_MMQ_RAW")
            && (id / 32) % 8 == 0
        {
            let nchunk = (id / 32) as i32;
            let ntb = ((nt as i64 + 63) / 64) as i32;
            unsafe {
                // r49: A-quantize prepass via the consecutive-window cache —
                // a same-A (>q/k/v, gate/up) matmul reuses qa8g/sdag without a
                // fresh quantize launch.
                let (qa8g, sdag) = self.mmq_quantize_transposed(
                    x as *const f32,
                    id as i32,
                    nt as i32,
                    nchunk,
                    ntb,
                    stream,
                );
                // r53: pre-expanded B plane lookup by the padded weight's
                // device pointer (null on miss -> launcher selects the r41
                // in-kernel-expand instantiation).
                let w_exp = self
                    .q6k_exp
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                // r56 (Session E item 2b): the precomputed dsc f32-pair plane
                // (null on miss -> the r41 scalar dsc path in-kernel).
                let w_dsc = self
                    .q6k_dsc
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                // doc 92: same K-split contract as the q4_K path (T2: the
                // target is SM-count-parameterized — see auto_ksplit).
                let ksplit: usize = if ksplit_req < 0 {
                    self.auto_ksplit(nt, (od + 127) / 128, (nchunk as usize + 1) / 2)
                } else {
                    ksplit_req.max(1) as usize
                };
                let cpart = if ksplit > 1 {
                    Self::get_or_grow(&self.buf_mmq_ksplit, ksplit * nt * od * 4) as *mut f32
                } else {
                    std::ptr::null_mut()
                };
                if qa8g != 0
                    && sdag != 0
                    && (ksplit == 1 || !cpart.is_null())
                    && launch_mmq_raw_nb_bt_q6k_nt(
                        type_id,
                        wptr as *const u8,
                        w_exp as *const u8,
                        w_dsc as *const u8,
                        qa8g as *const u8,
                        sdag as *const u8,
                        out as *mut f32,
                        nt as i32,
                        od as i32,
                        id as i32,
                        nchunk,
                        block_stride,
                        stream,
                        8,
                        cpart,
                        ksplit as i32,
                    ) == 1
                {
                    if std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1") {
                        // r54: name WHY the r41 in-kernel expand is running —
                        // "exp=off" is the intentional MINFER_MMQ_Q6K_EXP=0
                        // switch; "fallback!" means a W_exp build was expected
                        // (padded weight, EXP gate on) but the map missed
                        // (alloc/upload failure or a registration bug). Raw
                        // 210-B weights never get a plane -> kept unqualified.
                        let b = if !w_exp.is_null() {
                            "W_exp-cp.async"
                        } else if !padded_q6k {
                            "in-kernel-expand"
                        } else if std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() == Ok("0") {
                            "in-kernel-expand(exp=off)"
                        } else {
                            "in-kernel-expand(fallback!)"
                        };
                        // r56: name the A/dsc staging paths too (liveness
                        // check per the r53 lesson — a fallback-correct fast
                        // path needs a visible label, parity cannot see it).
                        let a = if !w_dsc.is_null() {
                            "A=cp.async DSC=f32-plane"
                        } else {
                            "A=cp.async DSC=scalar"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT q6_K kernel active \
                             (r56 {}, r54 B={}, r39 KDR=2 double-buffer, \
                             A-transpose)",
                            a, b
                        );
                    }
                    return Ok(());
                }
            }
        }
        // P6: raw-byte staging variant (q4_K, whole super-blocks only).
        // Same quantized activations; the GEMM stages RAW weight bytes via
        // cp.async and dequants in registers (docs/CUDA_OPTIMIZATION.md).
        if type_id == 5 && Self::mmq_gate_on("MINFER_MMQ_RAW") && (id / 32) % 8 == 0 {
            let kd: i32 = std::env::var("MINFER_MMQ_RAW_KD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            let wide = std::env::var("MINFER_MMQ_RAW_WIDE").as_deref() == Ok("1");
            let nb = Self::mmq_gate_on("MINFER_MMQ_RAW_NB");
            let nb_debug = std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1");
            let at = Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE");
            unsafe {
                // P6 r34: relocate the A-side layout transform out of the mma
                // kernel into a quantize-transpose prepass (llama.cpp's design).
                // Under MINFER_MMQ_A_TRANSPOSE=1 the activations are emitted
                // PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the NB kernel's A
                // staging is a bulk LDG->STS; the native q8 buffer is only
                // filled on the (rare) bb-bt fallback below.
                let mut nb_ok = false;
                if nb && at && kd == 8 {
                    let nchunk = (id / 32) as i32;
                    let ntb = ((nt as i64 + 63) / 64) as i32;
                    // r49: cache-backed transposed A-quantize prepass (reuses
                    // the previous same-A matmul's qa8g/sdag when consecutive).
                    let (qa8g, sdag) = self.mmq_quantize_transposed(
                        x as *const f32,
                        id as i32,
                        nt as i32,
                        nchunk,
                        ntb,
                        stream,
                    );
                    // r59: the q4_K W_dsc f32-pair plane (null on miss ->
                    // the DSC=false in-kernel scalar decode instantiation).
                    let w_dsc = self
                        .q4k_dsc
                        .lock()
                        .unwrap()
                        .get(&(wptr as usize))
                        .map(|cp| cp.0)
                        .unwrap_or(std::ptr::null_mut());
                    // doc 92: K-split the K range across grid.z slots when
                    // the grid is M-starved (ntb == 1 => only od/128 blocks).
                    // ksplit_req < 0 = auto (small-M gate): target >= 256
                    // resident blocks (about 2/SM); 1 = the unsplit path
                    // (default prefill, bitwise unchanged).
                    // T2: the auto formula moved into auto_ksplit (shared
                    // with the q6_K NB path; SM-count-parameterized target).
                    let ksplit: usize = if ksplit_req < 0 {
                        self.auto_ksplit(nt, (od + 127) / 128, (nchunk as usize + 7) / 8)
                    } else {
                        ksplit_req.max(1) as usize
                    };
                    let cpart = if ksplit > 1 {
                        Self::get_or_grow(&self.buf_mmq_ksplit, ksplit * nt * od * 4) as *mut f32
                    } else {
                        std::ptr::null_mut()
                    };
                    nb_ok = qa8g != 0
                        && sdag != 0
                        && (ksplit == 1 || !cpart.is_null())
                        && launch_mmq_raw_nb_bt_nt(
                            type_id,
                            wptr as *const u8,
                            w_dsc as *const u8,
                            qa8g as *const u8,
                            sdag as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            nchunk,
                            stream,
                            kd,
                            cpart,
                            ksplit as i32,
                        ) == 1;
                    if nb_ok && nb_debug {
                        // r59: name the dsc staging path (liveness check per
                        // the r53 lesson — a fallback-correct fast path needs
                        // a visible label, parity cannot see it). "fallback!"
                        // means a plane was expected (gate on) but the map
                        // missed (alloc/upload failure or registration bug).
                        let d = if !w_dsc.is_null() {
                            "DSC=f32-plane"
                        } else if std::env::var("MINFER_MMQ_Q4K_DSC").as_deref() == Ok("0") {
                            "DSC=in-kernel(dsc=off)"
                        } else {
                            "DSC=in-kernel(fallback!)"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT kernel active \
                             (KD=8, A-transpose, r59 {d})"
                        );
                    }
                }
                if !nb_ok {
                    // r49: cache-backed native A-quantize prepass. The helper
                    // returns 0 (no buffer) on OOM; the launchers below only
                    // run when a buffer is present (the original unconditional
                    // `q8` came from a pre-call get_or_grow, now inside the
                    // helper).
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    // Direction-A NB raw-nibble kernel is KD=8-native; it
                    // activates only under the full MMQ gate set. launcher
                    // returns 0 on KD!=8 or smem/reg cap failure -> clean
                    // fallback to the wide/narrow raw path below.
                    nb_ok = q8 != 0
                        && nb
                        && launch_mmq_raw_nb_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if nb_ok && nb_debug {
                        eprintln!("minfer/cuda: mmq raw NB kernel active (KD=8)");
                    }
                }
                if !nb_ok {
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    if q8 == 0 {
                        // r52: mmq_quantize_native refuses a mode-2 dead-write
                        // A (and plain q8 OOM is pre-checked at fn entry) —
                        // never fall through to a GEMM on a null/garbage A.
                        return Err("cuda: prefill MMQ: A-quantize unavailable (mode-2 \
                             dead-write A refused or q8 OOM); MINFER_MMQ_A_FUSE=2 \
                             window violated"
                            .to_string());
                    }
                    let wide_ok = q8 != 0
                        && wide
                        && launch_mmq_raw_wide_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if !wide_ok
                        && launch_mmq_raw_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 0
                    {
                        // #147: the terminal raw-narrow launcher refused the
                        // launch (its dynamic-smem opt-in failed or the launch
                        // itself errored). It named the call, the instantiation
                        // and `cudaGetErrorName` on stderr — do not let the
                        // graph continue on an unwritten output.
                        return Err("cuda: prefill MMQ: launch_mmq_raw_nt refused the launch \
                                    (see the named CUDA error on stderr); no kernel ran"
                            .to_string());
                    }
                }
            }
            return Ok(());
        }
        unsafe {
            let q8 = self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
            if q8 == 0 {
                return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
            }
            if launch_mmq_nt(
                type_id,
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                nt as i32,
                od as i32,
                id as i32,
                block_stride,
                stream,
            ) == 0
            {
                // #147: same contract as launch_mmq_raw_nt above.
                return Err(
                    "cuda: prefill MMQ: launch_mmq_nt refused the launch (see the \
                            named CUDA error on stderr); no kernel ran"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}
