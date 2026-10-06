// Metal backend L2 op → encoding table (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split. Every method here is an inherent
// method of `MpsCommandBuffer`, whose type and private fields are declared in
// the parent `src/metal.rs`; the shared dispatch primitives
// (`set_params`/`dispatch_*`/`gemm_dispatch`/`trace_op`) stay there too, so the
// siblings reach them through the parent with no `pub(super)`.

use super::*;

#[cfg(target_os = "macos")]
impl MpsCommandBuffer<'_> {
    pub fn quant_matmul_f32_on_gpu_buf(
        &self,
        wb: &MetalBuffer,
        w_off: u64,
        ttype: TensorType,
        x: &MetalBuffer,
        x_off: u64,
        out: &MetalBuffer,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        self.trace_op("matmul");
        // GPU safety (M1): the K-quant (super-block) kernels index weights by
        // K/256 super-blocks (floor). A non-256-aligned id silently drops the
        // remainder (wrong results, not a fault) — refuse rather than risk it.
        if matches!(
            ttype,
            TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K
        ) && id % 256 != 0
        {
            gpu_abort(&format!(
                "matmul input dim id={id} is not 256-aligned for {ttype:?} (K-quant kernels use K/256 super-block floor)"
            ));
        }
        match ttype {
            TensorType::Q8_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q8_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q8_0_f32_multi
                        } else {
                            &self.state.pl_q8_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    const NW: u64 = 32;
                    const NSG: u64 = 4;
                    const NR0: u64 = 2;
                    const TG_MEM: u64 = NW * NR0 * std::mem::size_of::<f32>() as u64; // 256 bytes
                    unsafe {
                        self.enc
                            .setThreadgroupMemoryLength_atIndex((TG_MEM) as usize, (0) as usize)
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 1) / 2) as u64, grid_y, NW, NSG);
                }
            }
            TensorType::Q4_K | TensorType::Q6_K => {
                // Q6_K has a simdgroup GEMM (super-block); Q4_K still falls back
                // to the scalar f32 multi (no Q4_K in the shipped 0.5B K_M models).
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    // both Q4_K and Q6_K have simdgroup GEMMs
                    let pl = if ttype == TensorType::Q6_K {
                        &self.state.pl_q6_k_mm_f32
                    } else {
                        &self.state.pl_q4_k_mm_f32
                    };
                    self.gemm_dispatch(pl, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    let pl: &MetalComputePipelineState = if ttype == TensorType::Q4_K {
                        if nt > 1 {
                            &self.state.pl_q4_k_f32_multi
                        } else {
                            &self.state.pl_q4_k_f32
                        }
                    } else {
                        if nt > 1 {
                            &self.state.pl_q6_k_f32_multi
                        } else {
                            &self.state.pl_q6_k_f32
                        }
                    };
                    self.enc.setComputePipelineState(&**(pl));
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    // Q6_K/Q4_K: llama's kernel_mul_mv_q6_K/q4_K_f32_impl use
                    // TG(32, nsg=2); the stride-2 (q6_K) / stride-4 (q4_K) thread
                    // layout keeps all threads busy for small id (nb super-blocks),
                    // unlike the old stride-64 scalar loop.
                    if ttype == TensorType::Q6_K || ttype == TensorType::Q4_K {
                        self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 32, 2);
                    } else {
                        self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 64, 1);
                    }
                }
            }
            TensorType::Q4_1 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q4_1_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q4_1_f32_multi
                        } else {
                            &self.state.pl_q4_1_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_0_f32_multi
                        } else {
                            &self.state.pl_q5_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_1 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_1_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_1_f32_multi
                        } else {
                            &self.state.pl_q5_1_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_K => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_k_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_k_f32_multi
                        } else {
                            &self.state.pl_q5_k_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q4_0 => {
                // Prefill uses the simdgroup GEMM (faithful llama.cpp port, float
                // accumulation). MINFER_GEMM=0 disables it (f32 multi fallback) for
                // A/B comparison. GEMM wins for nt >= ~16 (fixed dispatch overhead
                // dominates for tiny prefills).
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q4_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q4_0_f32_multi
                        } else {
                            &self.state.pl_q4_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::F16 => {
                // #164: f16 weight rows × f32 activations; the weights stay 2
                // B/element on the device (no registration-time f32 copy). A
                // second 2 B/element dtype (bf16, #208) adds its own arm here.
                self.enc.setComputePipelineState(&*self.state.pl_f16_f32);
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (1) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
                };
                let mm_p = [od as i32, id as i32, nt as i32];
                unsafe {
                    self.enc.setBytes_length_atIndex(
                        NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                            .unwrap(),
                        (12) as usize,
                        (3) as usize,
                    )
                };
                // NR0*NSG = 8 rows per threadgroup, one 64-thread threadgroup.
                self.dispatch_2d(((od + 7) / 8) as u64, 1, 32, 2);
            }
            _ => {
                self.enc.setComputePipelineState(
                    &**(if nt > 1 {
                        &self.state.pl_q4_0_f32_multi
                    } else {
                        &self.state.pl_q4_0_f32
                    }),
                );
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (1) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
                };
                let mm_p = [od as i32, id as i32, nt as i32];
                unsafe {
                    self.enc.setBytes_length_atIndex(
                        NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                            .unwrap(),
                        (12) as usize,
                        (3) as usize,
                    )
                };
                let grid_y = if nt > 1 { 1 } else { nt as u64 };
                self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
            }
        }
    }

    /// GPU embedding lookup: dequantize Q4_0 embedding rows for nt token ids.
    /// Writes f32 hidden state [nt][ne] to dst (buf_hidden).
    pub fn embed_tokens_gpu(
        &self,
        wb: &MetalBuffer,
        w_off: u64,
        ids: &MetalBuffer,
        dst: &MetalBuffer,
        ne: usize,
        nt: usize,
        ttype: TensorType,
    ) {
        self.trace_op("embed");
        let (pl, nb) = match ttype {
            TensorType::Q4_0 => (&self.state.pl_get_rows_q4_0, ne / 32),
            TensorType::Q4_1 => (&self.state.pl_get_rows_q4_1, ne / 32),
            TensorType::Q5_0 => (&self.state.pl_get_rows_q5_0, ne / 32),
            TensorType::Q5_1 => (&self.state.pl_get_rows_q5_1, ne / 32),
            TensorType::Q8_0 => (&self.state.pl_get_rows_q8_0, ne / 32),
            TensorType::Q4_K => (&self.state.pl_get_rows_q4_k, (ne / 256) * 16),
            TensorType::Q6_K => (&self.state.pl_get_rows_q6_k, (ne / 256) * 16),
            TensorType::Q5_K => (&self.state.pl_get_rows_q5_k, (ne / 256) * 16),
            // #164: f16 embedding rows decode one element per thread (nb = ne).
            TensorType::F16 => (&self.state.pl_get_rows_f16, ne),
            _ => unreachable!("embed_tokens_gpu called with unsupported type {ttype:?}"),
        };
        self.enc.setComputePipelineState(&**(pl));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(ids)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(ne as i32));
        self.set_params(4, &(nt as i32));
        self.dispatch_1d((nt * nb) as u64, 256);
    }

    /// Generic f32 row selection: out[t] = x[ids[t]] (graph n_out tail rows).
    pub fn get_rows_f32(
        &self,
        x: &MetalBuffer,
        ids: &MetalBuffer,
        out: &MetalBuffer,
        ne: usize,
        nt: usize,
    ) {
        self.trace_op("get_rows_f32");
        self.enc
            .setComputePipelineState(&*self.state.pl_get_rows_f32);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(ids)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(ne as i32));
        self.dispatch_2d(nt as u64, ne as u64, 1, 1);
    }

    /// RMSNorm: y = x * rsqrt(mean(x²)+eps) * w
    pub fn rms_norm(
        &self,
        x: &MetalBuffer,
        w: Option<&MetalBuffer>,
        w_off: u64,
        y: &MetalBuffer,
        d: usize,
        n: usize,
        eps: f32,
        off: u64,
        y_off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc.setComputePipelineState(&*self.state.pl_rms_norm);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (off) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(w.unwrap_or(y))),
                (w_off) as usize,
                (1) as usize,
            )
        }; // dummy if no weight
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (2) as usize)
        };
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        self.dispatch_2d(n as u64, 1, 32, 1);
    }

    /// RMSNorm with a 256-thread multi-simdgroup kernel (P1 2026-08-10, llama
    /// transcription). Same math as rms_norm but the threadgroup is 256 threads
    /// so a single 896-element row isn't DRAM-latency-bound (the 32-thread
    /// kernel measured ~7x the per-dispatch cost of 256-thread elementwise ops).
    /// Requires a threadgroup buffer of n_simdgroups floats (8 for 256 threads).
    pub fn rms_norm_256(
        &self,
        x: &MetalBuffer,
        w: Option<&MetalBuffer>,
        w_off: u64,
        y: &MetalBuffer,
        d: usize,
        n: usize,
        eps: f32,
        off: u64,
        y_off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc
            .setComputePipelineState(&*self.state.pl_rms_norm_256);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (off) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(w.unwrap_or(y))),
                (w_off) as usize,
                (1) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (2) as usize)
        };
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((32 * 4) as usize, (0) as usize)
        };
        // 256 threads = 8 simdgroups; one threadgroup per row.
        self.dispatch_2d(n as u64, 1, 32, 8);
    }

    /// Element-wise add: z = x + y
    pub fn add_f32(&self, x: &MetalBuffer, y: &MetalBuffer, z: &MetalBuffer, n: usize) {
        self.add_f32_off(x, y, z, n, 0, 0, 0);
    }

    /// Element-wise add with per-buffer byte offsets (last-layer output-rows
    /// reduction: x/z read/write the tail n_out rows of `hidden`, y starts at 0).
    pub fn add_f32_off(
        &self,
        x: &MetalBuffer,
        y: &MetalBuffer,
        z: &MetalBuffer,
        n: usize,
        x_off: u64,
        y_off: u64,
        z_off: u64,
    ) {
        self.trace_op("add");
        self.enc.setComputePipelineState(&*self.state.pl_add);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(z)), (z_off) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        // float4 kernel: 4 elements/thread (ceil for the scalar tail)
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// Add 1-D bias to rows: y[t][i] += b[i]. `off` = element offset into `y`
    /// (used by the fused QKV path to bias the q/k/v sections of one buffer).
    pub fn add_bias_f32(
        &self,
        y: &MetalBuffer,
        b: &MetalBuffer,
        b_off: u64,
        d: usize,
        n: usize,
        off: usize,
    ) {
        self.trace_op("bias");
        self.enc.setComputePipelineState(&*self.state.pl_add_bias);
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(y)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(b)), (b_off) as usize, (1) as usize)
        };
        self.set_params(2, &(d as i32));
        // float4 kernel: 4 dims/thread
        self.dispatch_2d(n as u64, ((d as u64) + 3) / 4, 1, 64);
    }

    /// Element-wise multiply: z = x * y
    pub fn mul_f32(&self, x: &MetalBuffer, y: &MetalBuffer, z: &MetalBuffer, n: usize) {
        self.enc.setComputePipelineState(&*self.state.pl_mul);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(z)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SiLU in-place: y = y / (1 + exp(-y))
    pub fn silu_f32(&self, y: &MetalBuffer, n: usize) {
        self.enc.setComputePipelineState(&*self.state.pl_silu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (0) as usize, (0) as usize)
        };
        self.set_params(1, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SwiGLU fused: dst = silu(gate) * up  (dst may alias gate)
    pub fn swiglu_f32(&self, gate: &MetalBuffer, up: &MetalBuffer, dst: &MetalBuffer, n: usize) {
        self.trace_op("swiglu");
        self.enc.setComputePipelineState(&*self.state.pl_swiglu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(gate)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(up)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// SwiGLU over a fused gate+up buffer: gate at offset 0, up at `up_off`
    /// elements (fused FFN gate+up path). Writes silu(gate)*up back to gate.
    pub fn swiglu_f32_off(
        &self,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        dst: &MetalBuffer,
        n: usize,
        up_off: usize,
    ) {
        self.trace_op("swiglu");
        self.enc.setComputePipelineState(&*self.state.pl_swiglu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(gate)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(up)),
                ((up_off * 4) as u64) as usize,
                (1) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// RoPE (in-place): x layout [nt][n_head][n_dims]. `off` = element offset
    /// into `x` (fused QKV: K section lives mid-buffer).
    /// rope_style: 0 = non-interleaved (Qwen2), 1 = interleaved (LLaMA).
    pub fn rope_f32(
        &self,
        x: &MetalBuffer,
        n_head: usize,
        n_dims: usize,
        nt: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: &MetalBuffer,
        rope_style: i32,
        off: usize,
    ) {
        self.trace_op("rope");
        self.enc.setComputePipelineState(&*self.state.pl_rope);
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(x)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        self.set_params(1, &(n_head as i32));
        self.set_params(2, &(n_dims as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(freq_base.to_bits() as i32));
        self.set_params(5, &(freq_scale.to_bits() as i32));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (6) as usize)
        };
        self.set_params(7, &rope_style);
        // P7: one thread per (dim, head, token) instead of one per (token, head)
        self.dispatch_3d((n_dims / 2) as u64, n_head as u64, nt as u64, 1, 1, 1);
    }

    /// Flash Attention: one threadgroup per (token, KV_head), tiled K/V
    /// with online softmax. Each simdgroup processes one query head.
    /// K/V tiles loaded into threadgroup-shared memory, reused by all
    /// query heads in the GQA group.
    pub fn gqa_attn_f32(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        f16: bool,
    ) {
        self.gqa_attn_f32_off(q, 0, k, 0, v, 0, o, positions, nh, nk, hd, scale, nt, f16);
    }

    /// Offset variant of `gqa_attn_f32` — K/V may live at byte offsets inside a
    /// shared buffer (the graph backend's `[K | V]` contiguous KV region).
    pub fn gqa_attn_f32_off(
        &self,
        q: &MetalBuffer,
        q_off: u64,
        k: &MetalBuffer,
        k_off: u64,
        v: &MetalBuffer,
        v_off: u64,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        f16: bool,
    ) {
        self.trace_op("gqa_attn");
        let gqa = nh / nk;
        self.enc.setComputePipelineState(
            &**(if f16 {
                &self.state.pl_gqa_attn_f16
            } else {
                &self.state.pl_gqa_attn
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (q_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (k_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (v_off) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_2d(nt as u64, nk as u64, 32, gqa as u64);
    }

    /// E1 `attn_span` windowed attention (issue #44, G5a): the read side of the
    /// explicit window. Same grid and threadgroup shape as [`Self::gqa_attn_f32`]
    /// — one threadgroup per `(query, KV head)`, `gqa` simdgroups — but `window`
    /// (buffer 4) is the `attn_span` input: `lo` at `window[t]`, `hi` at
    /// `window[nt + t]`, and K/V are read at the run's cells `[lo, hi)` rather
    /// than `[0, positions[t] + 1)`. f32/f16 selected from the engine's KV format.
    ///
    /// `dispatch_2d(nt, nk, 32, gqa)` keeps every simdgroup inside the
    /// threadgroup on a barrier-bearing path even when `nh % nk != 0`, exactly as
    /// the causal kernel does; the kernel handles the window length internally,
    /// so `nt == 1` (a decode) and `nt > 1` (a batch prefill) use one launch.
    pub fn gqa_attn_window(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        window: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        f16: bool,
    ) {
        self.trace_op("gqa_attn_window");
        let gqa = nh / nk;
        self.enc.setComputePipelineState(
            &**(if f16 {
                &self.state.pl_gqa_attn_window_f16
            } else {
                &self.state.pl_gqa_attn_window
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(window)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_2d(nt as u64, nk as u64, 32, gqa as u64);
    }

    /// KV-parallel split attention for nt==1 decode (the classic kernel's grid
    /// is only (1, nk) threadgroups that loop the KV sequentially — the measured
    /// #1 decode bottleneck). Two passes: partial per KV chunk (grid (nt,nk,P)),
    /// then combine (grid (nt,nh)). Requires the partials buffer (`buf_attn_partial`)
    /// sized for nt*nh*P*(2+hd) floats, grown on demand here.
    pub fn gqa_attn_split_f32(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        n_chunks: usize,
        f16: bool,
    ) {
        self.trace_op("gqa_attn_split");
        let gqa = nh / nk;
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: partials per (token, KV_head, chunk) — f16 cache picks the
        // f16 partial kernel (K/V read as half, staged to f32 float4 tiles).
        self.enc.setComputePipelineState(
            &**(if f16 {
                &self.state.pl_gqa_attn_partial_f16
            } else {
                &self.state.pl_gqa_attn_partial
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_3d(nt as u64, nk as u64, n_chunks as u64, 32, gqa as u64, 1);

        // pass 2: combine
        self.enc
            .setComputePipelineState(&*self.state.pl_gqa_attn_combine);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nh as i32));
        self.set_params(3, &(hd as i32));
        self.set_params(4, &(nt as i32));
        self.set_params(5, &(n_chunks as i32));
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Flash-attention port (llama kernel_flash_attn_ext_vec, NSG=1 fixed
    /// DK=DV=64/NE=2/C=32 shape) for nt==1 decode. Replaces the split pair with
    /// a single-simdgroup-per-(t,h,iwg) kernel whose Q*K^T reduce is
    /// shuffle-based (simd_shuffle_down 8,4,2,1 + broadcast) instead of
    /// threadgroup barriers — llama's structural advantage over the split
    /// attention (~7-10x isolated at nkv=430). Output partials are {M,S,O[hd]}
    /// in the SAME layout as kernel_gqa_attn_partial_f32, so the shared combine
    /// kernel merges them unchanged. Grid (nt, nh, n_chunks), 32 threads.
    /// Host guard: layer_gpu only dispatches this when hd==64 (fixed DK/DV);
    /// otherwise the split path is used.
    pub fn gqa_attn_flash(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        n_chunks: usize,
        f16: bool,
    ) {
        self.trace_op("gqa_attn_flash");
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: flash partials — f16 cache reads the half K/V directly.
        self.enc.setComputePipelineState(
            &**(match (f16, hd) {
                (false, 128) => &self.state.pl_flash_attn_hd128,
                (true, 128) => &self.state.pl_flash_attn_hd128_f16,
                (false, _) => &self.state.pl_flash_attn,
                (true, _) => &self.state.pl_flash_attn_f16,
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        // shmem (hd=64): sq4 (16 float4 = 256 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 896 → 1024
        // shmem (hd=128): sq4 (32 float4 = 512 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 1152
        let shmem = if hd == 128 { 1152 } else { 1024 };
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_3d(nt as u64, nh as u64, n_chunks as u64, 32, 1, 1);

        // pass 2: combine (shared with the split path)
        self.enc
            .setComputePipelineState(&*self.state.pl_gqa_attn_combine);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nh as i32));
        self.set_params(3, &(hd as i32));
        self.set_params(4, &(nt as i32));
        self.set_params(5, &(n_chunks as i32));
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Scatter nt rows of src[nt][nkt] into dst[positions[t]][nkt].
    /// Writes f32 (default) or f16 (MINFER_CACHE_TYPE=f16) into the KV cache.
    pub fn store_kv(
        &self,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        nkt: usize,
        nt: usize,
        positions: &MetalBuffer,
        off: usize,
        f16: bool,
    ) {
        self.trace_op("store_kv");
        self.enc.setComputePipelineState(
            &**(if f16 {
                &self.state.pl_store_kv_f16
            } else {
                &self.state.pl_store_kv
            }),
        );
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(src)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nkt as i32));
        self.set_params(3, &(nt as i32));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.dispatch_2d(nt as u64, nkt as u64, 1, 1);
    }

    /// Prefill parallel attention (P1 2026-08-11): replaces the classic
    /// latency-bound attention kernel for nt>1 (grid (nt,nk), sequential KV loop
    /// with ~24K barriers at nt=430 → ~100ms, 48% of prefill, ~25x llama's).
    /// This 3-pass replacement is fully parallel (no threadgroup barriers):
    ///   1. scores[t][h][kv] = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale
    ///   2. masked softmax over kv per (t,h) row
    ///   3. out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..]
    /// q: [nt][nqt], kv_k/kv_v: [nkv][nkt], out: [nt][nqt]. nkv = real KV length
    /// (max_pos+1); the scores buffer is [nt][nh][nkv] (no padding needed — all
    /// three kernels handle arbitrary nkv).
    pub fn attn_parallel_prefill(
        &self,
        q: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        out: &MetalBuffer,
        positions: &MetalBuffer,
        nkv: usize,
        nkt: usize,
        _nqt: usize,
        nt: usize,
        nh: usize,
        hd: usize,
        gqa: usize,
        scale: f32,
    ) {
        self.trace_op("attn_parallel");
        let dev = &self.state.device;
        let scores =
            MpsState::get_or_grow(&self.state.buf_attn_scores, (nt * nh * nkv * 4) as u64, dev);

        // pass 1: scores [nt*nh][nkv] — one 256-thread TG per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_scores);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nh as i32));
        self.set_params(4, &(hd as i32));
        self.set_params(5, &(nkv as i32));
        self.set_params(6, &(nt as i32));
        self.set_params(7, &(gqa as i32));
        self.set_params(8, &(nkt as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.dispatch_2d((nt * nh) as u64, 1, 256, 1);

        // pass 2: masked softmax over kv per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_softmax_attn);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nkv as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(nh as i32));
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((32 * 4) as usize, (0) as usize)
        };
        self.dispatch_2d((nt * nh) as u64, 1, 32, 8);

        // pass 3: out = softmax · V — one 256-thread TG per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_output);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nh as i32));
        self.set_params(4, &(hd as i32));
        self.set_params(5, &(nkv as i32));
        self.set_params(6, &(nt as i32));
        self.set_params(7, &(gqa as i32));
        self.set_params(8, &(nkt as i32));
        self.dispatch_2d((nt * nh) as u64, 1, 256, 1);
    }

    /// Prefill flash attention (2026-08-14, llama kernel_flash_attn_ext_blk port):
    /// ONE kernel replaces the 3-pass parallel attention for nt>1 (measured 46 ms
    /// of 135 ms prefill GPU vs llama's ~3 ms). Fixed-shape NSG=4/Q=8/C=64/
    /// DK=DV=64: grid (ceil(nt/8), nh) of 128-thread threadgroups (32 lanes × 4
    /// simdgroups), each computing Q=8 query tokens × ALL KV for head h via
    /// simdgroup_matrix QK^T + online softmax + PV with an inline causal mask.
    /// GQA head hk = h/gqa is baked into the K/V base inside the kernel.
    /// The host copies the last partial KV block (nkv % 64 != 0) into a
    /// [2][64][nkt] tail-pad buffer first (kernel_kv_tail_pad); padded rows are
    /// zero + masked, so a pad buffer is always bound but only populated then.
    pub fn attn_flash_prefill(
        &self,
        q: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        out: &MetalBuffer,
        positions: &MetalBuffer,
        nkv: usize,
        nkt: usize,
        nt: usize,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        f16: bool,
    ) {
        self.trace_op("attn_flash_blk");
        let dev = &self.state.device;
        let elem = if f16 { 2u64 } else { 4u64 };
        let pad =
            MpsState::get_or_grow(&self.state.buf_attn_pad, (2 * 64 * nkt as u64) * elem, dev);

        if nkv % 64 != 0 {
            self.enc
                .setComputePipelineState(&*self.state.pl_kv_tail_pad);
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (0) as usize)
            };
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (1) as usize)
            };
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&*pad), (0) as usize, (2) as usize)
            };
            self.set_params(3, &(nkv as i32));
            self.set_params(4, &(nkt as i32));
            self.set_params(5, &(if f16 { 1 } else { 0 }));
            self.dispatch_2d(nkt as u64, 64, 1, 1);
        }

        self.enc.setComputePipelineState(
            &**(if f16 {
                if hd == 128 {
                    &self.state.pl_flash_attn_blk_hd128_f16
                } else {
                    &self.state.pl_flash_attn_blk_f16
                }
            } else {
                if hd == 128 {
                    &self.state.pl_flash_attn_blk_hd128
                } else {
                    &self.state.pl_flash_attn_blk
                }
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*pad), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (4) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (5) as usize)
        };
        self.set_params(6, &(nh as i32));
        self.set_params(7, &(nk as i32));
        self.set_params(8, &(hd as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.set_params(10, &(nt as i32));
        self.set_params(11, &(nkv as i32));
        // shmem: hd=64: sq (512 half = 1024 B) | so (512 f32 = 2048 B) | ss (1024 f32 = 4096 B);
        //        hd=128: sq (1024 half = 2048 B) | so (1024 f32 = 4096 B) | ss (1024 f32 = 4096 B)
        let shmem = if hd == 128 { 10240u64 } else { 7168u64 };
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_2d(((nt + 7) / 8) as u64, nh as u64, 32, 4);
    }

    /// Fused bias-add + RoPE + KV-store for nt==1 decode: ONE kernel replaces
    /// add_bias×3 + rope×2 + store_kv×2 (7 dispatches). `bqkv` layout is
    /// [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt]; biases are the raw
    /// per-section buffers. `pos` = the single token position. The KV store
    /// writes f32 or f16 (per the engine's `kv_format`) into kv_k/kv_v.
    pub fn attn_bias_rope_store(
        &self,
        bqkv: &MetalBuffer,
        bias_q: &MetalBuffer,
        bq_off: u64,
        bias_k: &MetalBuffer,
        bk_off: u64,
        bias_v: &MetalBuffer,
        bv_off: u64,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        pos: i32,
        rope_style: i32,
        f16: bool,
    ) {
        self.trace_op("attn_bias_rope_store");
        self.enc.setComputePipelineState(&*self.state.pl_attn_bsr);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bqkv)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_q)), (bq_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_k)), (bk_off) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_v)), (bv_off) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (4) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (5) as usize)
        };
        self.set_params(6, &(nqt as i32));
        self.set_params(7, &(nkt as i32));
        self.set_params(8, &(hd as i32));
        self.set_params(9, &(freq_base.to_bits() as i32));
        self.set_params(10, &(freq_scale.to_bits() as i32));
        self.set_params(11, &pos);
        self.set_params(12, &rope_style);
        self.set_params(13, &(if f16 { 1 } else { 0 }));
        let grid = nqt / 2 + nkt / 2 + nkt;
        self.dispatch_1d(grid as u64, 256);
    }

    /// Fused decode QKV rope+store WITHOUT attention biases (Qwen3): the concat
    /// buffer `bqkv` holds q|k|v (q = rows 0..nqt, k = nqt..nqt+nkt,
    /// v = nqt+nkt..), and the per-head Q/K RMSNorm was already applied in place
    /// by the preceding rms_norm_256 dispatches. This kernel only RoPEs q in
    /// place, RoPEs + stores K, and stores V into the persistent KV regions.
    /// Grid: nqt/2 + nkt/2 + nkt, 256 threads. Buffer indices: 0=bqkv, 1=kv_k,
    /// 2=kv_v; params 3..=10.
    pub fn attn_rope_store(
        &self,
        bqkv: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        pos: i32,
        rope_style: i32,
        f16: bool,
    ) {
        self.trace_op("attn_rope_store");
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_rope_store);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bqkv)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nqt as i32));
        self.set_params(4, &(nkt as i32));
        self.set_params(5, &(hd as i32));
        self.set_params(6, &(freq_base.to_bits() as i32));
        self.set_params(7, &(freq_scale.to_bits() as i32));
        self.set_params(8, &pos);
        self.set_params(9, &rope_style);
        self.set_params(10, &(if f16 { 1 } else { 0 }));
        let grid = nqt / 2 + nkt / 2 + nkt;
        self.dispatch_1d(grid as u64, 256);
    }
}
