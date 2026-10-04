// CUDA kernels for minfer — the shrinking remainder of the single
// pre-#263 translation unit.
//
// Each stage of #263 moved one kernel family into src/cuda/kernels/;
// this file is deleted when it is empty (stage 6).
#include "cuda/kernels/common.cuh"

extern "C" {
} // extern "C" (C8b S4: fa_stage_kv_async became a template, which cannot have
  // C linkage — the launchers above keep theirs; the FA kernel and its launcher
  // below open a block of their own)
extern "C" {
} // extern "C"
extern "C" {
} // extern "C"
extern "C" {
} // extern "C"
