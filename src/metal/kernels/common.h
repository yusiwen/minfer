// Metal shaders for minfer — Q4_0 matmul + element-wise ops.

#include <metal_stdlib>
using namespace metal;

constant int Q4B = 18;
constant int Q41B = 20;
constant int Q5B = 22;

// Shared kernel launch parameters
constant short NW_Q = 32;
constant short NQ_Q = 16;
constant short QK   = 32;
