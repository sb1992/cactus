#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 1D transposed convolution (PyTorch nn.ConvTranspose1d, single batch).
//
// Inputs (row-major fp32):
//   x:      [C_in, T_in]
//   W:      [C_in, C_out, K]    Note: PyTorch ConvTranspose1d weight order is (in, out, k)
//                                — DIFFERENT from regular Conv1d which is (out, in, k)
//   b:      [C_out]              bias (or NULL for no bias)
//
// Outputs:
//   y:      [C_out, T_out]       where T_out = (T_in - 1) * stride - 2*padding + K
//
// Math (scatter formulation):
//   y[oc, t_out] = sum over ic, kk where (t_in * stride + kk - padding) == t_out:
//                     x[ic, t_in] * W[ic, oc, kk]
//   then add b[oc].
//
// Scalar fp32 reference. NEON/SIMD optimization deferred.
void cactus_conv1d_transpose(
    const float* x, const float* W, const float* b, float* y,
    int c_in, int c_out, int t_in, int k, int stride, int padding);

#ifdef __cplusplus
}
#endif
