#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Real-to-complex forward FFT (scipy.fft.rfft compatible layout).
//
//   input:  pointer to N float samples (time domain)
//   output: pointer to (N/2 + 1) * 2 floats interleaved
//           [re_0, im_0, re_1, im_1, ..., re_{N/2}, im_{N/2}]
//           Note: im_0 and im_{N/2} are always 0 by construction of the real DFT.
//   n:      transform size; must be supported by PFFFT
//           (n >= 32 and n is a multiple of 32 for the SIMD path).
//
// Implementation wraps the vendored PFFFT (libs/pffft/) ordered transform and
// unpacks PFFFT's packed real-FFT layout into scipy's interleaved layout.
void cactus_fft_r2c(const float* input, float* output, size_t n);

#ifdef __cplusplus
}
#endif
