#include "kernel_fft.h"
#include "pffft.h"

#include <stdexcept>
#include <vector>

// PFFFT_REAL ordered forward output for an N-point input is packed as:
//   tmp[0] = re_0
//   tmp[1] = re_{N/2}                      (Nyquist real, packed into slot 1)
//   tmp[2] = re_1,  tmp[3] = im_1
//   tmp[4] = re_2,  tmp[5] = im_2
//   ...
//   tmp[N-2] = re_{N/2-1}, tmp[N-1] = im_{N/2-1}
//
// We unpack to scipy's interleaved layout (length 2*(N/2+1) = N+2 floats):
//   output[0]    = re_0,         output[1]    = 0          (im_0 ≡ 0)
//   output[2k]   = re_k,         output[2k+1] = im_k       for k = 1..N/2-1
//   output[N]    = re_{N/2},     output[N+1]  = 0          (im_{N/2} ≡ 0)

void cactus_fft_r2c(const float* input, float* output, size_t n) {
    if (n < 32 || (n % 32) != 0) {
        throw std::runtime_error(
            "cactus_fft_r2c: n must be >= 32 and a multiple of 32 "
            "(PFFFT SIMD constraint)");
    }

    PFFFT_Setup* setup = pffft_new_setup(static_cast<int>(n), PFFFT_REAL);
    if (!setup) throw std::runtime_error("pffft_new_setup failed");

    std::vector<float> tmp(n);
    pffft_transform_ordered(setup, input, tmp.data(), nullptr, PFFFT_FORWARD);

    output[0] = tmp[0];           // re_0
    output[1] = 0.0f;             // im_0
    for (size_t k = 1; k < n / 2; ++k) {
        output[2 * k]     = tmp[2 * k];      // re_k
        output[2 * k + 1] = tmp[2 * k + 1];  // im_k
    }
    output[n]     = tmp[1];       // re_{N/2}
    output[n + 1] = 0.0f;         // im_{N/2}

    pffft_destroy_setup(setup);
}

// Inverse of cactus_fft_r2c: re-pack scipy interleaved layout into PFFFT's
// packed layout, run PFFFT_BACKWARD, then normalize by 1/N (PFFFT inverse
// is unnormalized).
//
// scipy interleaved (length N+2) -> PFFFT packed (length N):
//   tmp[0] = input[0]                        (re_0)
//   tmp[1] = input[N]                        (re_{N/2}, packed into PFFFT slot 1)
//   tmp[2k]   = input[2k]                    (re_k)
//   tmp[2k+1] = input[2k+1]                  (im_k)   for k = 1..N/2-1

void cactus_fft_c2r(const float* input, float* output, size_t n) {
    if (n < 32 || (n % 32) != 0) {
        throw std::runtime_error(
            "cactus_fft_c2r: n must be >= 32 and a multiple of 32 "
            "(PFFFT SIMD constraint)");
    }

    PFFFT_Setup* setup = pffft_new_setup(static_cast<int>(n), PFFFT_REAL);
    if (!setup) throw std::runtime_error("pffft_new_setup failed");

    std::vector<float> tmp(n);
    tmp[0] = input[0];            // re_0
    tmp[1] = input[n];            // re_{N/2} packed into slot 1
    for (size_t k = 1; k < n / 2; ++k) {
        tmp[2 * k]     = input[2 * k];      // re_k
        tmp[2 * k + 1] = input[2 * k + 1];  // im_k
    }

    pffft_transform_ordered(setup, tmp.data(), output, nullptr, PFFFT_BACKWARD);

    // PFFFT inverse is unnormalized — divide by N to recover the original signal.
    const float scale = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) output[i] *= scale;

    pffft_destroy_setup(setup);
}
