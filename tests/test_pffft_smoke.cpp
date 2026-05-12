// Smoke test for vendored PFFFT (libs/pffft/).
// Validates a 256-point real-FFT forward/inverse round-trip against a known
// sine wave. PFFFT does not normalize the inverse transform; we divide by N
// to recover the original signal.

#include "test_utils.h"
#include "../libs/pffft/pffft.h"

#include <cmath>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static bool test_pffft_real_roundtrip_256() {
    constexpr int N = 256;

    PFFFT_Setup* setup = pffft_new_setup(N, PFFFT_REAL);
    if (!setup) {
        std::fprintf(stderr, "pffft_new_setup failed for N=%d\n", N);
        return false;
    }

    std::vector<float> input(N);
    std::vector<float> spectrum(N);
    std::vector<float> output(N);

    // 5-cycle sine wave across N samples.
    for (int i = 0; i < N; ++i) {
        input[i] = std::sin(2.0f * static_cast<float>(M_PI) * 5.0f *
                            static_cast<float>(i) / static_cast<float>(N));
    }

    pffft_transform_ordered(setup, input.data(), spectrum.data(), nullptr, PFFFT_FORWARD);
    pffft_transform_ordered(setup, spectrum.data(), output.data(), nullptr, PFFFT_BACKWARD);

    // PFFFT inverse is unnormalized; rescale by 1/N.
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i) output[i] *= inv_n;

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        const float diff = std::fabs(input[i] - output[i]);
        if (diff > 1e-5f) {
            if (errors < 5) {
                std::fprintf(stderr,
                             "mismatch at i=%d: input=%g output=%g diff=%g\n",
                             i, input[i], output[i], diff);
            }
            ++errors;
        }
    }

    pffft_destroy_setup(setup);

    if (errors > 0) {
        std::fprintf(stderr,
                     "PFFFT round-trip failed (%d/%d samples diverged)\n",
                     errors, N);
        return false;
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("PFFFT Smoke Tests");
    runner.run_test("PFFFT 256-point Real FFT Round-trip", test_pffft_real_roundtrip_256());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
