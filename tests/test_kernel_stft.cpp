// Validates cactus_stft / cactus_istft via an ISTFT round-trip on a
// 24 kHz chirp. The forward STFT applies a Hann window per frame and
// runs cactus_fft_r2c; the inverse runs cactus_fft_c2r per frame,
// re-windows, overlap-adds, and normalizes by the per-sample
// window-sum. Compared against scipy's istft reference output;
// edge samples (first/last n_fft) leak from windowing as expected.

#include "test_utils.h"
#include "../cactus/kernel/kernel_stft.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string fixture_path(const char* name) {
    const char* env = std::getenv("CACTUS_TEST_FFT_FIXTURES");
    std::string base = env ? env : "../fixtures/data/";
    return base + name;
}

static std::vector<float> load_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<float> out(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
    return out;
}

static bool test_istft_roundtrip_chirp() {
    auto chirp    = load_f32(fixture_path("stft_input_chirp.bin"));
    auto expected = load_f32(fixture_path("istft_output_chirp.bin"));

    constexpr int N_FFT = 512;
    constexpr int HOP   = 128;
    const int n_freq    = N_FFT / 2 + 1;
    const int n_frames  = (static_cast<int>(chirp.size()) - N_FFT) / HOP + 1;

    std::vector<float> spec(static_cast<size_t>(n_frames) * n_freq * 2);
    cactus_stft(chirp.data(), static_cast<int>(chirp.size()),
                spec.data(), N_FFT, HOP, n_frames);

    std::vector<float> recovered(chirp.size(), 0.0f);
    cactus_istft(spec.data(), n_frames,
                 recovered.data(), static_cast<int>(chirp.size()),
                 N_FFT, HOP);

    // Compare INTERIOR samples — first/last N_FFT samples leak from windowing edge effects.
    int errors = 0;
    for (size_t i = N_FFT; i + N_FFT < expected.size(); ++i) {
        const float diff = std::fabs(recovered[i] - expected[i]);
        if (diff > 1e-3f) {
            if (errors < 5) std::fprintf(stderr, "round-trip mismatch i=%zu rec=%g exp=%g diff=%g\n",
                                          i, recovered[i], expected[i], diff);
            ++errors;
        }
    }
    return errors == 0;
}

int main() {
    TestUtils::TestRunner runner("cactus_stft / cactus_istft");
    runner.run_test("ISTFT round-trip on chirp matches scipy", test_istft_roundtrip_chirp());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
