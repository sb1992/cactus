// Validates cactus_fft_r2c (forward real-to-complex FFT) against scipy
// reference outputs in tests/fixtures/data/. The wrapper unpacks PFFFT's
// packed real-FFT layout into scipy.fft.rfft's interleaved [re, im] layout
// of length 2*(N/2+1) with explicit zeros for im_0 and im_{N/2}.
//
// Fixtures are produced by tests/fixtures/fft_reference.py and are stored
// as little-endian float32 arrays prefixed with a uint32 element count.

#include "test_utils.h"
#include "../cactus/kernel/kernel_fft.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> load_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<float> out(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
    return out;
}

static std::string fixture_path(const std::string& name) {
    // Tests run from tests/build/, so fixtures live at ../fixtures/data/.
    if (const char* env = std::getenv("CACTUS_TEST_FFT_FIXTURES")) {
        return std::string(env) + "/" + name;
    }
    return std::string("../fixtures/data/") + name;
}

static bool test_r2c_256_sine_matches_scipy() {
    auto input    = load_f32(fixture_path("fft_input_256_sine.bin"));
    auto expected = load_f32(fixture_path("fft_output_256_sine.bin"));
    if (input.size() != 256 || expected.size() != (256 / 2 + 1) * 2) {
        std::fprintf(stderr, "fixture sizes wrong: input=%zu expected=%zu\n",
                     input.size(), expected.size());
        return false;
    }
    std::vector<float> output((256 / 2 + 1) * 2);
    cactus_fft_r2c(input.data(), output.data(), 256);

    int errors = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::fabs(output[i] - expected[i]);
        if (diff > 1e-3f) {
            if (errors < 5) {
                std::fprintf(stderr,
                             "256-sine mismatch i=%zu out=%g exp=%g diff=%g\n",
                             i, output[i], expected[i], diff);
            }
            ++errors;
        }
    }
    return errors == 0;
}

static bool test_r2c_512_random_matches_scipy() {
    auto input    = load_f32(fixture_path("fft_input_512_random.bin"));
    auto expected = load_f32(fixture_path("fft_output_512_random.bin"));
    if (input.size() != 512 || expected.size() != (512 / 2 + 1) * 2) {
        std::fprintf(stderr, "fixture sizes wrong: input=%zu expected=%zu\n",
                     input.size(), expected.size());
        return false;
    }
    std::vector<float> output((512 / 2 + 1) * 2);
    cactus_fft_r2c(input.data(), output.data(), 512);

    int errors = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::fabs(output[i] - expected[i]);
        if (diff > 1e-3f) {
            if (errors < 5) {
                std::fprintf(stderr,
                             "512-random mismatch i=%zu out=%g exp=%g diff=%g\n",
                             i, output[i], expected[i], diff);
            }
            ++errors;
        }
    }
    return errors == 0;
}

int main() {
    TestUtils::TestRunner runner("cactus_fft_r2c");
    runner.run_test("R2C 256 sine matches scipy",   test_r2c_256_sine_matches_scipy());
    runner.run_test("R2C 512 random matches scipy", test_r2c_512_random_matches_scipy());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
