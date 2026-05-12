#include "test_utils.h"
#include "../cactus/kernel/kernel_conv1d_transpose.h"

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

static bool test_conv1d_transpose_matches_torch_C4_K4_S2_P1() {
    auto W = load_f32(fixture_path("convT_weight.bin"));
    auto b = load_f32(fixture_path("convT_bias.bin"));
    auto x = load_f32(fixture_path("convT_x.bin"));
    auto y_expected = load_f32(fixture_path("convT_y.bin"));

    constexpr int C_IN = 4, C_OUT = 4, K = 4, STRIDE = 2, PAD = 1;
    constexpr int T_IN = 16;
    constexpr int T_OUT = (T_IN - 1) * STRIDE - 2 * PAD + K;  // 32

    if (W.size() != static_cast<size_t>(C_IN * C_OUT * K) ||
        b.size() != static_cast<size_t>(C_OUT) ||
        x.size() != static_cast<size_t>(C_IN * T_IN) ||
        y_expected.size() != static_cast<size_t>(C_OUT * T_OUT)) {
        std::fprintf(stderr, "convT fixture sizes wrong\n");
        return false;
    }

    std::vector<float> y(static_cast<size_t>(C_OUT * T_OUT));
    cactus_conv1d_transpose(x.data(), W.data(), b.data(), y.data(),
                            C_IN, C_OUT, T_IN, K, STRIDE, PAD);

    int errors = 0;
    for (size_t i = 0; i < y_expected.size(); ++i) {
        const float diff = std::fabs(y[i] - y_expected[i]);
        if (diff > 1e-4f) {
            if (errors < 5) std::fprintf(stderr, "convT mismatch i=%zu got=%g exp=%g\n",
                                          i, y[i], y_expected[i]);
            ++errors;
        }
    }
    return errors == 0;
}

int main() {
    TestUtils::TestRunner runner("cactus_conv1d_transpose");
    runner.run_test("ConvT1d matches torch (C=4 K=4 S=2 P=1)",
                    test_conv1d_transpose_matches_torch_C4_K4_S2_P1());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
