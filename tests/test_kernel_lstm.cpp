#include "test_utils.h"
#include "../cactus/kernel/kernel_lstm.h"

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

static bool test_lstm_cell_matches_torch_64x128() {
    auto W_ih = load_f32(fixture_path("lstm_W_ih.bin"));
    auto W_hh = load_f32(fixture_path("lstm_W_hh.bin"));
    auto b_ih = load_f32(fixture_path("lstm_b_ih.bin"));
    auto b_hh = load_f32(fixture_path("lstm_b_hh.bin"));
    auto x    = load_f32(fixture_path("lstm_x.bin"));
    auto h0   = load_f32(fixture_path("lstm_h0.bin"));
    auto c0   = load_f32(fixture_path("lstm_c0.bin"));
    auto h1_expected = load_f32(fixture_path("lstm_h1.bin"));
    auto c1_expected = load_f32(fixture_path("lstm_c1.bin"));

    constexpr int I = 64;
    constexpr int H = 128;
    if (W_ih.size() != static_cast<size_t>(4 * H * I) ||
        W_hh.size() != static_cast<size_t>(4 * H * H) ||
        b_ih.size() != static_cast<size_t>(4 * H) ||
        b_hh.size() != static_cast<size_t>(4 * H) ||
        x.size() != static_cast<size_t>(I) ||
        h0.size() != static_cast<size_t>(H) ||
        c0.size() != static_cast<size_t>(H)) {
        std::fprintf(stderr, "lstm fixture sizes wrong\n");
        return false;
    }

    std::vector<float> h1(H), c1(H);
    cactus_lstm_cell(x.data(), h0.data(), c0.data(),
                     W_ih.data(), W_hh.data(), b_ih.data(), b_hh.data(),
                     h1.data(), c1.data(), I, H);

    int errors = 0;
    for (int i = 0; i < H; ++i) {
        const float dh = std::fabs(h1[i] - h1_expected[i]);
        const float dc = std::fabs(c1[i] - c1_expected[i]);
        if (dh > 1e-4f) {
            if (errors < 5) std::fprintf(stderr, "h mismatch i=%d got=%g exp=%g\n", i, h1[i], h1_expected[i]);
            ++errors;
        }
        if (dc > 1e-4f) {
            if (errors < 5) std::fprintf(stderr, "c mismatch i=%d got=%g exp=%g\n", i, c1[i], c1_expected[i]);
            ++errors;
        }
    }
    return errors == 0;
}

int main() {
    TestUtils::TestRunner runner("cactus_lstm_cell");
    runner.run_test("LSTM cell matches torch (I=64, H=128)", test_lstm_cell_matches_torch_64x128());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
