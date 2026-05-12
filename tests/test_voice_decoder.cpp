#include "test_utils.h"
#include "../cactus/models/kokoro/decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------------------------------------
// Fixture helpers (mirror test_voice_predictor.cpp)
// ----------------------------------------------------------------------
static std::string fixture_path(const std::string& name) {
    const char* env = std::getenv("CACTUS_TEST_FFT_FIXTURES");
    std::string base = env ? env : "../fixtures/data/";
    return base + name;
}

template <typename T>
static std::vector<T> load_typed(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<T> out(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(T));
    if (!f) throw std::runtime_error("short read for " + path);
    return out;
}

static std::string find_weights_dir() {
    const char* candidates[] = {
        "assets/kokoro/weights",
        "../assets/kokoro/weights",
        "../../assets/kokoro/weights",
    };
    for (const char* p : candidates) {
        std::ifstream f(std::string(p) + "/decoder_F0_conv_weight.weights");
        if (f) return p;
    }
    return "assets/kokoro/weights";
}

static cactus::kokoro::DecoderEncodeStage load_decoder() {
    cactus::kokoro::DecoderEncodeStage d;
    const auto dir = find_weights_dir();
    if (!d.load_weights(dir)) {
        throw std::runtime_error("decoder load_weights failed (dir=" + dir + ")");
    }
    return d;
}

// Dual abs/rel tolerance check (same shape as test_voice_predictor).
static bool layer_close(const std::vector<float>& got,
                        const std::vector<float>& expected,
                        float abs_tol, float rel_tol, const char* label) {
    if (got.size() != expected.size()) {
        std::fprintf(stderr, "%s: SIZE MISMATCH got=%zu exp=%zu\n",
                     label, got.size(), expected.size());
        return false;
    }
    float max_abs = 0.0f, max_rel = 0.0f;
    int errors = 0;
    int n_violations = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - expected[i]);
        if (d > max_abs) max_abs = d;
        const float r = d / std::max(std::fabs(expected[i]), 1e-9f);
        if (r > max_rel) max_rel = r;
        const bool ok = (d <= abs_tol) || (r <= rel_tol);
        if (!ok) {
            ++n_violations;
            if (errors < 3) {
                std::fprintf(stderr, "%s: i=%zu got=%g exp=%g abs=%g rel=%g\n",
                             label, i, got[i], expected[i], d, r);
                ++errors;
            }
        }
    }
    const bool pass = (n_violations == 0);
    std::fprintf(stderr,
        "%s: max_abs=%.6g max_rel=%.6g (abs_tol=%.1g rel_tol=%.1g) %s\n",
        label, max_abs, max_rel, abs_tol, rel_tol,
        pass ? "PASS" : "FAIL");
    return pass;
}

namespace {

struct PhraseInputs {
    std::vector<float> asr;        // (T_mel, 512)
    std::vector<float> F0;         // (2*T_mel,)
    std::vector<float> N;          // (2*T_mel,)
    std::vector<float> style128;   // (128,)
    int T_mel = 0;
};

PhraseInputs load_inputs(const std::string& tag) {
    PhraseInputs in;
    in.asr      = load_typed<float>(fixture_path("decoder_" + tag + "_input_asr.bin"));
    in.F0       = load_typed<float>(fixture_path("decoder_" + tag + "_input_F0.bin"));
    in.N        = load_typed<float>(fixture_path("decoder_" + tag + "_input_N.bin"));
    in.style128 = load_typed<float>(fixture_path("decoder_" + tag + "_input_style128.bin"));
    if (in.style128.size() != 128) {
        throw std::runtime_error("style128 size != 128 for " + tag);
    }
    if (in.F0.size() != in.N.size() || in.F0.size() % 2 != 0) {
        throw std::runtime_error("F0/N size mismatch for " + tag);
    }
    in.T_mel = static_cast<int>(in.F0.size() / 2);
    if (in.asr.size() != static_cast<size_t>(in.T_mel) * 512) {
        throw std::runtime_error("asr size mismatch for " + tag);
    }
    return in;
}

std::vector<float> load_expected(const std::string& tag, const std::string& name) {
    return load_typed<float>(fixture_path("decoder_" + tag + "_" + name + ".bin"));
}

std::vector<float> run_stage(const cactus::kokoro::DecoderEncodeStage& d,
                             const PhraseInputs& in,
                             cactus::kokoro::DecoderEncodeStage::Stage s) {
    std::vector<float> buf;
    d.run_until_stage(in.asr.data(), in.T_mel,
                      in.F0.data(), in.N.data(),
                      in.style128.data(), s, buf);
    return buf;
}

}  // namespace

using Stage = cactus::kokoro::DecoderEncodeStage::Stage;

#define DEFINE_STAGE_TEST(NAME, STAGE, FIX_NAME, ABS, REL)             \
    static bool test_##NAME() {                                        \
        auto d = load_decoder();                                       \
        auto in = load_inputs("p01");                                  \
        auto got = run_stage(d, in, STAGE);                            \
        auto exp = load_expected("p01", FIX_NAME);                     \
        return layer_close(got, exp, ABS, REL, #NAME);                 \
    }

// F0/N pre-conv outputs are scalar-per-frame; magnitudes ~ raw F0/N (up to
// hundreds for F0). The rel bound dominates.
DEFINE_STAGE_TEST(F0_after_conv_p01, Stage::F0AfterConv, "F0_after_conv", 1.0f,  2e-2f)
DEFINE_STAGE_TEST(N_after_conv_p01,  Stage::NAfterConv,  "N_after_conv",  1.0f,  2e-2f)
// Encode/decode block outputs are typical activation magnitudes (~few).
DEFINE_STAGE_TEST(after_encode_p01,        Stage::AfterEncode,  "after_encode",        1e-1f, 2e-2f)
DEFINE_STAGE_TEST(asr_res_p01,             Stage::AsrRes,       "asr_res",             1e-1f, 2e-2f)
DEFINE_STAGE_TEST(after_decode_block_0_p01,Stage::AfterDecode0, "after_decode_block_0",2e-1f, 2e-2f)
DEFINE_STAGE_TEST(after_decode_block_1_p01,Stage::AfterDecode1, "after_decode_block_1",2e-1f, 2e-2f)
DEFINE_STAGE_TEST(after_decode_block_2_p01,Stage::AfterDecode2, "after_decode_block_2",3e-1f, 2e-2f)
DEFINE_STAGE_TEST(after_decode_block_3_p01,Stage::AfterDecode3, "after_decode_block_3",3e-1f, 2e-2f)

// E2E across all 5 phrases: full forward up through decode.3.
static bool test_e2e_all_phrases() {
    auto d = load_decoder();
    bool ok = true;
    for (const char* tag : {"p01", "p02", "p03", "p04", "p05"}) {
        auto in = load_inputs(tag);
        std::vector<float> got;
        d.forward(in.asr.data(), in.T_mel, in.F0.data(), in.N.data(),
                  in.style128.data(), got);
        auto exp = load_expected(tag, "after_decode_block_3");
        char lab[64];
        std::snprintf(lab, sizeof(lab), "e2e %s decode.3", tag);
        ok = layer_close(got, exp, 5e-1f, 5e-2f, lab) && ok;
    }
    return ok;
}

int main() {
    TestUtils::TestRunner runner("cactus::kokoro::DecoderEncodeStage");
    try {
        runner.run_test("F0 after conv (p01)",   test_F0_after_conv_p01());
        runner.run_test("N after conv (p01)",    test_N_after_conv_p01());
        runner.run_test("After encode (p01)",    test_after_encode_p01());
        runner.run_test("asr_res (p01)",         test_asr_res_p01());
        runner.run_test("After decode 0 (p01)",  test_after_decode_block_0_p01());
        runner.run_test("After decode 1 (p01)",  test_after_decode_block_1_p01());
        runner.run_test("After decode 2 (p01)",  test_after_decode_block_2_p01());
        runner.run_test("After decode 3 (p01)",  test_after_decode_block_3_p01());
        runner.run_test("E2E up to decode.3 (5 phrases)", test_e2e_all_phrases());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Test threw: %s\n", e.what());
        runner.run_test("Test threw exception", false);
    }
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
