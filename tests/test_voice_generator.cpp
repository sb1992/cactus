#include "test_utils.h"
#include "../cactus/models/kokoro/generator.h"

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
// Fixture loaders (mirror test_voice_decoder.cpp)
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

static cactus::kokoro::Generator load_generator() {
    cactus::kokoro::Generator g;
    const auto dir = find_weights_dir();
    if (!g.load_weights(dir)) {
        throw std::runtime_error("Generator load_weights failed (dir=" + dir + ")");
    }
    return g;
}

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
    std::vector<float> x_decode3;   // (2*T_mel, 512) row-major
    std::vector<float> har_source;  // (600*T_mel,)
    std::vector<float> F0;          // (2*T_mel,)
    std::vector<float> style128;    // (128,)
    int T_mel = 0;
};

PhraseInputs load_inputs(const std::string& tag) {
    PhraseInputs in;
    in.x_decode3  = load_typed<float>(fixture_path("decoder_" + tag + "_after_decode_block_3.bin"));
    in.har_source = load_typed<float>(fixture_path("decoder_" + tag + "_har_source.bin"));
    in.F0         = load_typed<float>(fixture_path("decoder_" + tag + "_input_F0.bin"));
    in.style128   = load_typed<float>(fixture_path("decoder_" + tag + "_input_style128.bin"));
    if (in.style128.size() != 128) {
        throw std::runtime_error("style128 size != 128 for " + tag);
    }
    if (in.F0.size() % 2 != 0) {
        throw std::runtime_error("F0 size mismatch for " + tag);
    }
    in.T_mel = static_cast<int>(in.F0.size() / 2);
    if (in.x_decode3.size() != static_cast<size_t>(2 * in.T_mel) * 512) {
        throw std::runtime_error("x_decode3 size mismatch for " + tag);
    }
    if (in.har_source.size() != static_cast<size_t>(600 * in.T_mel)) {
        throw std::runtime_error("har_source size mismatch for " + tag);
    }
    return in;
}

std::vector<float> load_expected(const std::string& tag, const std::string& name) {
    return load_typed<float>(fixture_path("decoder_" + tag + "_" + name + ".bin"));
}

}  // namespace

using Stage = cactus::kokoro::Generator::Stage;

static std::vector<float> run_stage(const cactus::kokoro::Generator& g,
                                    const PhraseInputs& in,
                                    Stage s) {
    std::vector<float> buf;
    g.run_until_stage_with_har_source(
        in.x_decode3.data(), in.T_mel,
        in.har_source.data(), static_cast<int>(in.har_source.size()),
        in.style128.data(),
        s, buf);
    return buf;
}

// --- Per-stage tests ---
static bool test_har_spec_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    std::vector<float> mag, phase;
    g.compute_har_stft(in.har_source.data(),
                       static_cast<int>(in.har_source.size()),
                       mag, phase);
    auto exp_mag   = load_expected("p01", "har_spec");
    auto exp_phase = load_expected("p01", "har_phase");
    bool ok = layer_close(mag,   exp_mag,   1e-3f, 1e-2f, "har_spec");
    ok = layer_close(phase, exp_phase, 1e-3f, 1e-2f, "har_phase") && ok;
    return ok;
}

static bool test_after_ups0_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    auto got = run_stage(g, in, Stage::AfterUps0);
    auto exp = load_expected("p01", "after_ups0");
    return layer_close(got, exp, 5e-2f, 2e-2f, "after_ups0");
}

static bool test_after_resblocks0_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    auto got = run_stage(g, in, Stage::AfterResblocks0);
    auto exp = load_expected("p01", "after_resblocks0");
    return layer_close(got, exp, 1e-1f, 5e-2f, "after_resblocks0");
}

static bool test_after_ups1_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    auto got = run_stage(g, in, Stage::AfterUps1);
    auto exp = load_expected("p01", "after_ups1");
    return layer_close(got, exp, 1e-1f, 5e-2f, "after_ups1");
}

static bool test_after_resblocks1_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    auto got = run_stage(g, in, Stage::AfterResblocks1);
    auto exp = load_expected("p01", "after_resblocks1");
    return layer_close(got, exp, 2e-1f, 5e-2f, "after_resblocks1");
}

static bool test_after_conv_post_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    auto got = run_stage(g, in, Stage::AfterConvPost);
    auto exp = load_expected("p01", "after_conv_post");
    return layer_close(got, exp, 2e-1f, 5e-2f, "after_conv_post");
}

static bool test_stft_components_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    std::vector<float> real_part, imag_part;
    g.compute_stft_components_with_har_source(
        in.x_decode3.data(), in.T_mel,
        in.har_source.data(), static_cast<int>(in.har_source.size()),
        in.style128.data(), real_part, imag_part);
    auto exp_real = load_expected("p01", "stft_real");
    auto exp_imag = load_expected("p01", "stft_imag");
    bool ok = layer_close(real_part, exp_real, 5e-1f, 5e-2f, "stft_real");
    ok = layer_close(imag_part, exp_imag, 5e-1f, 5e-2f, "stft_imag") && ok;
    return ok;
}

static bool test_audio_p01() {
    auto g = load_generator();
    auto in = load_inputs("p01");
    std::vector<float> got;
    g.forward_with_har_source(in.x_decode3.data(), in.T_mel,
                              in.har_source.data(),
                              static_cast<int>(in.har_source.size()),
                              in.style128.data(), got);
    auto exp = load_expected("p01", "audio");
    return layer_close(got, exp, 5e-2f, 1e-1f, "audio");
}

static bool test_e2e_audio_all_phrases() {
    auto g = load_generator();
    bool ok = true;
    for (const char* tag : {"p01", "p02", "p03", "p04", "p05"}) {
        auto in = load_inputs(tag);
        std::vector<float> got;
        g.forward_with_har_source(in.x_decode3.data(), in.T_mel,
                                  in.har_source.data(),
                                  static_cast<int>(in.har_source.size()),
                                  in.style128.data(), got);
        auto exp = load_expected(tag, "audio");
        char lab[64];
        std::snprintf(lab, sizeof(lab), "audio %s", tag);
        ok = layer_close(got, exp, 5e-2f, 1e-1f, lab) && ok;
    }
    return ok;
}

int main() {
    TestUtils::TestRunner runner("cactus::kokoro::Generator");
    try {
        runner.run_test("har_spec/phase (p01)",      test_har_spec_p01());
        runner.run_test("after_ups0 (p01)",          test_after_ups0_p01());
        runner.run_test("after_resblocks0 (p01)",    test_after_resblocks0_p01());
        runner.run_test("after_ups1 (p01)",          test_after_ups1_p01());
        runner.run_test("after_resblocks1 (p01)",    test_after_resblocks1_p01());
        runner.run_test("after_conv_post (p01)",     test_after_conv_post_p01());
        runner.run_test("stft components (p01)",     test_stft_components_p01());
        runner.run_test("audio (p01)",               test_audio_p01());
        runner.run_test("E2E audio all 5 phrases",   test_e2e_audio_all_phrases());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Test threw: %s\n", e.what());
        runner.run_test("Test threw exception", false);
    }
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
