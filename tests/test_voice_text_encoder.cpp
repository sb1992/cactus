#include "test_utils.h"
#include "../cactus/models/kokoro/text_encoder.h"

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
// Fixture helpers — match the convention used by test_kernel_lstm.cpp
// (uint32 element-count prefix, raw little-endian bytes, default dir is
// ../fixtures/data/ relative to the test build dir).
// ----------------------------------------------------------------------
static std::string fixture_path(const char* name) {
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
        std::ifstream f(std::string(p) + "/text_encoder_embedding.weights");
        if (f) return p;
    }
    return "assets/kokoro/weights";
}

static cactus::kokoro::TextEncoder load_encoder() {
    cactus::kokoro::TextEncoder te;
    const auto dir = find_weights_dir();
    if (!te.load_weights(dir)) {
        throw std::runtime_error("text_encoder load_weights failed (dir=" + dir + ")");
    }
    return te;
}

static bool layer_close(const std::vector<float>& got,
                        const std::vector<float>& expected,
                        float tol, const char* label) {
    if (got.size() != expected.size()) {
        std::fprintf(stderr, "%s: SIZE MISMATCH got=%zu exp=%zu\n",
                     label, got.size(), expected.size());
        return false;
    }
    float max_abs = 0.0f, max_rel = 0.0f;
    int errors = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - expected[i]);
        if (d > max_abs) max_abs = d;
        const float r = d / std::max(std::fabs(expected[i]), 1e-9f);
        if (r > max_rel) max_rel = r;
        if (d > tol && errors < 3) {
            std::fprintf(stderr, "%s: i=%zu got=%g exp=%g abs=%g\n",
                         label, i, got[i], expected[i], d);
            ++errors;
        }
    }
    std::fprintf(stderr,
        "%s: max_abs=%.6g max_rel=%.6g (tol=%.1g) %s\n",
        label, max_abs, max_rel, tol,
        (max_abs <= tol ? "PASS" : "FAIL"));
    return max_abs <= tol;
}

// Run partial forward up to `last`, return the output for fixture `tag`.
static std::vector<float> run_phrase(const cactus::kokoro::TextEncoder& te,
                                     const char* tag,
                                     cactus::kokoro::TextEncoder::Layer last) {
    auto phonemes = load_typed<int64_t>(
        fixture_path((std::string("text_encoder_") + tag + "_input_phonemes.bin").c_str()));
    std::vector<float> out;
    te.run_until_layer(phonemes.data(), static_cast<int>(phonemes.size()), last, out);
    return out;
}

static std::vector<float> load_expected(const char* tag, const char* layer_basename) {
    return load_typed<float>(fixture_path(
        (std::string("text_encoder_") + tag + "_" + layer_basename + ".bin").c_str()));
}

// ----------------------------------------------------------------------
// Per-layer tests (phrase p01)
// ----------------------------------------------------------------------
static bool test_embedding_p01() {
    auto te = load_encoder();
    auto got = run_phrase(te, "p01", cactus::kokoro::TextEncoder::Layer::Embedding);
    auto exp = load_expected("p01", "after_embedding");
    // Embedding is INT8 group-quantized in the .weights files; the
    // converter's round-trip verifier documents <=5e-2 abs rel error.
    // Observed max_abs is ~1.4e-2, well within bound.
    return layer_close(got, exp, 2e-2f, "embedding p01");
}

static bool test_after_cnn0_p01() {
    auto te = load_encoder();
    auto got = run_phrase(te, "p01", cactus::kokoro::TextEncoder::Layer::Cnn0);
    auto exp = load_expected("p01", "after_cnn0");
    return layer_close(got, exp, 1e-2f, "cnn0 p01");
}

static bool test_after_cnn1_p01() {
    auto te = load_encoder();
    auto got = run_phrase(te, "p01", cactus::kokoro::TextEncoder::Layer::Cnn1);
    auto exp = load_expected("p01", "after_cnn1");
    return layer_close(got, exp, 2e-2f, "cnn1 p01");
}

static bool test_after_cnn2_p01() {
    auto te = load_encoder();
    auto got = run_phrase(te, "p01", cactus::kokoro::TextEncoder::Layer::Cnn2);
    auto exp = load_expected("p01", "after_cnn2");
    return layer_close(got, exp, 3e-2f, "cnn2 p01");
}

static bool test_after_lstm_p01() {
    auto te = load_encoder();
    auto got = run_phrase(te, "p01", cactus::kokoro::TextEncoder::Layer::Lstm);
    auto exp = load_expected("p01", "after_lstm");
    return layer_close(got, exp, 5e-2f, "lstm p01");
}

static bool test_e2e_all_phrases() {
    auto te = load_encoder();
    bool ok = true;
    for (const char* tag : {"p01", "p02", "p03", "p04", "p05"}) {
        auto got = run_phrase(te, tag, cactus::kokoro::TextEncoder::Layer::Lstm);
        auto exp = load_expected(tag, "after_lstm");
        char label[64];
        std::snprintf(label, sizeof(label), "e2e %s", tag);
        ok = layer_close(got, exp, 5e-2f, label) && ok;
    }
    return ok;
}

int main() {
    TestUtils::TestRunner runner("cactus::kokoro::TextEncoder");
    try {
        runner.run_test("Embedding (p01)",            test_embedding_p01());
        runner.run_test("After CNN block 0 (p01)",    test_after_cnn0_p01());
        runner.run_test("After CNN block 1 (p01)",    test_after_cnn1_p01());
        runner.run_test("After CNN block 2 (p01)",    test_after_cnn2_p01());
        runner.run_test("After LSTM (p01)",           test_after_lstm_p01());
        runner.run_test("End-to-end 5 phrases",       test_e2e_all_phrases());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Test threw: %s\n", e.what());
        runner.run_test("Test threw exception", false);
    }
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
