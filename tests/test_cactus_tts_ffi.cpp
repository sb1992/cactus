// Smoke test for the cactus_tts C FFI. Verifies symbols link, the
// create/synth/destroy lifecycle works, the size-query protocol returns
// a nonzero required size, and the actual synth produces non-silent audio.

#include "test_utils.h"
#include "../cactus/ffi/cactus_tts.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::string find_path(const char* const* candidates, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        std::ifstream f(candidates[i], std::ios::binary);
        if (f) return candidates[i];
    }
    return n > 0 ? candidates[0] : std::string();
}

static std::string find_weights_dir() {
    const char* candidates[] = {
        "assets/kokoro/weights/bert_word_embeddings.weights",
        "../assets/kokoro/weights/bert_word_embeddings.weights",
        "../../assets/kokoro/weights/bert_word_embeddings.weights",
    };
    std::string marker = find_path(candidates, sizeof(candidates) / sizeof(candidates[0]));
    // Strip the trailing "/bert_word_embeddings.weights"
    auto pos = marker.find_last_of('/');
    return (pos == std::string::npos) ? std::string() : marker.substr(0, pos);
}

static std::string find_g2p_dict() {
    const char* candidates[] = {
        "assets/kokoro/kokoro-g2p.dict",
        "../assets/kokoro/kokoro-g2p.dict",
        "../../assets/kokoro/kokoro-g2p.dict",
    };
    return find_path(candidates, sizeof(candidates) / sizeof(candidates[0]));
}

static std::string find_voice_path() {
    const char* candidates[] = {
        "tests/fixtures/data/voice_af_bella_raw.bin",
        "../fixtures/data/voice_af_bella_raw.bin",
        "../../tests/fixtures/data/voice_af_bella_raw.bin",
        "fixtures/data/voice_af_bella_raw.bin",
    };
    return find_path(candidates, sizeof(candidates) / sizeof(candidates[0]));
}

static bool test_create_synth_destroy() {
    const std::string weights = find_weights_dir();
    const std::string g2p     = find_g2p_dict();
    const std::string voice   = find_voice_path();

    if (weights.empty() || g2p.empty() || voice.empty()) {
        std::fprintf(stderr, "[skip] kokoro assets not found "
                              "(weights='%s' g2p='%s' voice='%s')\n",
                              weights.c_str(), g2p.c_str(), voice.c_str());
        return false;
    }

    auto* h = cactus_tts_create(weights.c_str(), g2p.c_str(), voice.c_str());
    if (!h) {
        std::fprintf(stderr, "create failed\n");
        return false;
    }
    if (cactus_tts_sample_rate(h) != 24000) {
        std::fprintf(stderr, "unexpected sample rate\n");
        cactus_tts_destroy(h);
        return false;
    }

    // Size query: pass NULL out + n=0; expect rc != 0 and n > 0.
    size_t n = 0;
    int rc = cactus_tts_synthesize(h, "Hello.", 1.0f, nullptr, &n);
    if (rc == 0 || n == 0) {
        std::fprintf(stderr, "size query weird rc=%d n=%zu\n", rc, n);
        cactus_tts_destroy(h);
        return false;
    }

    // Real synth into a properly sized buffer.
    std::vector<float> audio(n);
    rc = cactus_tts_synthesize(h, "Hello.", 1.0f, audio.data(), &n);
    cactus_tts_destroy(h);
    if (rc != 0) {
        std::fprintf(stderr, "synth failed rc=%d\n", rc);
        return false;
    }

    // Sanity: nontrivial energy.
    double sum_sq = 0;
    for (float s : audio) sum_sq += static_cast<double>(s) * s;
    double mean_sq = sum_sq / static_cast<double>(audio.size());
    if (mean_sq < 1e-6) {
        std::fprintf(stderr, "audio is silence (mean_sq=%g)\n", mean_sq);
        return false;
    }
    return true;
}

// Read a [u32 n][i64 * n] phoneme-id fixture (the format predictor_p<NN>_input_phonemes.bin uses).
static bool read_phoneme_id_fixture(const std::string& path, std::vector<int64_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!f || n == 0 || n > (1u << 20)) return false;
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n) * sizeof(int64_t));
    return static_cast<bool>(f);
}

static std::string find_p01_phonemes_path() {
    const char* candidates[] = {
        "tests/fixtures/data/predictor_p01_input_phonemes.bin",
        "../fixtures/data/predictor_p01_input_phonemes.bin",
        "../../tests/fixtures/data/predictor_p01_input_phonemes.bin",
        "fixtures/data/predictor_p01_input_phonemes.bin",
    };
    return find_path(candidates, sizeof(candidates) / sizeof(candidates[0]));
}

static bool test_synthesize_phonemes_p01() {
    const std::string weights = find_weights_dir();
    const std::string g2p     = find_g2p_dict();
    const std::string voice   = find_voice_path();
    const std::string fix     = find_p01_phonemes_path();

    if (weights.empty() || g2p.empty() || voice.empty() || fix.empty()) {
        std::fprintf(stderr,
            "[skip] kokoro assets not found "
            "(weights='%s' g2p='%s' voice='%s' fixture='%s')\n",
            weights.c_str(), g2p.c_str(), voice.c_str(), fix.c_str());
        return false;
    }

    std::vector<int64_t> ids;
    if (!read_phoneme_id_fixture(fix, ids)) {
        std::fprintf(stderr, "failed to read p01 fixture %s\n", fix.c_str());
        return false;
    }
    if (ids.empty() || ids.front() != 0 || ids.back() != 0) {
        std::fprintf(stderr, "fixture missing leading/trailing pad (front=%lld back=%lld n=%zu)\n",
                     (long long)ids.front(), (long long)ids.back(), ids.size());
        return false;
    }

    auto* h = cactus_tts_create(weights.c_str(), g2p.c_str(), voice.c_str());
    if (!h) { std::fprintf(stderr, "create failed\n"); return false; }

    // Size query.
    size_t n = 0;
    int rc = cactus_tts_synthesize_phonemes(h, ids.data(), ids.size(), 1.0f,
                                            nullptr, &n);
    if (rc == 0 || n == 0) {
        std::fprintf(stderr, "phonemes size query weird rc=%d n=%zu\n", rc, n);
        cactus_tts_destroy(h);
        return false;
    }

    std::vector<float> audio(n);
    rc = cactus_tts_synthesize_phonemes(h, ids.data(), ids.size(), 1.0f,
                                        audio.data(), &n);
    if (rc != 0) {
        std::fprintf(stderr, "phonemes synth failed rc=%d\n", rc);
        cactus_tts_destroy(h);
        return false;
    }

    // Reject speed != 1.0
    size_t n2 = audio.size();
    int rc_speed = cactus_tts_synthesize_phonemes(h, ids.data(), ids.size(), 1.5f,
                                                  audio.data(), &n2);
    cactus_tts_destroy(h);
    if (rc_speed != -4) {
        std::fprintf(stderr, "expected rc=-4 for speed!=1.0, got %d\n", rc_speed);
        return false;
    }

    double sum_sq = 0;
    for (float s : audio) sum_sq += static_cast<double>(s) * s;
    double mean_sq = sum_sq / static_cast<double>(audio.size());
    if (mean_sq < 1e-6) {
        std::fprintf(stderr, "phonemes audio is silence (mean_sq=%g)\n", mean_sq);
        return false;
    }
    return true;
}

int main() {
    TestUtils::TestRunner r("cactus_tts FFI");
    r.run_test("create + synth + destroy", test_create_synth_destroy());
    r.run_test("synthesize_phonemes (p01 misaki ids)", test_synthesize_phonemes_p01());
    r.print_summary();
    return r.all_passed() ? 0 : 1;
}
