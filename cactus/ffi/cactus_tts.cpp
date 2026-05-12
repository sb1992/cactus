// Thin C FFI over cactus::kokoro::KokoroModel. No model logic here —
// just object lifetime, argument validation, and a size-query/copy
// protocol for the audio buffer.

#include "cactus_tts.h"
#include "../models/kokoro/model.h"

#include <cstring>
#include <new>
#include <string>
#include <vector>

struct cactus_tts {
    cactus::kokoro::KokoroModel model;
};

extern "C" cactus_tts* cactus_tts_create(const char* weights_dir,
                                         const char* g2p_dict_path,
                                         const char* voice_path) {
    if (!weights_dir || !g2p_dict_path || !voice_path) return nullptr;
    auto* h = new (std::nothrow) cactus_tts;
    if (!h) return nullptr;
    if (!h->model.load(weights_dir, g2p_dict_path, voice_path)) {
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" void cactus_tts_destroy(cactus_tts* h) { delete h; }

extern "C" int cactus_tts_synthesize(cactus_tts* h,
                                     const char* text, float speed,
                                     float* out, size_t* out_n) {
    if (!h || !text || !out_n) return -1;
    std::vector<float> buf;
    if (!h->model.synthesize(std::string(text), speed, buf)) return -2;
    const size_t cap = *out_n;
    *out_n = buf.size();
    if (out == nullptr || cap < buf.size()) return -3;
    std::memcpy(out, buf.data(), buf.size() * sizeof(float));
    return 0;
}

extern "C" int cactus_tts_synthesize_phonemes(cactus_tts* h,
                                              const int64_t* ids, size_t n_ids,
                                              float speed,
                                              float* out, size_t* out_n) {
    if (!h || !ids || !out_n) return -1;
    if (speed != 1.0f) return -4;
    std::vector<float> buf;
    if (!h->model.synthesize_with_phonemes(ids, static_cast<int>(n_ids), buf)) return -2;
    const size_t cap = *out_n;
    *out_n = buf.size();
    if (out == nullptr || cap < buf.size()) return -3;
    std::memcpy(out, buf.data(), buf.size() * sizeof(float));
    return 0;
}

extern "C" int cactus_tts_sample_rate(cactus_tts* h) {
    (void)h;
    return cactus::kokoro::KokoroModel::SAMPLE_RATE;
}
