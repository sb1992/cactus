#include "model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

namespace {

// Load a flat float32 raw binary into `out`. Returns false on failure.
bool load_raw_floats(const std::string& path, std::vector<float>& out,
                     size_t expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "KokoroModel: cannot open voice file '%s'\n", path.c_str());
        return false;
    }
    const auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    if (size != expected_count * sizeof(float)) {
        std::fprintf(stderr,
            "KokoroModel: voice file '%s' size %zu B, expected %zu B "
            "(=%zu floats)\n",
            path.c_str(), size, expected_count * sizeof(float), expected_count);
        return false;
    }
    out.resize(expected_count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expected_count * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "KokoroModel: short read on '%s'\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace

bool KokoroModel::load(const std::string& weights_dir,
                       const std::string& g2p_dict_path,
                       const std::string& voice_path) {
    if (!g2p_.load(g2p_dict_path)) {
        std::fprintf(stderr, "KokoroModel::load: G2P load failed (%s)\n",
                     g2p_dict_path.c_str());
        return false;
    }
    if (!text_encoder_.load_weights(weights_dir)) return false;
    if (!bert_.load_weights(weights_dir))         return false;
    if (!predictor_.load_weights(weights_dir))    return false;
    if (!decoder_encode_.load_weights(weights_dir)) return false;
    if (!generator_.load_weights(weights_dir))    return false;
    if (!load_raw_floats(voice_path, voice_embedding_,
                         static_cast<size_t>(VOICE_SLICES) * VOICE_DIM)) {
        return false;
    }
    return true;
}

bool KokoroModel::get_style_vectors(int T,
                                    std::vector<float>& s_predictor,
                                    std::vector<float>& s_decoder) const {
    if (T <= 0 || T > VOICE_SLICES) return false;
    if (voice_embedding_.size() !=
        static_cast<size_t>(VOICE_SLICES) * VOICE_DIM) return false;
    const float* slice = voice_embedding_.data() +
                         static_cast<size_t>(T - 1) * VOICE_DIM;
    s_decoder.assign(slice, slice + STYLE_DIM);                  // [:128]
    s_predictor.assign(slice + STYLE_DIM, slice + VOICE_DIM);    // [128:]
    return true;
}

bool KokoroModel::synthesize_with_phonemes(const int64_t* phoneme_ids, int n,
                                           std::vector<float>& audio,
                                           float speed,
                                           const float* har_source,
                                           int har_len,
                                           uint64_t seed) const {
    audio.clear();
    if (n <= 0 || phoneme_ids == nullptr) return false;
    const int T = n;

    // ---- 1. Voice slice -----------------------------------------------------
    std::vector<float> s_predictor, s_decoder;
    if (!get_style_vectors(T, s_predictor, s_decoder)) {
        std::fprintf(stderr, "KokoroModel: T=%d out of voice range\n", T);
        return false;
    }

    // ---- 2. text_encoder ----------------------------------------------------
    std::vector<float> t_en;          // (T, 512) row-major
    text_encoder_.forward(phoneme_ids, T, t_en);

    // ---- 3. bert + bert_encoder --------------------------------------------
    std::vector<float> bert_dur;      // (T, 768)
    bert_.forward(phoneme_ids, T, bert_dur);
    std::vector<float> d_en;          // (T, 512) row-major
    bert_.apply_bert_encoder(bert_dur.data(), T, d_en);

    // ---- 4. predictor -------------------------------------------------------
    Predictor::Output pred;
    predictor_.forward(phoneme_ids, T, s_predictor.data(), d_en.data(), pred);
    const int T_mel = pred.T_mel;
    if (T_mel <= 0) return false;

    // Apply `speed` scaling externally if needed. The Predictor's forward
    // currently bakes speed=1.0; for a per-text speed, scale durations here.
    if (std::fabs(speed - 1.0f) > 1e-6f) {
        // Reproduce KModel's behavior: divide pre-round duration by speed.
        // Predictor::forward already applied speed=1, so we'd need to
        // re-run; mark as unsupported for now.
        std::fprintf(stderr,
            "KokoroModel: speed != 1.0 not yet supported (got %g)\n", speed);
        return false;
    }

    // ---- 5. asr = (t_en^T @ alignment)^T  -> (T_mel, 512) ------------------
    // Build alignment from durations (same as predictor internally).
    // pred.durations: (T,) i64; T_mel = sum.
    std::vector<float> asr(static_cast<size_t>(T_mel) * TextEncoder::HIDDEN, 0.0f);
    {
        const int C = TextEncoder::HIDDEN;
        int m = 0;
        for (int p = 0; p < T; ++p) {
            const int dur = static_cast<int>(pred.durations[p]);
            for (int k = 0; k < dur; ++k) {
                // copy t_en[p, :] into asr[m, :]
                std::memcpy(asr.data() + static_cast<size_t>(m) * C,
                            t_en.data() + static_cast<size_t>(p) * C,
                            sizeof(float) * C);
                ++m;
            }
        }
        if (m != T_mel) {
            std::fprintf(stderr,
                "KokoroModel: alignment build mismatch: m=%d T_mel=%d\n",
                m, T_mel);
            return false;
        }
    }

    // ---- 6. decoder.encode (encode + 4 decode blocks) ----------------------
    std::vector<float> x_decode3;     // (2*T_mel, 512)
    decoder_encode_.forward(asr.data(), T_mel,
                            pred.F0.data(), pred.N.data(),
                            s_decoder.data(),
                            x_decode3);

    // ---- 7. generator -> audio ---------------------------------------------
    if (har_source != nullptr) {
        if (har_len != 600 * T_mel) {
            std::fprintf(stderr,
                "KokoroModel: har_len=%d, expected 600*T_mel=%d\n",
                har_len, 600 * T_mel);
            return false;
        }
        generator_.forward_with_har_source(x_decode3.data(), T_mel,
                                           har_source, har_len,
                                           s_decoder.data(),
                                           audio);
    } else {
        generator_.forward(x_decode3.data(), T_mel,
                           pred.F0.data(), s_decoder.data(),
                           audio, seed);
    }

    return true;
}

bool KokoroModel::synthesize(const std::string& text, float speed,
                             std::vector<float>& audio,
                             uint64_t seed) const {
    audio.clear();

    // G2P (ARPAbet phonemes) -> map to Kokoro vocab IDs.
    // NOTE: Kokoro's vocab is IPA, while our local CMUdict-derived G2P
    // produces ARPAbet. The full ARPAbet->IPA mapping is implemented as a
    // best-effort lookup table below; quality depends on misaki for production
    // use. For golden-fixture testing, the test path uses
    // synthesize_with_phonemes() with pre-computed misaki phoneme IDs.
    const auto phs = g2p_.phonemize(text);
    if (phs.empty()) {
        std::fprintf(stderr, "KokoroModel::synthesize: empty phonemization\n");
        return false;
    }

    // ARPAbet (CMU 39-phone set) -> Kokoro IPA vocab. The mapping below is
    // taken from the canonical IPA equivalents used by misaki/Kokoro's vocab
    // (see hexgrad/Kokoro-82M config.json). Vowels collapse stress so this
    // list mirrors arpabet entries WITHOUT stress digits.
    //
    // Kokoro vocab IDs were extracted from
    // ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/.../config.json.
    // For unmapped tokens we drop silently (consistent with misaki's behavior
    // when fed unsupported chars).
    static const struct { const char* arpa; int id; } MAPPING[] = {
        // Punctuation pseudo-tokens (preserved by G2P::phonemize)
        {".", 4}, {"!", 5}, {"?", 6}, {",", 3}, {";", 1}, {":", 2},
        // Stops
        {"P", 56}, {"B", 43}, {"T", 36}, {"D", 102}, {"K", 53}, {"G", 47},
        // Affricates
        {"CH", 23}, {"JH", 19},
        // Fricatives
        {"F", 46}, {"V", 38}, {"TH", 81}, {"DH", 28}, {"S", 35}, {"Z", 42},
        {"SH", 86}, {"ZH", 67}, {"HH", 50},
        // Nasals
        {"M", 54}, {"N", 55}, {"NG", 87},
        // Liquids / glides
        {"L", 51}, {"R", 156}, {"Y", 41}, {"W", 39},
        // Vowels (IPA equivalents in Kokoro vocab)
        {"AA", 24}, {"AE", 65}, {"AH", 42}, {"AO", 31}, {"AW", 64}, {"AY", 92},
        {"EH", 45}, {"ER", 62}, {"EY", 76}, {"IH", 79}, {"IY", 75},
        {"OW", 83}, {"OY", 119}, {"UH", 96}, {"UW", 75},
    };

    std::vector<int64_t> ids;
    ids.reserve(phs.size() + 2);
    ids.push_back(0);  // KModel leading pad
    for (const auto& p : phs) {
        bool found = false;
        for (const auto& m : MAPPING) {
            if (p == m.arpa) {
                ids.push_back(static_cast<int64_t>(m.id));
                found = true;
                break;
            }
        }
        if (!found) {
            // Silently drop — mirrors misaki on unsupported chars.
        }
    }
    ids.push_back(0);  // KModel trailing pad

    if (ids.size() <= 2) {
        std::fprintf(stderr,
            "KokoroModel::synthesize: no phonemes mapped to Kokoro vocab\n");
        return false;
    }

    return synthesize_with_phonemes(ids.data(),
                                    static_cast<int>(ids.size()),
                                    audio, speed,
                                    /*har_source=*/nullptr, 0,
                                    seed);
}

}}  // namespace cactus::kokoro
