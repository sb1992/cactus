#pragma once

#include "g2p.h"
#include "text_encoder.h"
#include "bert.h"
#include "predictor.h"
#include "decoder.h"
#include "generator.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Top-level Kokoro TTS pipeline. Wires together:
//   text -> G2P phonemize -> phoneme IDs (i64)
//   phoneme IDs -> text_encoder -> text features (T, 512)
//   phoneme IDs -> bert (ALBERT-tiny) -> bert features (T, 768)
//   bert features -> bert_encoder linear -> d_en (T, 512)
//   d_en + style[128:] -> predictor -> durations, F0, N
//   durations -> alignment -> text_encoder * alignment -> asr (T_mel, 512)
//   asr + F0 + N + style[:128] -> decoder.encode -> (2*T_mel, 512)
//   + F0 + style[:128] -> generator -> 24 kHz PCM
//
// See assets/kokoro/kokoromodel_pipeline.md for the reference data flow.
class KokoroModel {
public:
    static constexpr int SAMPLE_RATE = 24000;
    static constexpr int VOICE_SLICES = 511;
    static constexpr int VOICE_DIM    = 256;       // 128-d style x 2 halves
    static constexpr int STYLE_DIM    = 128;

    // Load all five sub-models from a flat weights directory + the G2P
    // dictionary + the voices.bin embedding file. Returns true on success.
    bool load(const std::string& weights_dir,
              const std::string& g2p_dict_path,
              const std::string& voice_path);

    // Synthesize text -> 24 kHz mono PCM (production API).
    //   text:   English text (G2P internally)
    //   speed:  speech rate multiplier (1.0 = normal)
    //   audio:  populated with samples, length 600 * T_mel
    //   seed:   PRNG seed for SineGen (0 -> std::mt19937 default seed = 0)
    bool synthesize(const std::string& text, float speed,
                    std::vector<float>& audio,
                    uint64_t seed = 0) const;

    // Test mode: bypass G2P, take phoneme IDs directly. Optionally inject a
    // pre-computed har_source to bypass SineGen PRNG (used by golden tests).
    //   phoneme_ids: KModel-padded (caller must include the leading and
    //                trailing 0). T = n.
    //   speed:       speech rate (default 1.0)
    //   audio:       populated with samples
    //   har_source:  if non-null, must have har_len = 600 * T_mel samples
    bool synthesize_with_phonemes(const int64_t* phoneme_ids, int n,
                                  std::vector<float>& audio,
                                  float speed = 1.0f,
                                  const float* har_source = nullptr,
                                  int har_len = 0,
                                  uint64_t seed = 0) const;

    // Phoneme / style helpers (exposed for KokoroModel-level tests).
    const TextEncoder& text_encoder() const { return text_encoder_; }
    const Bert&        bert()         const { return bert_; }
    const Predictor&   predictor()    const { return predictor_; }
    const G2P&         g2p()          const { return g2p_; }

    // Get the (128,) style vector for the bert/predictor branch given a
    // phoneme count INCLUDING the KModel pad tokens. Returns false if T is out
    // of voice-embedding range.
    bool get_style_vectors(int T,
                           std::vector<float>& s_predictor,  // (128,)
                           std::vector<float>& s_decoder)    // (128,)
                          const;

private:
    G2P                 g2p_;
    TextEncoder         text_encoder_;
    Bert                bert_;
    Predictor           predictor_;
    DecoderEncodeStage  decoder_encode_;
    Generator           generator_;

    // Voice embedding (511, 1, 256) flattened to (511 * 256) row-major.
    std::vector<float>  voice_embedding_;
};

}}  // namespace cactus::kokoro
