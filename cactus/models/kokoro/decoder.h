#pragma once

#include "kokoro_internal.h"

#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Stage 1 of the Kokoro decoder (encode + 4 decode blocks).
// The generator (Hi-Fi-GAN with Snake1d, harmonic source, MRF resblocks,
// and iSTFT) lands in Task 8c as a separate class. This class covers:
//
//   F0 = F0_conv(F0_curve.unsqueeze(1))      # weight-normed Conv1d k=3 s=2 p=1
//   N  = N_conv (N.unsqueeze(1))
//   x  = cat([asr, F0, N], axis=1)            # (1, 514, T_mel)
//   x  = encode(x, s)                         # AdainResBlk1d 514 -> 1024
//   asr_res = asr_res(asr)                    # 1x1 Conv1d 512 -> 64
//   for block in decode (4 AdainResBlk1d):
//       x = cat([x, asr_res, F0, N], axis=1)  # 1024 + 64 + 1 + 1 = 1090
//       x = block(x, s)
//   # decode.3 has upsample=True (depthwise ConvT1d 2x), so output has
//   # 2*T_mel time steps and 512 channels.
//
// See assets/kokoro/decoder_architecture.md for the canonical reference.
class DecoderEncodeStage {
public:
    enum class Stage {
        F0AfterConv,       // (T_mel,)        — F0_conv output, 1 channel
        NAfterConv,        // (T_mel,)        — N_conv output, 1 channel
        AfterEncode,       // (T_mel, 1024)   — encode(x, s) row-major
        AsrRes,            // (T_mel, 64)     — asr_res(asr) row-major
        AfterDecode0,      // (T_mel, 1024)
        AfterDecode1,      // (T_mel, 1024)
        AfterDecode2,      // (T_mel, 1024)
        AfterDecode3,      // (2*T_mel, 512)  — note 2x upsample
    };

    static constexpr int STYLE_DIM     = 128;
    static constexpr int ASR_CH        = 512;     // asr input channel count
    static constexpr int ASR_RES_CH    = 64;      // asr_res output channels
    static constexpr int ENCODE_OUT_CH = 1024;
    static constexpr int DECODE_OUT_CH = 1024;    // decode.0..2 channels
    static constexpr int DECODE3_OUT_CH = 512;    // decode.3 output channels
    static constexpr float LEAKY_SLOPE = 0.2f;
    static constexpr float INSTNORM_EPS = 1e-5f;

    bool load_weights(const std::string& weights_dir);

    // Full forward up through decode.3. Output is (2*T_mel, 512) row-major.
    //   asr     : (T_mel, 512) row-major   — input from text_encoder (transposed)
    //   F0      : (2*T_mel,)                — predictor F0_pred
    //   N       : (2*T_mel,)                — predictor N_pred
    //   style128: (128,)                    — first half of voice embedding
    void forward(const float* asr, int T_mel,
                 const float* F0, const float* N,
                 const float* style128,
                 std::vector<float>& out) const;

    // Partial forward — runs through `last`, populating `out` in the layout
    // documented for each Stage above.
    void run_until_stage(const float* asr, int T_mel,
                         const float* F0, const float* N,
                         const float* style128,
                         Stage last,
                         std::vector<float>& out) const;

private:
    using Conv1d        = internal::Conv1d;
    using AdaLNFc       = internal::AdaLNFc;
    using AdainResBlk1d = internal::AdainResBlk1d;

    Conv1d        F0_conv_;       // (1, 1, 3) k=3 s=2 p=1 + bias (1,)
    Conv1d        N_conv_;        // same
    AdainResBlk1d encode_;        // 514 -> 1024, no upsample, learned shortcut
    Conv1d        asr_res_;       // 1x1 conv 512 -> 64 + bias
    AdainResBlk1d decode_[4];     // 0..2: 1090 -> 1024 (no up); 3: 1090 -> 512 (up)
};

}}  // namespace cactus::kokoro
