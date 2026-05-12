#pragma once

#include "kokoro_internal.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Kokoro `predictor` (StyleTTS2 ProsodyPredictor) — bert_encoder output `d_en`
// + 128-d style `s` -> per-phoneme durations + per-mel-frame F0/N curves.
//
// See assets/kokoro/predictor_architecture.md for the canonical PyTorch
// reference. Inference-only, batch=1, no padding mask.
//
// Matches the .weights basenames produced by python/src/converter_kokoro.py
// (file convention `predictor_*.weights`, 106 files total).
class Predictor {
public:
    // Stage enum for partial-forward debugging / per-stage validation.
    enum class Stage {
        DurEncLstm0,        // [T, 512] post-LSTM
        DurEncAln1,         // [T, 512] post-AdaLayerNorm
        DurEncLstm2,
        DurEncAln3,
        DurEncLstm4,
        DurEncAln5,
        D,                  // [T, 640] = DurationEncoder final
        TopLstm,            // [T, 512]
        DurationProj,       // [T, 50]
        Durations,          // [T] int64  (uses out_durations)
        Alignment,          // [T, T_mel] f32
        En,                 // [T_mel, 640]
        SharedLstm,         // [T_mel, 512]
        F0Block0, F0Block1, F0Block2, F0Pred,  // F0Block{0,1,2}: [Ti, Ci]; F0Pred: [2*T_mel]
        NBlock0,  NBlock1,  NBlock2,  NPred,
    };

    struct Output {
        std::vector<float>   F0;        // 2 * T_mel
        std::vector<float>   N;         // 2 * T_mel
        std::vector<int64_t> durations; // T
        int                  T_mel = 0;
    };

    static constexpr int STYLE_DIM   = 128;
    static constexpr int D_HID       = 512;       // d_model
    static constexpr int D_CAT       = 640;       // d_model + style_dim
    static constexpr int LSTM_HIDDEN = 256;       // each direction; concat -> 512
    static constexpr int MAX_DUR     = 50;
    static constexpr float LAYERNORM_EPS = 1e-5f;
    static constexpr float INSTNORM_EPS  = 1e-5f;
    static constexpr float LEAKY_SLOPE   = 0.2f;

    bool load_weights(const std::string& weights_dir);

    // Full forward: phonemes (unused at runtime but kept for parity with the
    // reference signature; predictor sees only `d_en` and `s`), `style128`,
    // and `d_en` (T, 512) row-major.
    void forward(const int64_t* phonemes, int T,
                 const float* style128,
                 const float* d_en,
                 Output& out) const;

    // Partial forward — runs through `last`, populating `out_buf` flattened
    // (Ti, dim) row-major (or copying into `out_durations` for the Durations
    // stage). Stages that produce length-doubled buffers (F0Block1/2, F0Pred,
    // N variants, Alignment, En, SharedLstm) follow the fixture layout in
    // tests/fixtures/kokoro_predictor_reference.py.
    void run_until_stage(const int64_t* phonemes, int T,
                         const float* style128,
                         const float* d_en,
                         Stage last,
                         std::vector<float>& out_buf,
                         std::vector<int64_t>* out_durations = nullptr) const;

private:
    // ---- Bidirectional LSTM weight bundle (one direction). Gates [i,f,g,o].
    struct LstmDir {
        std::vector<float> w_ih;   // (4H, I)
        std::vector<float> w_hh;   // (4H, H)
        std::vector<float> b_ih;   // (4H,)
        std::vector<float> b_hh;   // (4H,)
    };
    struct BiLSTM {
        LstmDir fwd, bwd;
    };

    // ---- AdaLayerNorm fc projection / Conv1d / AdainResBlk1d are shared
    // helpers in kokoro_internal.h (decoder.h reuses the same building blocks).
    using AdaLNFc       = internal::AdaLNFc;
    using Conv1d        = internal::Conv1d;
    using AdainResBlk1d = internal::AdainResBlk1d;

    // ---- DurationEncoder
    BiLSTM   dur_enc_lstm_[3];     // BiLSTM(640->512) x 3
    AdaLNFc  dur_enc_aln_[3];      // AdaLayerNorm(128, 512) x 3

    // ---- Top-level
    BiLSTM   top_lstm_;            // BiLSTM(640->512)
    std::vector<float> dur_proj_w_; // (50, 512)
    std::vector<float> dur_proj_b_; // (50,)

    // ---- F0 / N path
    BiLSTM   shared_lstm_;          // BiLSTM(640->512)
    AdainResBlk1d F0_blocks_[3];   // (512->512), (512->256, upsample), (256->256)
    AdainResBlk1d N_blocks_[3];
    Conv1d  F0_proj_;              // (1, 256, 1)
    Conv1d  N_proj_;               // (1, 256, 1)

};

}}  // namespace cactus::kokoro
