#pragma once

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

    // ---- AdaLayerNorm fc projection: Linear(STYLE_DIM=128 -> 2*C)
    struct AdaLNFc {
        std::vector<float> w;      // (2C, 128)
        std::vector<float> b;      // (2C,)
    };

    // ---- Conv1d weight bundle (folded weight_norm)
    struct Conv1d {
        std::vector<float> w;      // (C_out, C_in, K)
        std::vector<float> b;      // (C_out,)  (may be empty for no-bias variants)
        int C_in = 0, C_out = 0, K = 0;
    };

    // ---- AdainResBlk1d — F0/N building block. Optional `pool` (ConvTranspose1d)
    // and optional `conv1x1` (1x1 Conv1d, no bias) when dim_in != dim_out.
    struct AdainResBlk1d {
        int dim_in  = 0;
        int dim_out = 0;
        bool upsample = false;     // 'nearest' if true, 'none' if false
        bool learned_sc = false;   // dim_in != dim_out

        AdaLNFc norm1_fc;          // for AdaIN1d on dim_in
        AdaLNFc norm2_fc;          // for AdaIN1d on dim_out
        Conv1d  conv1;             // (dim_out, dim_in, K=3, p=1)
        Conv1d  conv2;             // (dim_out, dim_out, K=3, p=1)
        Conv1d  pool;              // ConvTranspose1d(dim_in, dim_in, K=3, s=2, groups=dim_in,
                                   //                  padding=1, output_padding=1)
                                   // Stored weight shape: (dim_in, 1, 3) PyTorch convention
                                   // (Conv*Transpose1d with groups=in, in_channels_per_group=1).
        Conv1d  conv1x1;           // (dim_out, dim_in, 1, 1, 0, no bias)  — only if learned_sc
    };

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

    // Helpers --------------------------------------------------------------
    // AdaLayerNorm (in DurationEncoder): in/out is (1, 512, T) channel-first.
    // No learned affine on the LayerNorm itself; modulation comes from the
    // fc(s) -> [gamma, beta] projection.
    void ada_layer_norm(float* x, int C, int T, const float* style128,
                        const AdaLNFc& fc) const;

    // AdaIN1d (in F0/N branches): in/out is (1, C, T) channel-first.
    // InstanceNorm1d (per-channel across T) with no learned affine in this
    // checkpoint (the v0_19 .pth lacks `.norm.weight/.norm.bias`; PyTorch
    // initializes them to (1, 0), an identity affine).
    void ada_in_1d(float* x, int C, int T, const float* style128,
                   const AdaLNFc& fc) const;

    // Run one AdainResBlk1d (1, dim_in, T) -> (1, dim_out, Tout)
    // where Tout = T (no upsample) or 2*T (upsample='nearest').
    void run_adain_res_blk(const AdainResBlk1d& blk, const float* style128,
                           const std::vector<float>& in_cf, int T,
                           std::vector<float>& out_cf, int& T_out) const;

    // Pool for upsample=nearest blocks: depthwise ConvTranspose1d
    //   stride=2, kernel=3, groups=dim_in, padding=1, output_padding=1
    // Input (C, T) channel-first; output (C, 2T) channel-first.
    // Output element formula (PyTorch):
    //   y[c, t_out] = bias[c] + sum over kernel positions k where
    //     a valid input index i_in = ((t_out + padding - k) / stride) exists
    //     and (t_out + padding - k) % stride == 0
    //   y[c, t_out] += x[c, i_in] * w[c, 0, k]
    // (groups=in, in_channels_per_group=1, out_channels_per_group=1.)
    void conv_transpose_1d_pool(const Conv1d& pool,
                                const std::vector<float>& in_cf, int C, int T,
                                std::vector<float>& out_cf) const;

    // Nearest-neighbor upsample by factor 2 (for the shortcut path).
    // Input (C, T) -> output (C, 2T) with each input copied twice.
    void upsample_nearest_2x(const std::vector<float>& in_cf, int C, int T,
                             std::vector<float>& out_cf) const;
};

}}  // namespace cactus::kokoro
