#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Kokoro `text_encoder` (StyleTTS2 lineage) — phoneme IDs -> per-phoneme
// hidden states. See assets/kokoro/text_encoder_architecture.md for the
// canonical PyTorch reference this port reproduces.
//
// Inference-only, batch=1, no padding mask (Kokoro never batches phonemes).
// Layout matches the reference layer-by-layer (see fixtures
// `tests/fixtures/data/text_encoder_p{01..05}_*.bin`).
class TextEncoder {
public:
    // Layer enum for partial-forward debugging.
    enum class Layer {
        Embedding,
        Cnn0,
        Cnn1,
        Cnn2,
        Lstm,  // full forward
    };

    // Load all 21 .weights files from `weights_dir/text_encoder_*.weights`.
    // Returns true on success.
    bool load_weights(const std::string& weights_dir);

    // Full forward: phoneme IDs -> per-phoneme hidden states.
    // `out` is resized to T*512 contiguous (row-major, shape (T, 512)).
    void forward(const int64_t* phonemes, int T, std::vector<float>& out) const;

    // Partial forward — runs through `last` and stops, populating `out`
    // with the corresponding intermediate flattened (T, 512) row-major.
    // (Internally CNN ops live in (HIDDEN, T) — we transpose back to
    // (T, HIDDEN) before returning, matching the Python fixture layout.)
    void run_until_layer(const int64_t* phonemes, int T, Layer last,
                         std::vector<float>& out) const;

    static constexpr int VOCAB        = 178;
    static constexpr int HIDDEN       = 512;
    static constexpr int LSTM_HIDDEN  = 256;  // each direction; concat -> 512
    static constexpr int CNN_KERNEL   = 5;
    static constexpr int CNN_PADDING  = 2;
    static constexpr float LEAKY_SLOPE = 0.2f;
    static constexpr float LAYERNORM_EPS = 1e-5f;

private:
    // Embedding table: VOCAB * HIDDEN floats (row-major (VOCAB, HIDDEN))
    std::vector<float> embedding_;

    // CNN block parameters (3 blocks, identical shape):
    //   conv weight: (HIDDEN_out=512, HIDDEN_in=512, K=5)
    //   conv bias:   (HIDDEN,)
    //   norm gamma:  (HIDDEN,)
    //   norm beta:   (HIDDEN,)
    struct CnnBlock {
        std::vector<float> conv_weight;  // 512 * 512 * 5
        std::vector<float> conv_bias;    // 512
        std::vector<float> norm_gamma;   // 512
        std::vector<float> norm_beta;    // 512
    };
    CnnBlock cnn_[3];

    // Bidirectional LSTM (single layer):
    //   For each direction:
    //     W_ih: (4*256, 512)   row-major
    //     W_hh: (4*256, 256)
    //     b_ih: (4*256,)
    //     b_hh: (4*256,)
    //   Gates ordered [i, f, g, o] (PyTorch standard, matches cactus_lstm_cell).
    struct LstmDir {
        std::vector<float> w_ih;
        std::vector<float> w_hh;
        std::vector<float> b_ih;
        std::vector<float> b_hh;
    };
    LstmDir lstm_fwd_;
    LstmDir lstm_bwd_;

    // Internals
    void embed_lookup(const int64_t* phonemes, int T, std::vector<float>& out) const;
    // Apply ONE CNN block on a (HIDDEN, T) channel-first buffer in place
    // (allocates a scratch buffer for the conv output).
    void apply_cnn_block(int block_idx, std::vector<float>& cf, int T) const;
    // Bidirectional LSTM: input (T, HIDDEN=512) row-major,
    //                     output (T, 2*LSTM_HIDDEN=512) row-major.
    void apply_bilstm(const std::vector<float>& in, int T,
                      std::vector<float>& out) const;
};

}}  // namespace cactus::kokoro
