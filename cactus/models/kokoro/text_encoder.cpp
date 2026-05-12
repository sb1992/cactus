#include "text_encoder.h"

#include "kokoro_internal.h"
#include "weights_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// ----------------------------------------------------------------------
// load_weights
// ----------------------------------------------------------------------
bool TextEncoder::load_weights(const std::string& dir) {
    auto load_named = [&](const std::string& base) -> WeightsFile {
        return load_weights_file(dir + "/text_encoder_" + base + ".weights");
    };
    try {
        // Embedding: INT8 2D, original shape (178, 512). Loader returns
        // (178, 512) row-major fp32 already cropped on N.
        embedding_ = load_named("embedding").data;
        if (static_cast<int>(embedding_.size()) != VOCAB * HIDDEN) {
            std::fprintf(stderr,
                "TextEncoder embedding: got %zu fp32 elements, expected %d\n",
                embedding_.size(), VOCAB * HIDDEN);
            return false;
        }

        for (int i = 0; i < 3; ++i) {
            const std::string s = std::to_string(i);
            cnn_[i].conv_weight = load_named("cnn_" + s + "_conv_weight").data;
            cnn_[i].conv_bias   = load_named("cnn_" + s + "_conv_bias").data;
            cnn_[i].norm_gamma  = load_named("cnn_" + s + "_norm_gamma").data;
            cnn_[i].norm_beta   = load_named("cnn_" + s + "_norm_beta").data;
            const int expected_w = HIDDEN * HIDDEN * CNN_KERNEL;
            if (static_cast<int>(cnn_[i].conv_weight.size()) != expected_w ||
                static_cast<int>(cnn_[i].conv_bias.size())   != HIDDEN ||
                static_cast<int>(cnn_[i].norm_gamma.size())  != HIDDEN ||
                static_cast<int>(cnn_[i].norm_beta.size())   != HIDDEN) {
                std::fprintf(stderr, "TextEncoder cnn[%d] size mismatch\n", i);
                return false;
            }
        }

        auto load_dir_lstm = [&](LstmDir& d, const char* dir_tag) {
            d.w_ih = load_named(std::string("lstm_") + dir_tag + "_0_weight_ih").data;
            d.w_hh = load_named(std::string("lstm_") + dir_tag + "_0_weight_hh").data;
            d.b_ih = load_named(std::string("lstm_") + dir_tag + "_0_bias_ih").data;
            d.b_hh = load_named(std::string("lstm_") + dir_tag + "_0_bias_hh").data;
            // Sanity: 4*256 rows
            const int expected_4H = 4 * LSTM_HIDDEN;
            if (static_cast<int>(d.w_ih.size()) != expected_4H * HIDDEN ||
                static_cast<int>(d.w_hh.size()) != expected_4H * LSTM_HIDDEN ||
                static_cast<int>(d.b_ih.size()) != expected_4H ||
                static_cast<int>(d.b_hh.size()) != expected_4H) {
                throw std::runtime_error(std::string("LSTM size mismatch ") + dir_tag);
            }
        };
        load_dir_lstm(lstm_fwd_, "fwd");
        load_dir_lstm(lstm_bwd_, "bwd");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TextEncoder load_weights failed: %s\n", e.what());
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------
// Embedding lookup: phonemes [T] (int64) -> hidden [T, HIDDEN] row-major.
// ----------------------------------------------------------------------
void TextEncoder::embed_lookup(const int64_t* phonemes, int T,
                               std::vector<float>& out) const {
    out.assign(static_cast<size_t>(T) * HIDDEN, 0.0f);
    for (int t = 0; t < T; ++t) {
        const int64_t id = phonemes[t];
        if (id < 0 || id >= VOCAB) continue;  // mask invalid -> zeros
        std::memcpy(&out[static_cast<size_t>(t) * HIDDEN],
                    &embedding_[static_cast<size_t>(id) * HIDDEN],
                    HIDDEN * sizeof(float));
    }
}

// ----------------------------------------------------------------------
// One CNN block, channel-first (HIDDEN, T) buffer.
//   1. Conv1d (HIDDEN_in=512 -> HIDDEN_out=512, K=5, pad=2, stride=1)
//   2. Custom LayerNorm over channel axis with affine gamma/beta
//   3. LeakyReLU(0.2) in place
// ----------------------------------------------------------------------
void TextEncoder::apply_cnn_block(int b, std::vector<float>& cf, int T) const {
    const auto& blk = cnn_[b];
    std::vector<float> conv_out;
    internal::conv1d_pad2_k5(cf.data(), HIDDEN, T,
                             blk.conv_weight.data(), blk.conv_bias.data(),
                             HIDDEN, conv_out);
    internal::channel_layer_norm(conv_out.data(), HIDDEN, T,
                                 blk.norm_gamma.data(), blk.norm_beta.data(),
                                 LAYERNORM_EPS);
    internal::leaky_relu_inplace(conv_out.data(), conv_out.size(), LEAKY_SLOPE);
    cf.swap(conv_out);
}

// ----------------------------------------------------------------------
// Bidirectional LSTM. `in` is (T, HIDDEN=512) row-major; `out` is
// (T, 2*LSTM_HIDDEN=512) row-major with the forward direction occupying
// columns [0, 256) and the reverse direction occupying [256, 512).
// (PyTorch nn.LSTM bidirectional concat order, verified by the upstream
// kokoro/StyleTTS2 architecture.md spec.)
// ----------------------------------------------------------------------
void TextEncoder::apply_bilstm(const std::vector<float>& in, int T,
                               std::vector<float>& out) const {
    internal::bidirectional_lstm(in.data(), T, HIDDEN, LSTM_HIDDEN,
        lstm_fwd_.w_ih.data(), lstm_fwd_.w_hh.data(),
        lstm_fwd_.b_ih.data(), lstm_fwd_.b_hh.data(),
        lstm_bwd_.w_ih.data(), lstm_bwd_.w_hh.data(),
        lstm_bwd_.b_ih.data(), lstm_bwd_.b_hh.data(),
        out);
}

// ----------------------------------------------------------------------
// Public driver
// ----------------------------------------------------------------------
void TextEncoder::run_until_layer(const int64_t* phonemes, int T, Layer last,
                                   std::vector<float>& out) const {
    // 1. Embedding (T, HIDDEN) row-major.
    embed_lookup(phonemes, T, out);
    if (last == Layer::Embedding) return;

    // 2. Transpose to (HIDDEN, T) channel-first for CNN blocks.
    std::vector<float> cf(static_cast<size_t>(HIDDEN) * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < HIDDEN; ++c) {
            cf[static_cast<size_t>(c) * T + t] = out[static_cast<size_t>(t) * HIDDEN + c];
        }
    }

    apply_cnn_block(0, cf, T);
    auto write_back_tx = [&](std::vector<float>& dst) {
        dst.assign(static_cast<size_t>(T) * HIDDEN, 0.0f);
        for (int c = 0; c < HIDDEN; ++c) {
            for (int t = 0; t < T; ++t) {
                dst[static_cast<size_t>(t) * HIDDEN + c] = cf[static_cast<size_t>(c) * T + t];
            }
        }
    };
    if (last == Layer::Cnn0) { write_back_tx(out); return; }

    apply_cnn_block(1, cf, T);
    if (last == Layer::Cnn1) { write_back_tx(out); return; }

    apply_cnn_block(2, cf, T);
    if (last == Layer::Cnn2) { write_back_tx(out); return; }

    // 3. Transpose back to (T, HIDDEN) for the LSTM.
    std::vector<float> seq(static_cast<size_t>(T) * HIDDEN, 0.0f);
    for (int c = 0; c < HIDDEN; ++c) {
        for (int t = 0; t < T; ++t) {
            seq[static_cast<size_t>(t) * HIDDEN + c] = cf[static_cast<size_t>(c) * T + t];
        }
    }

    // 4. Bidirectional LSTM -> (T, 2*LSTM_HIDDEN)
    apply_bilstm(seq, T, out);
}

void TextEncoder::forward(const int64_t* phonemes, int T, std::vector<float>& out) const {
    run_until_layer(phonemes, T, Layer::Lstm, out);
}

}}  // namespace cactus::kokoro
