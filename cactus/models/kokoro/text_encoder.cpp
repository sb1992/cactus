#include "text_encoder.h"

#include "../../kernel/kernel_lstm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// ----------------------------------------------------------------------
// .weights file loader (port of python/src/converter_kokoro.py
// load_tensor_with_header). Handles INT8 1D + INT8 2D-interleaved + FP16.
//
// We keep this private to text_encoder.cpp rather than add to the cactus
// graph runtime — the production loader (cactus/graph/graph_io.cpp
// MappedFile) is graph-coupled and not exported as a standalone API.
// ----------------------------------------------------------------------

namespace {

constexpr uint32_t CACT_MAGIC      = 0x54434143;  // 'CACT' little-endian
constexpr uint32_t FLAG_INTERLEAVED = 1u << 3;
constexpr int      INTERLEAVE_BLOCK = 4;
constexpr int      HEADER_SIZE      = 84;
constexpr int      ALIGNMENT        = 32;

inline int align_offset(int off, int align) {
    int rem = off % align;
    return rem ? off + (align - rem) : off;
}

// IEEE 754 fp16 -> fp32. Standard bit-twiddling, no platform deps.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3FFu;
            exp += 1;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct WeightsFile {
    std::vector<float> data;
    std::vector<int> shape;       // logical shape (NOT padded)
    bool is_int8 = false;
};

// Read a binary file in full; throws on error.
static std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) throw std::runtime_error("short read for " + path);
    return buf;
}

template <typename T>
inline T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// Inverse of tensor_io.interleave_weights: input flattened layout is
//   reshape(N_padded/B, K/4, B, 4) (C-order, B=4)
// We deinterleave to (N_padded, K) row-major int8.
static void deinterleave_weights(const int8_t* in,
                                 int N_padded, int K,
                                 std::vector<int8_t>& out) {
    constexpr int B = INTERLEAVE_BLOCK;
    out.assign(static_cast<size_t>(N_padded) * K, 0);
    const int blocks_n = N_padded / B;
    const int blocks_k = K / 4;
    for (int bn = 0; bn < blocks_n; ++bn) {
        for (int bk = 0; bk < blocks_k; ++bk) {
            for (int r = 0; r < B; ++r) {
                for (int c = 0; c < 4; ++c) {
                    const int src_idx = ((bn * blocks_k + bk) * B + r) * 4 + c;
                    const int dst_row = bn * B + r;
                    const int dst_col = bk * 4 + c;
                    out[static_cast<size_t>(dst_row) * K + dst_col] = in[src_idx];
                }
            }
        }
    }
}

// Inverse of tensor_io.interleave_scales: layout
//   reshape(N_padded/B, num_groups, B) -> (N_padded, num_groups)
static void deinterleave_scales(const float* in,
                                int N_padded, int num_groups,
                                std::vector<float>& out) {
    constexpr int B = INTERLEAVE_BLOCK;
    out.assign(static_cast<size_t>(N_padded) * num_groups, 0.0f);
    const int blocks_n = N_padded / B;
    for (int bn = 0; bn < blocks_n; ++bn) {
        for (int g = 0; g < num_groups; ++g) {
            for (int r = 0; r < B; ++r) {
                const int src = ((bn * num_groups) + g) * B + r;
                const int dst_row = bn * B + r;
                out[static_cast<size_t>(dst_row) * num_groups + g] = in[src];
            }
        }
    }
}

static WeightsFile load_weights_file(const std::string& path) {
    auto raw = read_all(path);
    if (raw.size() < static_cast<size_t>(HEADER_SIZE)) {
        throw std::runtime_error("file too small: " + path);
    }
    const uint8_t* p = raw.data();
    const uint32_t magic = read_le<uint32_t>(p + 0);
    if (magic != CACT_MAGIC) throw std::runtime_error("bad magic in " + path);

    const uint32_t flags = read_le<uint32_t>(p + 4);
    // alignment at +8 (always 32 here, ignore)
    const uint32_t ndim = read_le<uint32_t>(p + 12);
    int dims[4];
    for (int i = 0; i < 4; ++i) {
        const uint64_t d = read_le<uint64_t>(p + 16 + 8 * i);
        dims[i] = static_cast<int>(d);
    }
    const uint32_t precision = read_le<uint32_t>(p + 48);
    const uint64_t data_bytes = read_le<uint64_t>(p + 52);
    const uint64_t scales_bytes = read_le<uint64_t>(p + 60);
    const uint32_t group_size = read_le<uint32_t>(p + 68);
    const uint32_t num_groups = read_le<uint32_t>(p + 72);
    const uint64_t original_N = read_le<uint64_t>(p + 76);

    const int header_aligned = align_offset(HEADER_SIZE, ALIGNMENT);

    WeightsFile out;

    if (precision == 1) {
        // FP16, no scales. Body starts at aligned header.
        const int body_off = header_aligned;
        const size_t n_elems = static_cast<size_t>(data_bytes) / 2;
        if (body_off + data_bytes > raw.size()) {
            throw std::runtime_error("FP16 body OOB in " + path);
        }
        out.data.resize(n_elems);
        const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(p + body_off);
        for (size_t i = 0; i < n_elems; ++i) out.data[i] = fp16_to_fp32(fp16[i]);
        for (uint32_t i = 0; i < ndim; ++i) out.shape.push_back(dims[i]);
        return out;
    }

    if (precision != 0) {
        throw std::runtime_error("unsupported precision in " + path);
    }

    // INT8 path. Layout:
    //   header (84) -> pad to 32 -> scales (FP16) -> pad to 32 -> data (INT8)
    out.is_int8 = true;
    const int scales_off = header_aligned;
    const size_t n_scales = static_cast<size_t>(scales_bytes) / 2;
    if (scales_off + scales_bytes > raw.size()) {
        throw std::runtime_error("INT8 scales OOB in " + path);
    }
    std::vector<float> scales_fp32(n_scales);
    {
        const uint16_t* sfp16 = reinterpret_cast<const uint16_t*>(p + scales_off);
        for (size_t i = 0; i < n_scales; ++i) scales_fp32[i] = fp16_to_fp32(sfp16[i]);
    }
    const int data_off = align_offset(scales_off + static_cast<int>(scales_bytes), ALIGNMENT);
    if (data_off + data_bytes > raw.size()) {
        throw std::runtime_error("INT8 data OOB in " + path);
    }
    const int8_t* qd = reinterpret_cast<const int8_t*>(p + data_off);

    if (ndim == 2) {
        const int N_padded = dims[0];
        const int K = dims[1];
        const int orig_N = static_cast<int>(original_N);

        std::vector<int8_t> q2d;
        std::vector<float> s2d;

        if (flags & FLAG_INTERLEAVED) {
            deinterleave_weights(qd, N_padded, K, q2d);
            deinterleave_scales(scales_fp32.data(), N_padded, static_cast<int>(num_groups), s2d);
        } else {
            // Plain row-major INT8 + per-row-group scales (flat (N_padded, num_groups))
            q2d.assign(qd, qd + static_cast<size_t>(N_padded) * K);
            s2d = scales_fp32;
        }

        // Dequant: out[r, g*GS + j] = q2d[r, g*GS + j] * s2d[r, g]
        out.data.assign(static_cast<size_t>(orig_N) * K, 0.0f);
        const int gs = static_cast<int>(group_size);
        for (int r = 0; r < orig_N; ++r) {
            for (int g = 0; g < static_cast<int>(num_groups); ++g) {
                const float scale = s2d[static_cast<size_t>(r) * num_groups + g];
                for (int j = 0; j < gs; ++j) {
                    const int col = g * gs + j;
                    if (col >= K) break;
                    const int8_t qv = q2d[static_cast<size_t>(r) * K + col];
                    out.data[static_cast<size_t>(r) * K + col] = qv * scale;
                }
            }
        }
        out.shape = {orig_N, K};
        return out;
    }

    if (ndim == 1) {
        const int K = dims[0];
        out.data.assign(static_cast<size_t>(K), 0.0f);
        const int gs = static_cast<int>(group_size);
        for (int g = 0; g < static_cast<int>(num_groups); ++g) {
            const float scale = scales_fp32[g];
            for (int j = 0; j < gs; ++j) {
                const int col = g * gs + j;
                if (col >= K) break;
                out.data[col] = qd[col] * scale;
            }
        }
        out.shape = {K};
        return out;
    }

    throw std::runtime_error("unsupported INT8 ndim in " + path);
}

}  // namespace

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
//      with bias, weight shape (out, in, K).
//   2. Custom LayerNorm over channel axis: per-time-step normalize across C
//      then affine gamma/beta.
//   3. LeakyReLU(0.2).
// ----------------------------------------------------------------------
void TextEncoder::apply_cnn_block(int b, std::vector<float>& cf, int T) const {
    const auto& blk = cnn_[b];
    const int C = HIDDEN;
    const int K = CNN_KERNEL;
    const int P = CNN_PADDING;

    // Conv1d: cf is (C_in, T). Compute (C_out, T) into a fresh buffer.
    // out[oc, t] = bias[oc] + sum_{ic=0..C-1} sum_{k=0..K-1}
    //                w[oc, ic, k] * cf[ic, t + k - P]   (pad=2, stride=1)
    std::vector<float> conv_out(static_cast<size_t>(C) * T, 0.0f);
    for (int oc = 0; oc < C; ++oc) {
        const float bias = blk.conv_bias[oc];
        const float* wrow = &blk.conv_weight[static_cast<size_t>(oc) * C * K];
        float* outrow = &conv_out[static_cast<size_t>(oc) * T];
        for (int t = 0; t < T; ++t) outrow[t] = bias;
        for (int k = 0; k < K; ++k) {
            const int t_offset = k - P;  // adds to t to get input index
            for (int ic = 0; ic < C; ++ic) {
                const float w = wrow[ic * K + k];
                const float* incol = &cf[static_cast<size_t>(ic) * T];
                for (int t = 0; t < T; ++t) {
                    const int it = t + t_offset;
                    if (it < 0 || it >= T) continue;
                    outrow[t] += w * incol[it];
                }
            }
        }
    }

    // LayerNorm over channel axis: for each time t, normalize across C.
    //   mean_t = (1/C) * sum_c x[c, t]
    //   var_t  = (1/C) * sum_c (x[c, t] - mean_t)^2
    //   out[c, t] = (x[c, t] - mean_t) / sqrt(var_t + eps) * gamma[c] + beta[c]
    // Then LeakyReLU(0.2) in place.
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int c = 0; c < C; ++c) mean += conv_out[static_cast<size_t>(c) * T + t];
        mean /= C;
        double var = 0.0;
        for (int c = 0; c < C; ++c) {
            const double d = conv_out[static_cast<size_t>(c) * T + t] - mean;
            var += d * d;
        }
        var /= C;
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + LAYERNORM_EPS);
        for (int c = 0; c < C; ++c) {
            const float xn = (conv_out[static_cast<size_t>(c) * T + t] - static_cast<float>(mean)) * inv_std;
            float v = xn * blk.norm_gamma[c] + blk.norm_beta[c];
            // LeakyReLU(0.2)
            if (v < 0.0f) v *= LEAKY_SLOPE;
            conv_out[static_cast<size_t>(c) * T + t] = v;
        }
    }

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
    constexpr int H = LSTM_HIDDEN;
    constexpr int I = HIDDEN;
    out.assign(static_cast<size_t>(T) * 2 * H, 0.0f);

    std::vector<float> h(H, 0.0f), c(H, 0.0f);
    std::vector<float> h_new(H, 0.0f), c_new(H, 0.0f);

    // Forward
    std::fill(h.begin(), h.end(), 0.0f);
    std::fill(c.begin(), c.end(), 0.0f);
    for (int t = 0; t < T; ++t) {
        cactus_lstm_cell(
            &in[static_cast<size_t>(t) * I],
            h.data(), c.data(),
            lstm_fwd_.w_ih.data(), lstm_fwd_.w_hh.data(),
            lstm_fwd_.b_ih.data(), lstm_fwd_.b_hh.data(),
            h_new.data(), c_new.data(),
            I, H);
        std::memcpy(&out[static_cast<size_t>(t) * 2 * H], h_new.data(), H * sizeof(float));
        h.swap(h_new);
        c.swap(c_new);
    }

    // Backward
    std::fill(h.begin(), h.end(), 0.0f);
    std::fill(c.begin(), c.end(), 0.0f);
    for (int t = T - 1; t >= 0; --t) {
        cactus_lstm_cell(
            &in[static_cast<size_t>(t) * I],
            h.data(), c.data(),
            lstm_bwd_.w_ih.data(), lstm_bwd_.w_hh.data(),
            lstm_bwd_.b_ih.data(), lstm_bwd_.b_hh.data(),
            h_new.data(), c_new.data(),
            I, H);
        std::memcpy(&out[static_cast<size_t>(t) * 2 * H + H], h_new.data(), H * sizeof(float));
        h.swap(h_new);
        c.swap(c_new);
    }
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
