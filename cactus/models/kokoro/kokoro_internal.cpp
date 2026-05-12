#include "kokoro_internal.h"

#include "../../kernel/kernel_lstm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cactus { namespace kokoro { namespace internal {

void leaky_relu_inplace(float* x, size_t n, float slope) {
    for (size_t i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] *= slope;
    }
}

void channel_layer_norm(float* x, int C, int T, const float* gamma,
                        const float* beta, float eps) {
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int c = 0; c < C; ++c) mean += x[static_cast<size_t>(c) * T + t];
        mean /= C;
        double var = 0.0;
        for (int c = 0; c < C; ++c) {
            const double d = x[static_cast<size_t>(c) * T + t] - mean;
            var += d * d;
        }
        var /= C;
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int c = 0; c < C; ++c) {
            const float xn = (x[static_cast<size_t>(c) * T + t] -
                              static_cast<float>(mean)) * inv_std;
            const float g = gamma ? gamma[c] : 1.0f;
            const float b = beta  ? beta[c]  : 0.0f;
            x[static_cast<size_t>(c) * T + t] = xn * g + b;
        }
    }
}

void instance_norm_1d(float* x, int C, int T, const float* gamma,
                      const float* beta, float eps) {
    if (T <= 0) return;
    for (int c = 0; c < C; ++c) {
        float* row = &x[static_cast<size_t>(c) * T];
        double mean = 0.0;
        for (int t = 0; t < T; ++t) mean += row[t];
        mean /= T;
        double var = 0.0;
        for (int t = 0; t < T; ++t) {
            const double d = row[t] - mean;
            var += d * d;
        }
        var /= T;
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        const float g = gamma ? gamma[c] : 1.0f;
        const float b = beta  ? beta[c]  : 0.0f;
        for (int t = 0; t < T; ++t) {
            const float xn = (row[t] - static_cast<float>(mean)) * inv_std;
            row[t] = xn * g + b;
        }
    }
}

void conv1d_pad2_k5(const float* cf, int C_in, int T,
                    const float* weight, const float* bias,
                    int C_out, std::vector<float>& cf_out) {
    conv1d_padded(cf, C_in, T, weight, bias, C_out, /*K=*/5, /*padding=*/2, cf_out);
}

void conv1d_padded(const float* cf, int C_in, int T,
                   const float* weight, const float* bias,
                   int C_out, int K, int padding,
                   std::vector<float>& cf_out) {
    cf_out.assign(static_cast<size_t>(C_out) * T, 0.0f);
    for (int oc = 0; oc < C_out; ++oc) {
        const float bv = bias ? bias[oc] : 0.0f;
        const float* wrow = &weight[static_cast<size_t>(oc) * C_in * K];
        float* outrow = &cf_out[static_cast<size_t>(oc) * T];
        for (int t = 0; t < T; ++t) outrow[t] = bv;
        for (int k = 0; k < K; ++k) {
            const int t_offset = k - padding;  // adds to t to get input index
            for (int ic = 0; ic < C_in; ++ic) {
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
}

void bidirectional_lstm(const float* in, int T, int I, int H,
                        const float* w_ih_fwd, const float* w_hh_fwd,
                        const float* b_ih_fwd, const float* b_hh_fwd,
                        const float* w_ih_bwd, const float* w_hh_bwd,
                        const float* b_ih_bwd, const float* b_hh_bwd,
                        std::vector<float>& out) {
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
            w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd,
            h_new.data(), c_new.data(),
            I, H);
        std::memcpy(&out[static_cast<size_t>(t) * 2 * H],
                    h_new.data(), H * sizeof(float));
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
            w_ih_bwd, w_hh_bwd, b_ih_bwd, b_hh_bwd,
            h_new.data(), c_new.data(),
            I, H);
        std::memcpy(&out[static_cast<size_t>(t) * 2 * H + H],
                    h_new.data(), H * sizeof(float));
        h.swap(h_new);
        c.swap(c_new);
    }
}

// ---------------------------------------------------------------------
// AdaLayerNorm: x is (1, C, T) channel-first. Reference flow: LayerNorm
// over channel axis (no affine), then (1 + gamma) * x + beta where
// [gamma, beta] = fc(s).
// ---------------------------------------------------------------------
void ada_layer_norm(float* x, int C, int T, const float* style128,
                     const AdaLNFc& fc, float eps) {
    constexpr int STYLE_DIM = 128;
    std::vector<float> h(2 * C, 0.0f);
    for (int i = 0; i < 2 * C; ++i) {
        float acc = fc.b[i];
        const float* row = &fc.w[static_cast<size_t>(i) * STYLE_DIM];
        for (int k = 0; k < STYLE_DIM; ++k) acc += row[k] * style128[k];
        h[i] = acc;
    }
    const float* gamma = h.data();
    const float* beta  = h.data() + C;
    channel_layer_norm(x, C, T, /*gamma*/nullptr, /*beta*/nullptr, eps);
    for (int c = 0; c < C; ++c) {
        const float g = 1.0f + gamma[c];
        const float b = beta[c];
        float* row = &x[static_cast<size_t>(c) * T];
        for (int t = 0; t < T; ++t) row[t] = g * row[t] + b;
    }
}

// ---------------------------------------------------------------------
// AdaIN1d: x is (1, C, T) channel-first. InstanceNorm1d (per-channel
// across T) with no affine, then (1 + gamma(s)) * x + beta(s).
// ---------------------------------------------------------------------
void ada_in_1d(float* x, int C, int T, const float* style128,
                const AdaLNFc& fc, float eps) {
    constexpr int STYLE_DIM = 128;
    std::vector<float> h(2 * C, 0.0f);
    for (int i = 0; i < 2 * C; ++i) {
        float acc = fc.b[i];
        const float* row = &fc.w[static_cast<size_t>(i) * STYLE_DIM];
        for (int k = 0; k < STYLE_DIM; ++k) acc += row[k] * style128[k];
        h[i] = acc;
    }
    const float* gamma = h.data();
    const float* beta  = h.data() + C;
    instance_norm_1d(x, C, T, /*gamma*/nullptr, /*beta*/nullptr, eps);
    for (int c = 0; c < C; ++c) {
        const float g = 1.0f + gamma[c];
        const float b = beta[c];
        float* row = &x[static_cast<size_t>(c) * T];
        for (int t = 0; t < T; ++t) row[t] = g * row[t] + b;
    }
}

// ---------------------------------------------------------------------
// Depthwise ConvTranspose1d (groups=C, in_per_group=1, out_per_group=1,
// stride=2, kernel=3, padding=1, output_padding=1). Weight (C, 1, 3),
// bias (C,). Output length T_out = (T-1)*2 - 2 + 3 + 1 = 2T.
// ---------------------------------------------------------------------
void conv_transpose_1d_depthwise_pool(const Conv1d& pool,
                                       const std::vector<float>& in_cf,
                                       int C, int T,
                                       std::vector<float>& out_cf) {
    constexpr int K = 3;
    constexpr int stride = 2;
    constexpr int padding = 1;
    constexpr int output_padding = 1;
    const int T_out = (T - 1) * stride - 2 * padding + K + output_padding;
    out_cf.assign(static_cast<size_t>(C) * T_out, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float* w = &pool.w[static_cast<size_t>(c) * K];
        const float bias = pool.b.empty() ? 0.0f : pool.b[c];
        const float* in_row = &in_cf[static_cast<size_t>(c) * T];
        float* out_row = &out_cf[static_cast<size_t>(c) * T_out];
        for (int t_out = 0; t_out < T_out; ++t_out) out_row[t_out] = bias;
        for (int i_in = 0; i_in < T; ++i_in) {
            const float xv = in_row[i_in];
            for (int k = 0; k < K; ++k) {
                const int t_out = i_in * stride + k - padding;
                if (t_out < 0 || t_out >= T_out) continue;
                out_row[t_out] += xv * w[k];
            }
        }
    }
}

void upsample_nearest_2x(const std::vector<float>& in_cf, int C, int T,
                          std::vector<float>& out_cf) {
    out_cf.assign(static_cast<size_t>(C) * 2 * T, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float* in_row = &in_cf[static_cast<size_t>(c) * T];
        float* out_row = &out_cf[static_cast<size_t>(c) * 2 * T];
        for (int t = 0; t < T; ++t) {
            out_row[2*t]     = in_row[t];
            out_row[2*t + 1] = in_row[t];
        }
    }
}

// ---------------------------------------------------------------------
// AdainResBlk1d:
//   residual: AdaIN1d -> LeakyReLU(slope) -> [pool if upsample] -> conv1
//             -> AdaIN1d -> LeakyReLU(slope) -> conv2
//   shortcut: [upsample_nearest_2x if upsample] -> [conv1x1 if learned_sc]
//   out = (residual + shortcut) * rsqrt(2)
// ---------------------------------------------------------------------
void run_adain_res_blk(const AdainResBlk1d& blk, const float* style128,
                        const std::vector<float>& in_cf, int T,
                        std::vector<float>& out_cf, int& T_out,
                        float leaky_slope, float inst_eps) {
    const int Ci = blk.dim_in;
    const int Co = blk.dim_out;

    // ---- residual path ----
    std::vector<float> r = in_cf;  // (Ci, T)
    ada_in_1d(r.data(), Ci, T, style128, blk.norm1_fc, inst_eps);
    leaky_relu_inplace(r.data(), r.size(), leaky_slope);

    int T_after_pool = T;
    std::vector<float> r_pool;
    if (blk.upsample) {
        conv_transpose_1d_depthwise_pool(blk.pool, r, Ci, T, r_pool);
        T_after_pool = 2 * T;
        r.swap(r_pool);
    }

    std::vector<float> r_c1;
    conv1d_padded(r.data(), Ci, T_after_pool,
                  blk.conv1.w.data(), blk.conv1.b.data(),
                  Co, 3, 1, r_c1);

    ada_in_1d(r_c1.data(), Co, T_after_pool, style128, blk.norm2_fc, inst_eps);
    leaky_relu_inplace(r_c1.data(), r_c1.size(), leaky_slope);

    std::vector<float> r_c2;
    conv1d_padded(r_c1.data(), Co, T_after_pool,
                  blk.conv2.w.data(), blk.conv2.b.data(),
                  Co, 3, 1, r_c2);

    // ---- shortcut path ----
    std::vector<float> sc;
    int T_sc = T;
    if (blk.upsample) {
        upsample_nearest_2x(in_cf, Ci, T, sc);
        T_sc = 2 * T;
    } else {
        sc = in_cf;
    }
    if (blk.learned_sc) {
        std::vector<float> sc_out;
        conv1d_padded(sc.data(), Ci, T_sc,
                      blk.conv1x1.w.data(), nullptr,
                      Co, 1, 0, sc_out);
        sc.swap(sc_out);
    }

    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    out_cf.assign(static_cast<size_t>(Co) * T_after_pool, 0.0f);
    for (size_t i = 0; i < out_cf.size(); ++i) {
        out_cf[i] = (r_c2[i] + sc[i]) * inv_sqrt2;
    }
    T_out = T_after_pool;
}

}}}  // namespace cactus::kokoro::internal
