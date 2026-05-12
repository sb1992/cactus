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

}}}  // namespace cactus::kokoro::internal
