#pragma once

// Shared scalar helpers for the Kokoro C++ port.
//
// These primitives are duplicated nowhere else — both text_encoder.cpp and
// predictor.cpp pull them from here. They are intentionally tiny, allocate
// minimal scratch buffers, and are not exported beyond cactus/models/kokoro/.

#include <cstdint>
#include <vector>

namespace cactus { namespace kokoro { namespace internal {

// LeakyReLU(slope) in place over an entire flat buffer.
void leaky_relu_inplace(float* x, size_t n, float slope);

// LayerNorm over the channel axis of a (C, T) channel-first buffer in place.
//   For each time-step t:
//     mean_t = (1/C) sum_c x[c, t]
//     var_t  = (1/C) sum_c (x[c, t] - mean_t)^2
//     x[c, t] = (x[c, t] - mean_t) / sqrt(var_t + eps) * gamma[c] + beta[c]
//
// `gamma` / `beta` may be nullptr to skip the affine (i.e. plain LayerNorm
// with affine=False — used by AdaLayerNorm).
void channel_layer_norm(float* x, int C, int T, const float* gamma,
                        const float* beta, float eps);

// InstanceNorm1d over the (C, T) channel-first buffer in place. For each
// (B=1, C) slice, normalize across T:
//     mean_c = (1/T) sum_t x[c, t]
//     var_c  = (1/T) sum_t (x[c, t] - mean_c)^2
//     x[c, t] = (x[c, t] - mean_c) / sqrt(var_c + eps) * gamma[c] + beta[c]
//
// `gamma`/`beta` may be nullptr (AdaIN1d trained with affine=False even
// though the model has `affine=True` set; the v0_19 .pth is missing the
// `.norm.weight/.norm.bias` tensors so PyTorch defaults to weight=1, bias=0,
// i.e. an identity affine). See kokoro/istftnet.py:23 for the source comment.
void instance_norm_1d(float* x, int C, int T, const float* gamma,
                      const float* beta, float eps);

// Conv1d with kernel=5, padding=2, stride=1. `cf` is (C_in, T) channel-first;
// output is (C_out, T) into `cf_out`. Weight shape: (C_out, C_in, K=5).
void conv1d_pad2_k5(const float* cf, int C_in, int T,
                    const float* weight, const float* bias,
                    int C_out, std::vector<float>& cf_out);

// Generic Conv1d, padding=padding, stride=1. cf is (C_in, T); output is
// (C_out, T) into cf_out. Weight shape: (C_out, C_in, K).
void conv1d_padded(const float* cf, int C_in, int T,
                   const float* weight, const float* bias /* may be nullptr */,
                   int C_out, int K, int padding,
                   std::vector<float>& cf_out);

// Bidirectional single-layer LSTM. Input is (T, I) row-major; output is
// (T, 2*H) row-major with the forward direction in columns [0, H) and the
// reverse direction in [H, 2H) — matching nn.LSTM(bidirectional=True).
//
// All four bias and weight buffers are PyTorch-ordered: weight shape (4H, I)
// or (4H, H), gate ordering [i, f, g, o]. The two `_bwd_*` buffers describe
// the reverse direction.
void bidirectional_lstm(const float* in, int T, int I, int H,
                        const float* w_ih_fwd, const float* w_hh_fwd,
                        const float* b_ih_fwd, const float* b_hh_fwd,
                        const float* w_ih_bwd, const float* w_hh_bwd,
                        const float* b_ih_bwd, const float* b_hh_bwd,
                        std::vector<float>& out);

}}}  // namespace cactus::kokoro::internal
