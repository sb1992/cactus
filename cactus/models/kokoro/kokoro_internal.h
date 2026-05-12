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

// ---- Conv1d weight bundle (folded weight_norm) ---------------------------
// Stored as (C_out, C_in, K) row-major, matching PyTorch nn.Conv1d.weight.
struct Conv1d {
    std::vector<float> w;
    std::vector<float> b;       // empty for no-bias convs
    int C_in = 0, C_out = 0, K = 0;
};

// ---- AdaLayerNorm / AdaIN1d fc projection: Linear(STYLE_DIM=128 -> 2*C) --
struct AdaLNFc {
    std::vector<float> w;       // (2C, 128)
    std::vector<float> b;       // (2C,)
};

// ---- AdainResBlk1d (LeakyReLU variant, used by predictor F0/N branches and
//      by decoder encode/decode blocks). Optional `pool` for upsample, optional
//      `conv1x1` shortcut when dim_in != dim_out.
struct AdainResBlk1d {
    int dim_in  = 0;
    int dim_out = 0;
    bool upsample = false;      // 'nearest'-equivalent upsample if true
    bool learned_sc = false;    // true iff dim_in != dim_out

    AdaLNFc norm1_fc;           // for AdaIN1d on dim_in
    AdaLNFc norm2_fc;           // for AdaIN1d on dim_out
    Conv1d  conv1;              // (dim_out, dim_in, K=3, p=1)
    Conv1d  conv2;              // (dim_out, dim_out, K=3, p=1)
    Conv1d  pool;               // ConvTranspose1d depthwise (dim_in,1,3) s=2 p=1 op=1
    Conv1d  conv1x1;            // (dim_out, dim_in, 1) no bias
};

// AdaIN1d: in/out is (1, C, T) channel-first, in place.
//   InstanceNorm1d (per-channel across T) with no learned affine, then
//   (1 + gamma(s)) * x + beta(s) where [gamma, beta] = fc(s).
// `style128` is (128,). `eps` is the InstanceNorm epsilon.
void ada_in_1d(float* x, int C, int T, const float* style128,
               const AdaLNFc& fc, float eps);

// AdaLayerNorm: in/out is (1, C, T) channel-first, in place.
//   F.layer_norm over the channel axis (no affine), then
//   (1 + gamma(s)) * x + beta(s).
void ada_layer_norm(float* x, int C, int T, const float* style128,
                    const AdaLNFc& fc, float eps);

// Depthwise ConvTranspose1d "pool" used by AdainResBlk1d when upsampling.
//   stride=2, kernel=3, groups=C, padding=1, output_padding=1.
//   Weight shape (C, 1, 3); bias shape (C,).
//   Input  (C, T)  channel-first.
//   Output (C, 2T) channel-first.
void conv_transpose_1d_depthwise_pool(const Conv1d& pool,
                                       const std::vector<float>& in_cf,
                                       int C, int T,
                                       std::vector<float>& out_cf);

// Nearest-neighbor upsample by factor 2 over the time axis.
//   Input (C, T) -> output (C, 2T) channel-first.
void upsample_nearest_2x(const std::vector<float>& in_cf, int C, int T,
                         std::vector<float>& out_cf);

// One AdainResBlk1d forward pass. Input/output are channel-first.
//   in_cf shape  : (dim_in,  T)
//   out_cf shape : (dim_out, T_out) where T_out = T (no upsample) or 2T.
//   leaky_slope  : 0.2 in Kokoro
//   inst_eps     : InstanceNorm epsilon (1e-5 in Kokoro)
void run_adain_res_blk(const AdainResBlk1d& blk, const float* style128,
                       const std::vector<float>& in_cf, int T,
                       std::vector<float>& out_cf, int& T_out,
                       float leaky_slope, float inst_eps);

}}}  // namespace cactus::kokoro::internal
