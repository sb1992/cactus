#include "generator.h"

#include "kokoro_internal.h"
#include "weights_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

namespace {

// Hann window with `periodic=True` (matches torch.hann_window default).
//   w[k] = 0.5 * (1 - cos(2*pi*k/N))   for k = 0..N-1
std::vector<float> hann_periodic(int N) {
    std::vector<float> w(N);
    constexpr double PI = 3.14159265358979323846;
    for (int k = 0; k < N; ++k) {
        w[k] = 0.5f * (1.0f - static_cast<float>(std::cos(2.0 * PI * k / N)));
    }
    return w;
}

}  // namespace

// ============================================================================
// Snake1d
// ============================================================================
void Generator::snake1d_inplace(float* x, int C, int T, const float* alpha) {
    // x[c, t] = x[c, t] + (1/alpha[c]) * sin(alpha[c] * x[c, t])^2
    for (int c = 0; c < C; ++c) {
        const float a = alpha[c];
        const float inv_a = (a == 0.0f) ? 0.0f : (1.0f / a);
        float* row = &x[static_cast<size_t>(c) * T];
        for (int t = 0; t < T; ++t) {
            const float s = std::sin(a * row[t]);
            row[t] += inv_a * (s * s);
        }
    }
}

// ============================================================================
// Conv1d (strided + dilated)
//
// Computes:
//   out[oc, t] = bias[oc] + sum_ic sum_k w[oc, ic, k] * in[ic, t*stride + k*dil - pad]
// T_out = (T_in + 2*pad - dil*(K-1) - 1) / stride + 1
// ============================================================================
void Generator::conv1d_strided_dil(const float* cf, int C_in, int T_in,
                                   const float* weight, const float* bias,
                                   int C_out, int K,
                                   int stride, int padding, int dilation,
                                   std::vector<float>& cf_out, int& T_out) {
    T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (T_out < 0) T_out = 0;
    cf_out.assign(static_cast<size_t>(C_out) * T_out, 0.0f);
    for (int oc = 0; oc < C_out; ++oc) {
        const float bv = bias ? bias[oc] : 0.0f;
        const float* wrow = &weight[static_cast<size_t>(oc) * C_in * K];
        float* outrow = &cf_out[static_cast<size_t>(oc) * T_out];
        for (int t = 0; t < T_out; ++t) outrow[t] = bv;
        for (int t = 0; t < T_out; ++t) {
            const int t_in_base = t * stride - padding;
            for (int k = 0; k < K; ++k) {
                const int it = t_in_base + k * dilation;
                if (it < 0 || it >= T_in) continue;
                for (int ic = 0; ic < C_in; ++ic) {
                    outrow[t] += wrow[ic * K + k] *
                                 cf[static_cast<size_t>(ic) * T_in + it];
                }
            }
        }
    }
}

// ============================================================================
// ConvTranspose1d (groups=1).
//
// PyTorch nn.ConvTranspose1d weight layout: (in_channels, out_channels, K).
// Forward formula:
//   out[oc, t_out] = bias[oc] + sum_ic sum_k w[ic, oc, k] * in[ic, i_in]
// where t_out = i_in*stride + k - padding.
// T_out = (T_in - 1)*stride - 2*padding + K  (with output_padding=0).
// ============================================================================
void Generator::conv_transpose_1d(const Conv1d& w, const std::vector<float>& in_cf,
                                  int C_in, int C_out, int T_in,
                                  int K, int stride, int padding,
                                  std::vector<float>& out_cf) {
    const int T_out = (T_in - 1) * stride - 2 * padding + K;
    out_cf.assign(static_cast<size_t>(C_out) * T_out, 0.0f);
    if (!w.b.empty()) {
        for (int oc = 0; oc < C_out; ++oc) {
            const float bv = w.b[oc];
            float* row = &out_cf[static_cast<size_t>(oc) * T_out];
            for (int t = 0; t < T_out; ++t) row[t] = bv;
        }
    }
    // weight shape: (C_in, C_out, K)
    for (int ic = 0; ic < C_in; ++ic) {
        const float* in_row = &in_cf[static_cast<size_t>(ic) * T_in];
        const float* w_ic   = &w.w[static_cast<size_t>(ic) * C_out * K];
        for (int i_in = 0; i_in < T_in; ++i_in) {
            const float xv = in_row[i_in];
            if (xv == 0.0f) continue;
            const int t_out_base = i_in * stride - padding;
            for (int k = 0; k < K; ++k) {
                const int t_out = t_out_base + k;
                if (t_out < 0 || t_out >= T_out) continue;
                for (int oc = 0; oc < C_out; ++oc) {
                    out_cf[static_cast<size_t>(oc) * T_out + t_out] +=
                        xv * w_ic[oc * K + k];
                }
            }
        }
    }
}

// ============================================================================
// AdaINResBlock1 forward
// ============================================================================
void Generator::run_adain_resblock1(const AdaINResBlock1& blk,
                                    const float* style128,
                                    std::vector<float>& x_cf, int T) const {
    const int C = blk.channels;
    for (int j = 0; j < 3; ++j) {
        // Snake1d sub-block
        std::vector<float> xt = x_cf;  // (C, T)
        // adain1[j]
        internal::ada_in_1d(xt.data(), C, T, style128, blk.adain1_fc[j], INSTNORM_EPS);
        snake1d_inplace(xt.data(), C, T, blk.alpha1[j].data());
        // convs1[j]: dilated conv, padding = (K * dil - dil) / 2
        const int dil1   = blk.dilation[j];
        const int pad1   = (blk.K * dil1 - dil1) / 2;
        std::vector<float> y;
        int Ty = 0;
        conv1d_strided_dil(xt.data(), C, T,
                           blk.convs1[j].w.data(), blk.convs1[j].b.data(),
                           C, blk.K, /*stride=*/1, /*padding=*/pad1, /*dilation=*/dil1,
                           y, Ty);
        // adain2[j]
        internal::ada_in_1d(y.data(), C, Ty, style128, blk.adain2_fc[j], INSTNORM_EPS);
        snake1d_inplace(y.data(), C, Ty, blk.alpha2[j].data());
        // convs2[j]: dilation 1, padding (K-1)/2
        const int pad2 = (blk.K - 1) / 2;
        std::vector<float> z;
        int Tz = 0;
        conv1d_strided_dil(y.data(), C, Ty,
                           blk.convs2[j].w.data(), blk.convs2[j].b.data(),
                           C, blk.K, /*stride=*/1, /*padding=*/pad2, /*dilation=*/1,
                           z, Tz);
        // x = z + x  (residual)
        for (size_t i = 0; i < x_cf.size(); ++i) x_cf[i] += z[i];
    }
}

// ============================================================================
// load_weights
// ============================================================================
bool Generator::load_resblock(AdaINResBlock1& blk, const std::string& dir,
                              const std::string& base, int channels, int K,
                              const int dil[3]) {
    blk.channels = channels;
    blk.K = K;
    blk.dilation[0] = dil[0]; blk.dilation[1] = dil[1]; blk.dilation[2] = dil[2];

    auto load_named = [&](const std::string& fname) -> std::vector<float> {
        return load_weights_file(dir + "/decoder_" + fname + ".weights").data;
    };

    for (int j = 0; j < 3; ++j) {
        const std::string j_str = std::to_string(j);
        // adain1_<j>_fc
        blk.adain1_fc[j].w = load_named(base + "_adain1_" + j_str + "_fc_weight");
        blk.adain1_fc[j].b = load_named(base + "_adain1_" + j_str + "_fc_bias");
        if ((int)blk.adain1_fc[j].w.size() != 2 * channels * STYLE_DIM ||
            (int)blk.adain1_fc[j].b.size() != 2 * channels) {
            std::fprintf(stderr, "%s_adain1_%s_fc size mismatch (got %zu, %zu, exp %d, %d)\n",
                         base.c_str(), j_str.c_str(),
                         blk.adain1_fc[j].w.size(), blk.adain1_fc[j].b.size(),
                         2*channels*STYLE_DIM, 2*channels);
            return false;
        }
        blk.adain2_fc[j].w = load_named(base + "_adain2_" + j_str + "_fc_weight");
        blk.adain2_fc[j].b = load_named(base + "_adain2_" + j_str + "_fc_bias");

        // convs1_<j>: (channels, channels, K)
        blk.convs1[j].w = load_named(base + "_convs1_" + j_str + "_weight");
        blk.convs1[j].b = load_named(base + "_convs1_" + j_str + "_bias");
        if ((int)blk.convs1[j].w.size() != channels * channels * K) {
            std::fprintf(stderr, "%s_convs1_%s_weight size mismatch (got %zu, exp %d)\n",
                         base.c_str(), j_str.c_str(), blk.convs1[j].w.size(),
                         channels*channels*K);
            return false;
        }
        blk.convs1[j].C_in = channels; blk.convs1[j].C_out = channels; blk.convs1[j].K = K;

        blk.convs2[j].w = load_named(base + "_convs2_" + j_str + "_weight");
        blk.convs2[j].b = load_named(base + "_convs2_" + j_str + "_bias");
        blk.convs2[j].C_in = channels; blk.convs2[j].C_out = channels; blk.convs2[j].K = K;

        // alpha1_<j>, alpha2_<j>: (channels,)
        blk.alpha1[j] = load_named(base + "_alpha1_" + j_str);
        blk.alpha2[j] = load_named(base + "_alpha2_" + j_str);
        if ((int)blk.alpha1[j].size() != channels || (int)blk.alpha2[j].size() != channels) {
            std::fprintf(stderr, "%s_alpha%s size mismatch (got %zu/%zu, exp %d)\n",
                         base.c_str(), j_str.c_str(),
                         blk.alpha1[j].size(), blk.alpha2[j].size(), channels);
            return false;
        }
    }
    return true;
}

bool Generator::load_weights(const std::string& dir) {
    auto load_named = [&](const std::string& fname) -> std::vector<float> {
        return load_weights_file(dir + "/decoder_" + fname + ".weights").data;
    };

    auto load_conv = [&](Conv1d& c, const std::string& base,
                         int Cout, int Cin, int K, bool with_bias) {
        c.w = load_named(base + "_weight");
        if (with_bias) {
            c.b = load_named(base + "_bias");
            if ((int)c.b.size() != Cout) {
                throw std::runtime_error("conv bias size mismatch in " + base
                                         + " got " + std::to_string(c.b.size())
                                         + " exp " + std::to_string(Cout));
            }
        } else {
            c.b.clear();
        }
        if ((int)c.w.size() != Cout * Cin * K) {
            throw std::runtime_error("conv weight size mismatch in " + base
                                     + " got " + std::to_string(c.w.size())
                                     + " exp " + std::to_string(Cout*Cin*K));
        }
        c.C_in = Cin; c.C_out = Cout; c.K = K;
    };

    try {
        // m_source.l_linear: weight (1, 9), bias (1,)
        // INT8 2D padding: K=9 padded to 32, so loader returns 32 floats with
        // only the first 9 valid (rest zero). Bias is FP16 1D, size 1.
        auto raw_w = load_named("generator_m_source_l_linear_weight");
        auto bias_v = load_named("generator_m_source_l_linear_bias");
        if (raw_w.size() < HARMONICS || bias_v.size() != 1u) {
            std::fprintf(stderr, "m_source.l_linear size mismatch (w=%zu, b=%zu, expected w>=%d, b=1)\n",
                         raw_w.size(), bias_v.size(), HARMONICS);
            return false;
        }
        m_source_linear_w_.assign(raw_w.begin(), raw_w.begin() + HARMONICS);
        m_source_linear_b_ = bias_v[0];

        // noise_convs[0]: Conv1d (256, 22, 12) k=12, s=6, p=3
        load_conv(noise_convs_[0], "generator_noise_convs_0",
                  UPS0_OUT_CH, HAR_FEATS, 12, true);
        // noise_convs[1]: Conv1d (128, 22, 1) k=1
        load_conv(noise_convs_[1], "generator_noise_convs_1",
                  UPS1_OUT_CH, HAR_FEATS, 1, true);

        // noise_res[0]: ch=256 K=7 dil=(1,3,5)
        const int dil[3] = {1, 3, 5};
        if (!load_resblock(noise_res_[0], dir, "generator_noise_res_0", UPS0_OUT_CH, 7, dil))
            return false;
        // noise_res[1]: ch=128 K=11
        if (!load_resblock(noise_res_[1], dir, "generator_noise_res_1", UPS1_OUT_CH, 11, dil))
            return false;

        // ups[0]: ConvTranspose1d. PyTorch weight layout:
        //   shape (in_channels, out_channels, K) = (512, 256, 20).
        // Bias has shape (out_channels,) = (256,).
        // We use load_conv() helper but it validates bias size against the
        // first dim arg, so we load weight + bias separately to bypass that.
        ups_[0].w = load_named("generator_ups_0_weight");
        ups_[0].b = load_named("generator_ups_0_bias");
        ups_[0].C_in = IN_CH; ups_[0].C_out = UPS0_OUT_CH; ups_[0].K = 20;
        if ((int)ups_[0].w.size() != IN_CH * UPS0_OUT_CH * 20 ||
            (int)ups_[0].b.size() != UPS0_OUT_CH) {
            std::fprintf(stderr, "ups[0] size mismatch (w=%zu exp %d, b=%zu exp %d)\n",
                         ups_[0].w.size(), IN_CH*UPS0_OUT_CH*20,
                         ups_[0].b.size(), UPS0_OUT_CH);
            return false;
        }

        // ups[1]: weight shape (256, 128, 12), bias (128,)
        ups_[1].w = load_named("generator_ups_1_weight");
        ups_[1].b = load_named("generator_ups_1_bias");
        ups_[1].C_in = UPS0_OUT_CH; ups_[1].C_out = UPS1_OUT_CH; ups_[1].K = 12;
        if ((int)ups_[1].w.size() != UPS0_OUT_CH * UPS1_OUT_CH * 12 ||
            (int)ups_[1].b.size() != UPS1_OUT_CH) {
            std::fprintf(stderr, "ups[1] size mismatch (w=%zu exp %d, b=%zu exp %d)\n",
                         ups_[1].w.size(), UPS0_OUT_CH*UPS1_OUT_CH*12,
                         ups_[1].b.size(), UPS1_OUT_CH);
            return false;
        }

        // resblocks[0..2]: ch=256, K in {3,7,11}, dil=(1,3,5)
        const int Ks[3] = {3, 7, 11};
        for (int j = 0; j < 3; ++j) {
            std::string base = "generator_resblocks_" + std::to_string(j);
            if (!load_resblock(resblocks_[j], dir, base, UPS0_OUT_CH, Ks[j], dil))
                return false;
        }
        // resblocks[3..5]: ch=128, K in {3,7,11}
        for (int j = 0; j < 3; ++j) {
            std::string base = "generator_resblocks_" + std::to_string(3 + j);
            if (!load_resblock(resblocks_[3 + j], dir, base, UPS1_OUT_CH, Ks[j], dil))
                return false;
        }

        // conv_post: Conv1d (22, 128, 7) k=7 p=3
        load_conv(conv_post_, "generator_conv_post",
                  HAR_FEATS, UPS1_OUT_CH, 7, true);

        // ---- Precompute STFT/iSTFT kernels (CustomSTFT-equivalent) ---------
        stft_window_ = hann_periodic(N_FFT);
        // We use win_length == N_FFT, no zero-padding.
        // Forward DFT: angle = 2*pi*k*n / N_FFT, real = cos*window, imag = -sin*window
        // Shape: (FREQ_BINS, N_FFT) row-major.
        stft_w_real_.assign(static_cast<size_t>(FREQ_BINS) * N_FFT, 0.0f);
        stft_w_imag_.assign(static_cast<size_t>(FREQ_BINS) * N_FFT, 0.0f);
        constexpr double PI = 3.14159265358979323846;
        for (int k = 0; k < FREQ_BINS; ++k) {
            for (int n = 0; n < N_FFT; ++n) {
                const double ang = 2.0 * PI * k * n / N_FFT;
                stft_w_real_[k * N_FFT + n] =
                    static_cast<float>(std::cos(ang)) * stft_window_[n];
                stft_w_imag_[k * N_FFT + n] =
                    static_cast<float>(-std::sin(ang)) * stft_window_[n];
            }
        }
        // Inverse: idft_cos = cos(2*pi*n*k/N).T (FREQ_BINS, N_FFT)
        //          idft_sin = sin(2*pi*n*k/N).T
        // Multiply by inv_window = window * (1/N_FFT)
        istft_w_cos_.assign(static_cast<size_t>(FREQ_BINS) * N_FFT, 0.0f);
        istft_w_sin_.assign(static_cast<size_t>(FREQ_BINS) * N_FFT, 0.0f);
        const float inv_scale = 1.0f / static_cast<float>(N_FFT);
        for (int k = 0; k < FREQ_BINS; ++k) {
            for (int n = 0; n < N_FFT; ++n) {
                const double ang = 2.0 * PI * n * k / N_FFT;
                const float cs = static_cast<float>(std::cos(ang));
                const float sn = static_cast<float>(std::sin(ang));
                const float wn = stft_window_[n] * inv_scale;
                istft_w_cos_[k * N_FFT + n] = cs * wn;
                istft_w_sin_[k * N_FFT + n] = sn * wn;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Generator load_weights failed: %s\n", e.what());
        return false;
    }
    return true;
}

// ============================================================================
// CustomSTFT.transform — replicate-pad + conv1d (real, imag) -> (mag, phase).
//
// Replicate pad: prepend `pad_len` copies of waveform[0], append `pad_len`
// copies of waveform[T-1]. pad_len = N_FFT/2 = 10.
// T_out = (T + 2*pad_len - N_FFT)/hop + 1
//       = (T + N_FFT - N_FFT)/hop + 1 = T/hop + 1.   (only when N_FFT==2*pad_len)
// ============================================================================
void Generator::compute_har_stft(const float* har_source, int har_len,
                                 std::vector<float>& mag,
                                 std::vector<float>& phase) const {
    constexpr int pad_len = N_FFT / 2;   // 10
    const int padded_len = har_len + 2 * pad_len;
    std::vector<float> padded(padded_len, 0.0f);
    const float left = har_source[0];
    const float right = har_source[har_len - 1];
    for (int i = 0; i < pad_len; ++i) {
        padded[i] = left;
        padded[padded_len - 1 - i] = right;
    }
    std::memcpy(&padded[pad_len], har_source, har_len * sizeof(float));

    const int T_out = (padded_len - N_FFT) / HOP_LENGTH + 1;
    mag.assign(static_cast<size_t>(FREQ_BINS) * T_out, 0.0f);
    phase.assign(static_cast<size_t>(FREQ_BINS) * T_out, 0.0f);

    constexpr float PI_F = 3.14159265358979323846f;
    for (int t = 0; t < T_out; ++t) {
        const int start = t * HOP_LENGTH;
        for (int k = 0; k < FREQ_BINS; ++k) {
            float r = 0.0f, im = 0.0f;
            const float* wr = &stft_w_real_[k * N_FFT];
            const float* wi = &stft_w_imag_[k * N_FFT];
            for (int n = 0; n < N_FFT; ++n) {
                const float xv = padded[start + n];
                r  += wr[n] * xv;
                im += wi[n] * xv;
            }
            // magnitude
            const float m = std::sqrt(r * r + im * im + 1e-14f);
            mag[k * T_out + t] = m;
            // phase = atan2(imag, real); when imag==0 and real<0 -> +pi (per CustomSTFT)
            float ph = std::atan2(im, r);
            if (im == 0.0f && r < 0.0f) ph = PI_F;
            phase[k * T_out + t] = ph;
        }
    }
}

// ============================================================================
// CustomSTFT.inverse — conv_transpose1d with backward kernels, then crop.
//
//   real_part = mag * cos(phase)   shape (FREQ_BINS, T_frames)
//   imag_part = mag * sin(phase)
//   real_rec = ConvT1d(real_part, w_cos)  stride=hop, padding=0, K=N_FFT
//   imag_rec = ConvT1d(imag_part, w_sin)
//   wave = real_rec - imag_rec
//   wave = wave[..., pad_len : -pad_len]
//
// Backward kernel weight shape used by PyTorch ConvT1d: (in=FREQ_BINS,
// out=1, K=N_FFT). Mathematically:
//   wave[t_out] = sum_k sum_freq w[freq, 0, k] * input[freq, i_in]
//   where t_out = i_in*stride + k.
// ============================================================================
static void istft_inverse(const std::vector<float>& mag,
                          const std::vector<float>& phase,
                          int FREQ_BINS, int T_frames,
                          int N_FFT, int HOP,
                          const std::vector<float>& w_cos,   // (FREQ_BINS, N_FFT)
                          const std::vector<float>& w_sin,
                          std::vector<float>& audio_out,
                          int target_len) {
    // pre-compute real/imag parts
    const int total_len = (T_frames - 1) * HOP + N_FFT;
    std::vector<float> wave(total_len, 0.0f);
    for (int t = 0; t < T_frames; ++t) {
        const int t_out_base = t * HOP;
        for (int k = 0; k < FREQ_BINS; ++k) {
            const float m = mag[k * T_frames + t];
            const float p = phase[k * T_frames + t];
            const float r = m * std::cos(p);
            const float im = m * std::sin(p);
            const float* wc = &w_cos[k * N_FFT];
            const float* ws = &w_sin[k * N_FFT];
            for (int n = 0; n < N_FFT; ++n) {
                // wave += real_part * w_cos - imag_part * w_sin
                wave[t_out_base + n] += r * wc[n] - im * ws[n];
            }
        }
    }
    // Crop pad_len from each side (center=True -> pad_len = N_FFT/2)
    const int pad_len = N_FFT / 2;
    const int cropped_len = total_len - 2 * pad_len;
    audio_out.assign(static_cast<size_t>(target_len), 0.0f);
    const int copy_len = std::min(target_len, cropped_len);
    std::memcpy(audio_out.data(), wave.data() + pad_len,
                copy_len * sizeof(float));
}

// ============================================================================
// SineGen / SourceModuleHnNSF — production mode (PRNG-driven).
//
// Generates har_source of length 600*T_mel from F0 of length 2*T_mel:
//   1. f0_upsampled = nearest-upsample(F0, factor=300). shape (600*T_mel,).
//      Wait — istftnet uses nn.Upsample(scale_factor=300) which is nearest by
//      default for 1D. Need to verify... actually nn.Upsample default mode is
//      'nearest'. So replicate each F0 sample 300 times.
//      Actually no — reading more carefully, the input is shape (1, 1, 2*T_mel),
//      and nearest upsample by 300 gives (1, 1, 600*T_mel), with each sample
//      replicated 300x. Confirmed.
//   2. fn[t, h] = (h+1) * f0_upsampled[t]  for h = 0..8
//   3. rad_values[t, h] = (fn[t, h] / sampling_rate) % 1
//   4. rand_ini[h] = uniform(0,1); rand_ini[0] = 0
//      rad_values[0, h] += rand_ini[h]
//   5. Downsample rad_values by 1/300 (linear), cumsum*2pi, upsample by 300
//      (linear), sin -> sine_waves
//   6. uv[t] = (f0_upsampled[t] > voiced_threshold) ? 1 : 0
//   7. noise_amp = uv * noise_std + (1-uv) * sine_amp/3
//      noise = noise_amp * randn_like(sine_waves)
//   8. sine_waves = sine_waves * sine_amp * uv + noise
//   9. sine_merge = tanh(linear(sine_waves))   shape (600*T_mel,)
//   10. har_source = sine_merge.transpose.squeeze
//
// For production we replicate this. For the tests we use `forward_with_har_source`.
// ============================================================================
namespace {

void generate_har_source(const float* F0, int T_mel,
                         const float* l_linear_w,   // (9,)
                         float l_linear_b,
                         uint64_t seed,
                         std::vector<float>& har_out) {
    constexpr int upsample_scale = Generator::UPSAMPLE_SCALE;  // 300
    constexpr int dim = Generator::HARMONICS;                  // 9
    constexpr float sine_amp = Generator::SINE_AMP;            // 0.1
    constexpr float noise_std = Generator::NOISE_STD;          // 0.003
    constexpr float voiced_threshold = Generator::VOICED_THRESHOLD;  // 10
    constexpr int sr = Generator::SAMPLING_RATE;               // 24000
    constexpr double PI = 3.14159265358979323846;

    const int T_in = 2 * T_mel;
    const int T_audio = T_in * upsample_scale;

    // 1. nearest-upsample F0 to audio rate
    std::vector<float> f0_up(T_audio, 0.0f);
    for (int t = 0; t < T_in; ++t) {
        const float v = F0[t];
        for (int s = 0; s < upsample_scale; ++s) {
            f0_up[t * upsample_scale + s] = v;
        }
    }

    // 2-4. Build rad_values per harmonic, downsample by 1/upsample_scale
    // (linear interpolation), cumsum, upsample back by upsample_scale, sin.
    // To follow istftnet exactly we work at the original rate (T_in samples
    // per harmonic) for cumsum, then upsample by 300x linear.
    // rad_values_t[t, h] = ((h+1)*F0[t] / sr) % 1 + (h>0 ? rand_ini[h] : 0)
    // but rand_ini only applied at t=0.

    std::mt19937_64 rng(seed ? seed :
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> rand_ini(dim, 0.0f);
    for (int h = 1; h < dim; ++h) rand_ini[h] = uni01(rng);

    // rad_values shape (T_in, dim) but the linear interpolate in the reference
    // operates on the upsampled-rate values: rad_values is computed at T_audio
    // rate, then F.interpolate(scale=1/upsample_scale, mode=linear)
    // downsamples to T_in, then cumsum, then upsample back to T_audio.
    //
    // F.interpolate with scale_factor=1/300 mode='linear', input length T_audio,
    // outputs length T_audio/300 = T_in (assuming exact divisibility).
    //
    // The downsampled value at index t' (in [0..T_in)) is:
    //   linear interpolation of rad_at_audio_rate at the position
    //   (t' + 0.5) * upsample_scale - 0.5  in input indices (default
    //   align_corners=False).
    // But in our case, rad_values[i] is actually constant on each block of
    // upsample_scale samples (since f0_up was nearest-upsampled). Then linear
    // downsample by 1/upsample_scale gives... not exactly the per-block value.
    //
    // The interpolation is sampled at midpoints (align_corners=False):
    //   src_pos = (out_idx + 0.5) * (in_len / out_len) - 0.5
    // For in_len = T_audio, out_len = T_in, in_len/out_len = upsample_scale.
    // src_pos = (out_idx + 0.5)*upsample_scale - 0.5
    //         = out_idx*upsample_scale + (upsample_scale - 1)/2 + 0.5 - 0.5
    //         = out_idx*upsample_scale + (upsample_scale - 1)/2
    //         = out_idx*300 + 149.5
    //
    // The block boundaries are at 0, 300, 600, ... So src_pos for out_idx=0
    // is 149.5, which is squarely in block 0 (indices 0..299) — value is
    // F0[0]. For out_idx=1, src_pos = 449.5, in block 1 — value is F0[1].
    // So the downsampled rad_values are exactly the per-block rad values.
    //
    // But wait — at the boundary of two blocks, e.g. between samples 299
    // (F0[0]) and 300 (F0[1]), linear interpolation between adjacent
    // audio-rate samples could give intermediate values. But since src_pos
    // at out_idx falls inside a block (not at a boundary), the interpolation
    // gives the exact block value (both neighbors equal).
    //
    // Conclusion: downsampling is equivalent to F0 at original rate, scaled
    // by harmonic factors. Compute directly.
    std::vector<float> rad(T_in * dim, 0.0f);  // (T_in, dim)
    for (int t = 0; t < T_in; ++t) {
        const float f = F0[t];
        for (int h = 0; h < dim; ++h) {
            const float v = (h + 1) * f / static_cast<float>(sr);
            // (...) % 1   PyTorch fmod-like behavior, but for positive f -> just
            // fractional part. Reference uses floor-mod (torch's `%` operator).
            float vmod = v - std::floor(v);
            rad[t * dim + h] = vmod;
        }
    }
    // rand_ini applied at t=0 (after % 1 reduction)
    for (int h = 0; h < dim; ++h) {
        rad[0 * dim + h] += rand_ini[h];
    }

    // cumsum along time axis, multiply by 2*pi
    std::vector<float> phase_in(T_in * dim, 0.0f);
    for (int h = 0; h < dim; ++h) {
        double acc = 0.0;
        for (int t = 0; t < T_in; ++t) {
            acc += rad[t * dim + h];
            phase_in[t * dim + h] = static_cast<float>(acc * 2.0 * PI);
        }
    }

    // F.interpolate(phase * upsample_scale, scale_factor=upsample_scale, mode=linear)
    // — upsample by 300x with linear interp + multiplied by upsample_scale first.
    // This is critical: the ref code multiplies phase by upsample_scale before
    // upsampling, because cumulative sum at downsampled rate accumulates
    // upsample_scale-times-fewer increments. So phase_audio = upsampled(phase * 300).
    //
    // align_corners=False default. src_pos formula:
    //   src_pos = (out_idx + 0.5)*(T_in/T_audio) - 0.5
    //           = (out_idx + 0.5)/upsample_scale - 0.5
    // For out_idx=0: src_pos = 0.5/300 - 0.5 = -0.49833... -> clamped to 0
    // For out_idx=300: src_pos = 300.5/300 - 0.5 = 1.001666... - 0.5 = 0.50166
    // ... etc.
    //
    // Because of the ×300 scaling, phase_audio is effectively the cumulative
    // phase at audio rate. We then take sin().
    std::vector<float> sine_waves(T_audio * dim, 0.0f);
    for (int h = 0; h < dim; ++h) {
        for (int out_idx = 0; out_idx < T_audio; ++out_idx) {
            float src_pos = (out_idx + 0.5f) / static_cast<float>(upsample_scale) - 0.5f;
            if (src_pos < 0.0f) src_pos = 0.0f;
            if (src_pos > T_in - 1) src_pos = static_cast<float>(T_in - 1);
            const int idx0 = static_cast<int>(std::floor(src_pos));
            const int idx1 = std::min(idx0 + 1, T_in - 1);
            const float frac = src_pos - idx0;
            const float p0 = phase_in[idx0 * dim + h] * upsample_scale;
            const float p1 = phase_in[idx1 * dim + h] * upsample_scale;
            const float ph = (1.0f - frac) * p0 + frac * p1;
            sine_waves[out_idx * dim + h] = std::sin(ph);
        }
    }

    // Multiply by sine_amp
    for (size_t i = 0; i < sine_waves.size(); ++i) sine_waves[i] *= sine_amp;

    // Compute uv at audio rate
    std::vector<float> uv(T_audio, 0.0f);
    for (int t = 0; t < T_audio; ++t) {
        uv[t] = (f0_up[t] > voiced_threshold) ? 1.0f : 0.0f;
    }

    // Add noise + uv-mask
    for (int t = 0; t < T_audio; ++t) {
        const float u = uv[t];
        const float noise_amp = u * noise_std + (1.0f - u) * (sine_amp / 3.0f);
        for (int h = 0; h < dim; ++h) {
            const float n = noise_amp * normal(rng);
            sine_waves[t * dim + h] = sine_waves[t * dim + h] * u + n;
        }
    }

    // sine_merge = tanh(linear(sine_waves))   l_linear: (1, 9) -> (1,)
    har_out.assign(T_audio, 0.0f);
    for (int t = 0; t < T_audio; ++t) {
        float acc = l_linear_b;
        for (int h = 0; h < dim; ++h) {
            acc += l_linear_w[h] * sine_waves[t * dim + h];
        }
        har_out[t] = std::tanh(acc);
    }
}

}  // namespace

// ============================================================================
// run_until_stage_impl — main pipeline.
// ============================================================================
void Generator::run_until_stage_impl(const float* x_decode3, int T_mel,
                                     const float* har_source, int har_len,
                                     const float* style128,
                                     int stop_stage,
                                     std::vector<float>& out,
                                     std::vector<float>* audio_out) const {
    (void)har_len;

    // ---- 1. Compute har_spec, har_phase, concat to har: (22, T_har) -------
    std::vector<float> har_mag, har_phase;
    compute_har_stft(har_source, har_len, har_mag, har_phase);
    const int T_har = static_cast<int>(har_mag.size()) / FREQ_BINS;
    std::vector<float> har(static_cast<size_t>(HAR_FEATS) * T_har, 0.0f);
    std::memcpy(&har[0], har_mag.data(),
                static_cast<size_t>(FREQ_BINS) * T_har * sizeof(float));
    std::memcpy(&har[static_cast<size_t>(FREQ_BINS) * T_har],
                har_phase.data(),
                static_cast<size_t>(FREQ_BINS) * T_har * sizeof(float));

    if (stop_stage == static_cast<int>(Stage::AfterHarStft)) {
        out = har;
        return;
    }

    // ---- 2. Convert input from row-major (2*T_mel, 512) to channel-first
    //         (512, 2*T_mel)
    const int T_x = 2 * T_mel;
    std::vector<float> x(static_cast<size_t>(IN_CH) * T_x, 0.0f);
    for (int t = 0; t < T_x; ++t) {
        for (int c = 0; c < IN_CH; ++c) {
            x[static_cast<size_t>(c) * T_x + t] =
                x_decode3[static_cast<size_t>(t) * IN_CH + c];
        }
    }
    int Cx = IN_CH;
    int Tx = T_x;

    const int up_strides[2]   = {10, 6};
    const int up_kernels[2]   = {20, 12};
    const int up_paddings[2]  = {(20 - 10) / 2, (12 - 6) / 2};   // {5, 3}
    const int noise_kernels[2]  = {12, 1};
    const int noise_strides[2]  = {6,  1};
    const int noise_paddings[2] = {(6 + 1) / 2, 0};   // {3, 0}
    const int out_chs[2]      = {UPS0_OUT_CH, UPS1_OUT_CH};

    for (int i = 0; i < 2; ++i) {
        // F.leaky_relu(x, 0.1)
        internal::leaky_relu_inplace(x.data(), x.size(), 0.1f);

        // x_source = noise_convs[i](har), then noise_res[i](x_source, s)
        std::vector<float> x_source;
        int T_src = 0;
        conv1d_strided_dil(har.data(), HAR_FEATS, T_har,
                           noise_convs_[i].w.data(), noise_convs_[i].b.data(),
                           out_chs[i], noise_kernels[i],
                           noise_strides[i], noise_paddings[i], /*dilation=*/1,
                           x_source, T_src);
        run_adain_resblock1(noise_res_[i], style128, x_source, T_src);

        // x = ups[i](x)
        std::vector<float> x_up;
        conv_transpose_1d(ups_[i], x, Cx, out_chs[i], Tx,
                          up_kernels[i], up_strides[i], up_paddings[i], x_up);
        const int T_up = (Tx - 1) * up_strides[i] - 2 * up_paddings[i] + up_kernels[i];

        // reflection_pad((1, 0)) ONLY at i == num_upsamples-1 (i.e. i==1).
        std::vector<float> x_padded;
        int Tp = T_up;
        int Cp = out_chs[i];
        if (i == 1) {
            // Insert one reflected sample at the front: x[1] (mirror around index 0).
            // ReflectionPad1d((1,0)) prepends 1 element by reflecting index 1.
            Tp = T_up + 1;
            x_padded.assign(static_cast<size_t>(Cp) * Tp, 0.0f);
            for (int c = 0; c < Cp; ++c) {
                const float* in_row = &x_up[static_cast<size_t>(c) * T_up];
                float* out_row = &x_padded[static_cast<size_t>(c) * Tp];
                // Reflection: prepended sample is in[1] (mirror around index 0)
                out_row[0] = (T_up > 1) ? in_row[1] : in_row[0];
                std::memcpy(&out_row[1], in_row, T_up * sizeof(float));
            }
        }

        if (i == 0 && stop_stage == static_cast<int>(Stage::AfterUps0)) {
            out = x_up;
            return;
        }
        if (i == 1 && stop_stage == static_cast<int>(Stage::AfterUps1)) {
            // Reference fixture is BEFORE reflection_pad (length 13440).
            out = x_up;
            return;
        }

        // x = x_padded + x_source  (or x_up + x_source if i==0)
        if (i == 1) {
            // verify shapes: x_padded (Cp, Tp) and x_source must match
            if ((int)x_source.size() != Cp * Tp) {
                std::fprintf(stderr,
                             "WARN: i=1 size mismatch: x_padded=(%d,%d)=%d, x_source=%zu\n",
                             Cp, Tp, Cp*Tp, x_source.size());
            }
            for (size_t k = 0; k < x_padded.size(); ++k) x_padded[k] += x_source[k];
            x = std::move(x_padded);
            Cx = Cp; Tx = Tp;
        } else {
            for (size_t k = 0; k < x_up.size(); ++k) x_up[k] += x_source[k];
            x = std::move(x_up);
            Cx = out_chs[i]; Tx = T_up;
        }

        // MRF: average of NUM_KERNELS resblocks
        std::vector<float> sum_buf(x.size(), 0.0f);
        for (int j = 0; j < NUM_KERNELS; ++j) {
            std::vector<float> tmp = x;
            run_adain_resblock1(resblocks_[i * NUM_KERNELS + j], style128,
                                tmp, Tx);
            for (size_t k = 0; k < tmp.size(); ++k) sum_buf[k] += tmp[k];
        }
        const float inv_k = 1.0f / static_cast<float>(NUM_KERNELS);
        for (size_t k = 0; k < sum_buf.size(); ++k) sum_buf[k] *= inv_k;
        x = std::move(sum_buf);

        if (i == 0 && stop_stage == static_cast<int>(Stage::AfterResblocks0)) {
            out = x;
            return;
        }
        if (i == 1 && stop_stage == static_cast<int>(Stage::AfterResblocks1)) {
            out = x;
            return;
        }
    }

    // ---- conv_post: leaky_relu (default 0.01 slope), then Conv1d 128 -> 22
    internal::leaky_relu_inplace(x.data(), x.size(), 0.01f);
    std::vector<float> post;
    int T_post = 0;
    conv1d_strided_dil(x.data(), Cx, Tx,
                       conv_post_.w.data(), conv_post_.b.data(),
                       HAR_FEATS, /*K=*/7, /*stride=*/1, /*pad=*/3, /*dil=*/1,
                       post, T_post);
    if (stop_stage == static_cast<int>(Stage::AfterConvPost)) {
        out = post;
        return;
    }

    // ---- spec = exp(x[:11]); phase = sin(x[11:])
    // Then iSTFT.
    std::vector<float> spec(static_cast<size_t>(FREQ_BINS) * T_post, 0.0f);
    std::vector<float> phs(static_cast<size_t>(FREQ_BINS) * T_post, 0.0f);
    for (int k = 0; k < FREQ_BINS; ++k) {
        for (int t = 0; t < T_post; ++t) {
            spec[k * T_post + t] = std::exp(post[k * T_post + t]);
            phs[k * T_post + t]  = std::sin(post[(FREQ_BINS + k) * T_post + t]);
        }
    }

    if (audio_out) {
        const int target_audio_len = 600 * T_mel;
        istft_inverse(spec, phs, FREQ_BINS, T_post,
                      N_FFT, HOP_LENGTH,
                      istft_w_cos_, istft_w_sin_,
                      *audio_out, target_audio_len);
    }
}

void Generator::run_until_stage_with_har_source(const float* x_decode3, int T_mel,
                                                const float* har_source, int har_len,
                                                const float* style128,
                                                Stage last,
                                                std::vector<float>& out) const {
    run_until_stage_impl(x_decode3, T_mel, har_source, har_len, style128,
                         static_cast<int>(last), out, nullptr);
}

void Generator::compute_stft_components_with_har_source(
        const float* x_decode3, int T_mel,
        const float* har_source, int har_len,
        const float* style128,
        std::vector<float>& real_part,
        std::vector<float>& imag_part) const {
    // Run up through conv_post, then build spec*cos / spec*sin
    std::vector<float> post;
    run_until_stage_with_har_source(x_decode3, T_mel, har_source, har_len,
                                    style128, Stage::AfterConvPost, post);
    const int T_post = static_cast<int>(post.size()) / HAR_FEATS;
    real_part.assign(static_cast<size_t>(FREQ_BINS) * T_post, 0.0f);
    imag_part.assign(static_cast<size_t>(FREQ_BINS) * T_post, 0.0f);
    for (int k = 0; k < FREQ_BINS; ++k) {
        for (int t = 0; t < T_post; ++t) {
            const float spec = std::exp(post[k * T_post + t]);
            const float phs  = std::sin(post[(FREQ_BINS + k) * T_post + t]);
            real_part[k * T_post + t] = spec * std::cos(phs);
            imag_part[k * T_post + t] = spec * std::sin(phs);
        }
    }
}

void Generator::forward_with_har_source(const float* x_decode3, int T_mel,
                                        const float* har_source, int har_len,
                                        const float* style128,
                                        std::vector<float>& audio) const {
    std::vector<float> dummy;
    run_until_stage_impl(x_decode3, T_mel, har_source, har_len, style128,
                         /*stop_stage=*/-1, dummy, &audio);
}

void Generator::forward(const float* x_decode3, int T_mel,
                        const float* F0, const float* style128,
                        std::vector<float>& audio,
                        uint64_t seed) const {
    std::vector<float> har;
    generate_har_source(F0, T_mel,
                        m_source_linear_w_.data(),
                        m_source_linear_b_,
                        seed,
                        har);
    forward_with_har_source(x_decode3, T_mel, har.data(),
                            static_cast<int>(har.size()),
                            style128, audio);
}

}}  // namespace cactus::kokoro
