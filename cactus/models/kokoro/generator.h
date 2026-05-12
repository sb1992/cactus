#pragma once

#include "kokoro_internal.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Stage 2 of the Kokoro decoder — the Hi-Fi-GAN-style ISTFTNet generator.
// Converts the (2*T_mel, 512) feature map produced by DecoderEncodeStage
// into 24 kHz PCM audio of length 600*T_mel.
//
// Architecture (see assets/kokoro/decoder_architecture.md, Generator section):
//
//   m_source : SourceModuleHnNSF — sinusoidal harmonic generator from F0
//              + tanh(linear(9->1)). Output: har_source (1, 600*T_mel).
//   stft     : CustomSTFT (filter=20, hop=5, win=20, replicate-pad).
//              Computes magnitude+phase of har_source, concat -> (1, 22, T_har).
//   ups[0]   : weight-normed ConvT1d 512->256, k=20, s=10, p=5. Doubles the
//              T_mel axis to 20*T_mel.
//   noise_convs[0]: Conv1d 22->256, k=12, s=6, p=3. Downsamples har to match.
//   noise_res[0]:  AdaINResBlock1 (Snake1d), ch=256, k=7.
//   resblocks[0..2]: AdaINResBlock1 (Snake1d), ch=256, k={3,7,11}, dil={1,3,5}.
//                    Outputs averaged.
//
//   ups[1]   : ConvT1d 256->128, k=12, s=6, p=3.
//   reflection_pad : (1,0) — extends time axis from 120*T_mel to 120*T_mel+1
//   noise_convs[1]: Conv1d 22->128, k=1.
//   noise_res[1] : AdaINResBlock1 (Snake1d), ch=128, k=11.
//   resblocks[3..5]: ch=128, k={3,7,11}.
//
//   conv_post : weight-normed Conv1d 128->22, k=7, p=3.
//   spec  = exp(x[:11]); phase = sin(x[11:]).
//   audio = stft.inverse(spec * (cos(phase) + j*sin(phase))).
//
// Two forward modes:
//   * forward(): production — generates har_source internally via std::mt19937
//   * forward_with_har_source(): test mode — accepts pre-generated har_source
//                                (used to bypass PRNG mismatches in golden tests)
class Generator {
public:
    enum class Stage {
        AfterHarStft,        // (22, T_har)
        AfterUps0,           // (256, 20*T_mel) channel-first
        AfterResblocks0,     // (256, 20*T_mel) channel-first
        AfterUps1,           // (128, 120*T_mel) channel-first (PRE reflection_pad)
        AfterResblocks1,     // (128, 120*T_mel + 1) channel-first
        AfterConvPost,       // (22, 120*T_mel + 1)  channel-first
    };

    static constexpr int STYLE_DIM      = 128;
    static constexpr int N_FFT          = 20;
    static constexpr int HOP_LENGTH     = 5;
    static constexpr int FREQ_BINS      = N_FFT / 2 + 1;       // 11
    static constexpr int HAR_FEATS      = 2 * FREQ_BINS;       // 22 (mag+phase concat)
    static constexpr int UPSAMPLE_SCALE = 300;                 // prod([10,6])*5
    static constexpr int HARMONIC_NUM   = 8;                   // dim = 9
    static constexpr int HARMONICS      = HARMONIC_NUM + 1;
    static constexpr int SAMPLING_RATE  = 24000;
    static constexpr float SINE_AMP     = 0.1f;
    static constexpr float NOISE_STD    = 0.003f;
    static constexpr float VOICED_THRESHOLD = 10.0f;

    static constexpr int IN_CH         = 512;
    static constexpr int UPS0_OUT_CH   = 256;
    static constexpr int UPS1_OUT_CH   = 128;
    static constexpr int NUM_KERNELS   = 3;
    static constexpr float INSTNORM_EPS = 1e-5f;

    bool load_weights(const std::string& weights_dir);

    // Production forward — generates har_source internally via std::mt19937.
    //   x_decode3:  (2*T_mel, 512) row-major from DecoderEncodeStage::forward
    //   F0:         (2*T_mel,)
    //   style128:   (128,)
    //   audio:      (600*T_mel,) PCM
    //   seed:       PRNG seed (0 = current time)
    void forward(const float* x_decode3, int T_mel,
                 const float* F0, const float* style128,
                 std::vector<float>& audio,
                 uint64_t seed = 0) const;

    // Test mode — accept pre-generated har_source (bypass PRNG).
    //   har_len must equal 600*T_mel.
    void forward_with_har_source(const float* x_decode3, int T_mel,
                                 const float* har_source, int har_len,
                                 const float* style128,
                                 std::vector<float>& audio) const;

    // ---- Test introspection helpers ----
    void run_until_stage_with_har_source(const float* x_decode3, int T_mel,
                                         const float* har_source, int har_len,
                                         const float* style128,
                                         Stage last,
                                         std::vector<float>& out) const;

    // CustomSTFT.transform on har_source, returning magnitude (mag) and phase
    // each as channel-first (FREQ_BINS, T_har) row-major.
    void compute_har_stft(const float* har_source, int har_len,
                          std::vector<float>& mag,
                          std::vector<float>& phase) const;

    // Compute the (real, imag) pair fed to iSTFT. real = spec*cos(phase),
    // imag = spec*sin(phase). Both channel-first (FREQ_BINS, T_frames).
    void compute_stft_components_with_har_source(
        const float* x_decode3, int T_mel,
        const float* har_source, int har_len,
        const float* style128,
        std::vector<float>& real_part,
        std::vector<float>& imag_part) const;

private:
    using Conv1d        = internal::Conv1d;
    using AdaLNFc       = internal::AdaLNFc;

    // AdaINResBlock1 — Snake1d variant. 3 parallel residual sub-blocks.
    // Per sub-block i in {0,1,2}:
    //   xt = adain1[i](x, s);
    //   xt = xt + (1/alpha1[i]) * sin(alpha1[i] * xt)^2;     // Snake1d
    //   xt = convs1[i](xt);    // dilation = (1, 3, 5)[i], padding = get_padding(K, dil)
    //   xt = adain2[i](xt, s);
    //   xt = xt + (1/alpha2[i]) * sin(alpha2[i] * xt)^2;     // Snake1d
    //   xt = convs2[i](xt);    // dilation 1, padding (K-1)/2
    //   x = xt + x;
    struct AdaINResBlock1 {
        int channels = 0;
        int K = 0;                 // kernel size (3, 7, or 11)
        int dilation[3] = {1, 3, 5};
        AdaLNFc adain1_fc[3];
        AdaLNFc adain2_fc[3];
        Conv1d  convs1[3];         // (channels, channels, K), dilated
        Conv1d  convs2[3];         // (channels, channels, K), dilation 1
        std::vector<float> alpha1[3];   // per-channel learnable Snake1d alpha
        std::vector<float> alpha2[3];
    };

    // Apply Snake1d in place to a channel-first (C, T) buffer:
    //   x[c,t] += (1/alpha[c]) * sin(alpha[c] * x[c,t])^2
    static void snake1d_inplace(float* x, int C, int T, const float* alpha);

    // Run AdaINResBlock1 forward — returns (channels, T) channel-first.
    void run_adain_resblock1(const AdaINResBlock1& blk, const float* style128,
                             std::vector<float>& x_cf, int T) const;

    // Conv1d with stride / padding / dilation / no groups.
    static void conv1d_strided_dil(const float* cf, int C_in, int T_in,
                                   const float* weight, const float* bias,
                                   int C_out, int K,
                                   int stride, int padding, int dilation,
                                   std::vector<float>& cf_out, int& T_out);

    // ConvTranspose1d for ups[i] — full mode (groups=1), padding = (k-u)/2.
    static void conv_transpose_1d(const Conv1d& w, const std::vector<float>& in_cf,
                                  int C_in, int C_out, int T_in,
                                  int K, int stride, int padding,
                                  std::vector<float>& out_cf);

    bool load_resblock(AdaINResBlock1& blk, const std::string& dir,
                       const std::string& base, int channels, int K,
                       const int dil[3]);

    // ---- weight buffers ----
    // SourceModuleHnNSF.l_linear: (1, 9) weight + (1,) bias. The (1, 9) row is
    // stored as 9 floats; the bias is a single float.
    std::vector<float> m_source_linear_w_;   // (9,)
    float              m_source_linear_b_ = 0.0f;

    // CustomSTFT precomputed kernels — (FREQ_BINS, N_FFT) row-major.
    // computed once at load_weights() time.
    std::vector<float> stft_window_;          // (N_FFT,) Hann periodic
    std::vector<float> stft_w_real_;          // (FREQ_BINS, N_FFT)
    std::vector<float> stft_w_imag_;          // (FREQ_BINS, N_FFT)
    std::vector<float> istft_w_cos_;          // backward_real = cos*window/N_FFT
    std::vector<float> istft_w_sin_;          // backward_imag = sin*window/N_FFT

    Conv1d noise_convs_[2];   // [0] (256,22,12) k=12 s=6 p=3; [1] (128,22,1) k=1
    AdaINResBlock1 noise_res_[2];  // [0] ch=256 k=7; [1] ch=128 k=11

    Conv1d ups_[2];           // [0] (512,256,20) k=20 s=10 p=5
                              // [1] (256,128,12) k=12 s=6  p=3
                              // (PyTorch ConvT1d weight: (in_channels, out_channels, K))

    // resblocks[i*3 + j]: i in {0,1}, j in {0,1,2}
    AdaINResBlock1 resblocks_[6];   // ch in {256,128}, K in {3,7,11}

    Conv1d conv_post_;        // (22, 128, 7) k=7 p=3

    // Intermediate stage runner — does everything up to and including `last`.
    void run_until_stage_impl(const float* x_decode3, int T_mel,
                              const float* har_source, int har_len,
                              const float* style128,
                              int stop_stage,           // -1 = run all + iSTFT
                              std::vector<float>& out,  // stage output if stop_stage>=0
                              std::vector<float>* audio_out  // populated if stop_stage<0
                              ) const;
};

}}  // namespace cactus::kokoro
