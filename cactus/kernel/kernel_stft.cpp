#include "kernel_stft.h"
#include "kernel_fft.h"

#include <cmath>
#include <cstring>
#include <vector>

static std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * static_cast<float>(i)
                                       / static_cast<float>(n - 1)));
    }
    return w;
}

void cactus_stft(const float* input, int n_samples,
                 float* output, int n_fft, int hop, int n_frames) {
    auto window = hann_window(n_fft);
    std::vector<float> frame(n_fft);
    const int n_freq = n_fft / 2 + 1;

    for (int f = 0; f < n_frames; ++f) {
        const int start = f * hop;
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = (start + i < n_samples) ? input[start + i] * window[i] : 0.0f;
        }
        cactus_fft_r2c(frame.data(), output + static_cast<size_t>(f) * n_freq * 2, n_fft);
    }
}

void cactus_istft(const float* input, int n_frames,
                  float* output, int n_samples,
                  int n_fft, int hop) {
    auto window = hann_window(n_fft);
    std::vector<float> frame(n_fft);
    std::vector<float> norm(n_samples, 0.0f);
    const int n_freq = n_fft / 2 + 1;

    std::memset(output, 0, static_cast<size_t>(n_samples) * sizeof(float));

    for (int f = 0; f < n_frames; ++f) {
        cactus_fft_c2r(input + static_cast<size_t>(f) * n_freq * 2, frame.data(), n_fft);
        const int start = f * hop;
        for (int i = 0; i < n_fft; ++i) {
            const int idx = start + i;
            if (idx < n_samples) {
                output[idx] += frame[i] * window[i];
                norm[idx]   += window[i] * window[i];
            }
        }
    }

    // Normalize by window-sum to invert the synthesis windowing
    for (int i = 0; i < n_samples; ++i) {
        if (norm[i] > 1e-8f) output[i] /= norm[i];
    }
}
