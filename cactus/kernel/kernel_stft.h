#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Short-Time Fourier Transform with Hann window.
//
//   input:    pointer to n_samples float PCM samples
//   output:   pointer to n_frames * (n_fft/2 + 1) * 2 floats (interleaved re/im, frame-major,
//             scipy.fft.rfft layout per frame)
//   n_fft:    window/FFT size (must be valid for PFFFT — multiple of 32, >= 32)
//   hop:      hop size in samples
//   n_frames: number of frames; should equal (n_samples - n_fft) / hop + 1
void cactus_stft(const float* input, int n_samples,
                 float* output, int n_fft, int hop, int n_frames);

// Inverse STFT with Hann window + overlap-add + window-sum normalization.
//
//   input:    pointer to n_frames * (n_fft/2 + 1) * 2 floats (interleaved re/im, frame-major)
//   output:   pointer to n_samples floats; will be zero-initialized internally
//   n_samples must satisfy n_samples >= (n_frames - 1) * hop + n_fft
void cactus_istft(const float* input, int n_frames,
                  float* output, int n_samples,
                  int n_fft, int hop);

#ifdef __cplusplus
}
#endif
