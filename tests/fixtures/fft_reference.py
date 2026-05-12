"""Generate scipy FFT reference outputs for cactus_fft tests.

Run: python tests/fixtures/fft_reference.py
Outputs: tests/fixtures/data/fft_*.bin (generated, gitignored)

Each .bin is little-endian float32 with a uint32 length prefix:
  [n_floats: u32][f0: f32][f1: f32]...[f_{n-1}: f32]

Spectra are serialized as interleaved [re_0, im_0, re_1, im_1, ..., re_{N/2}, im_{N/2}]
matching scipy.fft.rfft output layout (im_0 and im_{N/2} are always 0).
"""

import os
import struct
import numpy as np
from scipy.fft import rfft, irfft

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)


def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))  # length prefix (uint32)
        f.write(arr.tobytes())


def rfft_to_interleaved(spec, n_total):
    """Convert scipy.rfft complex output (length n//2+1) to flat interleaved float32."""
    out = np.empty(spec.size * 2, dtype=np.float32)
    out[0::2] = spec.real
    out[1::2] = spec.imag
    return out


# Test 1: 256-point rfft of a 5-cycle sine
N = 256
sig = np.sin(2 * np.pi * 5 * np.arange(N) / N).astype(np.float32)
write_f32(os.path.join(OUT, "fft_input_256_sine.bin"), sig)
write_f32(os.path.join(OUT, "fft_output_256_sine.bin"), rfft_to_interleaved(rfft(sig), N))

# Test 2: 512-point rfft of a deterministic random signal (seeded)
np.random.seed(42)
sig = np.random.randn(512).astype(np.float32)
write_f32(os.path.join(OUT, "fft_input_512_random.bin"), sig)
write_f32(os.path.join(OUT, "fft_output_512_random.bin"), rfft_to_interleaved(rfft(sig), 512))

# Test 3: iFFT round-trip — single-bin spectrum at k=5 → known time signal
N = 256
spec = np.zeros(N // 2 + 1, dtype=np.complex64)
spec[5] = 1.0  # single bin at k=5
sig = irfft(spec, n=N).astype(np.float32)
write_f32(os.path.join(OUT, "ifft_input_256_singlebin.bin"), rfft_to_interleaved(spec, N))
write_f32(os.path.join(OUT, "ifft_output_256_singlebin.bin"), sig)

from scipy.signal import stft, istft

# Test 4: STFT/ISTFT round-trip on a chirp signal
fs = 24000
duration = 0.5
N = int(fs * duration)
t = np.arange(N) / fs
chirp = np.sin(2 * np.pi * (200 + 1000 * t) * t).astype(np.float32)

n_fft = 512
hop = 128

f, tt, Zxx = stft(chirp, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                  window='hann', boundary=None, padded=False)
write_f32(os.path.join(OUT, "stft_input_chirp.bin"), chirp)

# Save Zxx as interleaved [re, im] per frame, frame-major:
# layout: n_frames * (n_fft//2+1) * 2 floats
n_freq = n_fft // 2 + 1
n_frames = Zxx.shape[1]
stft_out = np.empty((n_frames, n_freq, 2), dtype=np.float32)
stft_out[..., 0] = Zxx.T.real
stft_out[..., 1] = Zxx.T.imag
write_f32(os.path.join(OUT, "stft_output_chirp.bin"), stft_out)

# ISTFT round-trip via scipy reference
_, recovered = istft(Zxx, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                     window='hann', boundary=None)
recovered = recovered[:N].astype(np.float32)
write_f32(os.path.join(OUT, "istft_output_chirp.bin"), recovered)

print(f"STFT params: n_fft={n_fft}, hop={hop}, window=hann, n_frames={n_frames}")

print(f"Wrote fixtures to {OUT}")
print(f"  Files: {sorted(os.listdir(OUT))}")
