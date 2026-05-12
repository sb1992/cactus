"""Generate torch ConvTranspose1d reference outputs for cactus_conv1d_transpose tests.

Used by ISTFTNet upsampling layers in Kokoro decoder.

Run: python3 tests/fixtures/conv_reference.py
Outputs: tests/fixtures/data/convT_*.bin (gitignored, regenerate locally)

Format: [u32 length][f32 ...]  (matches fft_reference.py / lstm_reference.py)
"""

import os
import struct
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)


def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


torch.manual_seed(1)

# Test 1: simple — C_in=4, C_out=4, K=4, stride=2, padding=1
# Mimics typical ISTFTNet upsampling: 2x temporal upsample
C_IN, C_OUT, K, STRIDE, PAD = 4, 4, 4, 2, 1
T_IN = 16

conv = torch.nn.ConvTranspose1d(C_IN, C_OUT, K, stride=STRIDE, padding=PAD).eval()
x = torch.randn(1, C_IN, T_IN)
with torch.no_grad():
    y = conv(x)

# T_out = (T_in - 1) * stride - 2*padding + K  = 15*2 - 2 + 4 = 32
T_OUT = (T_IN - 1) * STRIDE - 2 * PAD + K
assert y.shape == (1, C_OUT, T_OUT), f"got {y.shape}, expected (1, {C_OUT}, {T_OUT})"

# weight shape in PyTorch ConvTranspose1d: (C_in, C_out, K)  — note: in, out, k
write_f32(os.path.join(OUT, "convT_weight.bin"), conv.weight.detach().numpy())
write_f32(os.path.join(OUT, "convT_bias.bin"),   conv.bias.detach().numpy())
write_f32(os.path.join(OUT, "convT_x.bin"),      x.numpy())
write_f32(os.path.join(OUT, "convT_y.bin"),      y.numpy())

print(f"ConvT params: C_in={C_IN} C_out={C_OUT} K={K} stride={STRIDE} pad={PAD} T_in={T_IN} T_out={T_OUT}")
