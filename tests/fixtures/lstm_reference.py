"""Generate torch LSTMCell reference outputs for cactus_lstm_cell tests.

LSTMCell signature mirrors PyTorch nn.LSTMCell(input_size, hidden_size).
Small dims chosen for fast tests but real-shaped to catch SIMD edge cases later.

Run: python3 tests/fixtures/lstm_reference.py
Outputs: tests/fixtures/data/lstm_*.bin (gitignored, regenerate locally)

Format: [u32 length][f32 ...]  (same as fft_reference.py write_f32 helper)
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


torch.manual_seed(0)
INPUT_SIZE = 64
HIDDEN_SIZE = 128

cell = torch.nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE).eval()
x  = torch.randn(1, INPUT_SIZE)
h0 = torch.zeros(1, HIDDEN_SIZE)
c0 = torch.zeros(1, HIDDEN_SIZE)
with torch.no_grad():
    h1, c1 = cell(x, (h0, c0))

# Save weights in the order the C++ kernel expects:
#   W_ih: [4*H, I]   (gates ordered i, f, g, o per PyTorch's nn.LSTMCell)
#   W_hh: [4*H, H]
#   b_ih: [4*H]
#   b_hh: [4*H]
write_f32(os.path.join(OUT, "lstm_W_ih.bin"), cell.weight_ih.detach().numpy())
write_f32(os.path.join(OUT, "lstm_W_hh.bin"), cell.weight_hh.detach().numpy())
write_f32(os.path.join(OUT, "lstm_b_ih.bin"), cell.bias_ih.detach().numpy())
write_f32(os.path.join(OUT, "lstm_b_hh.bin"), cell.bias_hh.detach().numpy())
write_f32(os.path.join(OUT, "lstm_x.bin"),    x.numpy())
write_f32(os.path.join(OUT, "lstm_h0.bin"),   h0.numpy())
write_f32(os.path.join(OUT, "lstm_c0.bin"),   c0.numpy())
write_f32(os.path.join(OUT, "lstm_h1.bin"),   h1.numpy())
write_f32(os.path.join(OUT, "lstm_c1.bin"),   c1.numpy())

print(f"LSTM dims: I={INPUT_SIZE} H={HIDDEN_SIZE}")
print(f"Wrote fixtures to {OUT}")
