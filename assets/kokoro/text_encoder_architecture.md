# Kokoro `text_encoder` — Reference Architecture

This is the canonical PyTorch architecture of the Kokoro-82M `text_encoder`
submodule, extracted from the upstream reference implementation. It is the
spec the cactus C++ port must reproduce bit-for-bit (modulo float rounding).

## Reference source

- Package: `kokoro` (PyPI, v0.9.4) — maintained by hexgrad.
- File: `kokoro/modules.py`
- Class: `TextEncoder`
- Lines: **35–69** (definition + forward).
- Header comment in that file: `# https://github.com/yl4579/StyleTTS2/blob/main/models.py`
  — i.e. the architecture is inherited unchanged from StyleTTS2 (Li et al. 2023).

Local install path on this dev machine:
`/opt/homebrew/lib/python3.11/site-packages/kokoro/modules.py`.

## Instantiation (from `kokoro/model.py`, lines 58–61)

```python
self.text_encoder = TextEncoder(
    channels   = config['hidden_dim'],                 # 512
    kernel_size= config['text_encoder_kernel_size'],   # 5
    depth      = config['n_layer'],                    # 3
    n_symbols  = config['n_token'],                    # 178
)
```

Activation defaults to `nn.LeakyReLU(0.2)`.

Config source: `hexgrad/Kokoro-82M/config.json`
(local cache: `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/...`).

## Layer-by-layer (concrete dims for v0_19)

All shapes verified against `assets/kokoro/kokoro-v0_19.pth` (`net/text_encoder/...`).

| # | Layer | PyTorch op | Param shapes | Notes |
|---|-------|------------|--------------|-------|
| 0 | Input | `LongTensor[B, T]` phoneme IDs | — | int64; PAD = 0 implicitly (vocab IDs start at 1) |
| 1 | Embedding | `nn.Embedding(178, 512)` | `weight: (178, 512)` | Default init; lookup only |
| — | (transpose) | `x.transpose(1, 2)` | — | `[B, T, 512] -> [B, 512, T]` |
| — | (mask zero) | `x.masked_fill_(m.unsqueeze(1), 0.0)` | — | `m: BoolTensor[B, T]`; True at pad positions |
| 2a | CNN block 0 — Conv | `weight_norm(nn.Conv1d(512, 512, kernel_size=5, padding=2))` | `weight_v: (512,512,5)`, `weight_g: (512,1,1)`, `bias: (512)` | stride=1, dilation=1, groups=1, `padding_mode='zeros'`; **weight_norm parameterization**: `weight = weight_g * weight_v / ‖weight_v‖₂` (norm over dims 1,2 → keep dim 0) |
| 2b | CNN block 0 — Norm | `LayerNorm(512)` (custom, see below) | `gamma: (512)`, `beta: (512)` | Acts on the *channel* axis; `eps=1e-5`; affine; transposes internally to put C last for `F.layer_norm` |
| 2c | CNN block 0 — Act | `nn.LeakyReLU(0.2)` | — | Negative slope = 0.2 |
| 2d | CNN block 0 — Drop | `nn.Dropout(0.2)` | — | **Identity in eval mode** (we are inference-only) |
| —  | (mask zero) | `x.masked_fill_(m.unsqueeze(1), 0.0)` | — | After every CNN block |
| 3  | CNN block 1 | identical to block 0 | identical shapes | — |
| 4  | CNN block 2 | identical to block 0 | identical shapes | — |
| —  | (transpose) | `x.transpose(1, 2)` | — | `[B, 512, T] -> [B, T, 512]` |
| —  | (pack) | `pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)` | — | Lengths must be on CPU |
| 5  | LSTM | `nn.LSTM(input=512, hidden=256, num_layers=1, batch_first=True, bidirectional=True)` | `weight_ih_l0: (1024, 512)`, `weight_hh_l0: (1024, 256)`, `bias_ih_l0: (1024)`, `bias_hh_l0: (1024)`, plus `_reverse` versions of all four | Stacked-gate order is PyTorch's standard `[i, f, g, o]` (each (256,*)). Initial `(h0, c0)` are zero. |
| —  | (unpack) | `pad_packed_sequence(x, batch_first=True)` | — | Output: `[B, T_unpacked, 512]` (concat of fwd hidden 256 + bwd hidden 256) |
| —  | (transpose + pad) | manual zero-pad to original `T` then `transpose(-1, -2)` | — | `[B, 512, T]` |
| —  | (mask zero) | `x.masked_fill_(m.unsqueeze(1), 0.0)` | — | Final |
| 6  | Output | `FloatTensor[B, 512, T]` | — | Channel-first |

## Custom `LayerNorm` semantics (modules.py:21-32)

```python
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        self.gamma = nn.Parameter(torch.ones(channels))   # (512,)
        self.beta  = nn.Parameter(torch.zeros(channels))  # (512,)
        self.eps   = 1e-5

    def forward(self, x):           # x: [B, C, T]
        x = x.transpose(1, -1)      # -> [B, T, C]
        x = F.layer_norm(x, (C,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)   # -> [B, C, T]
```

Equivalent ops in cactus: per-position (each B,T) normalize across C with affine
`gamma*x_hat + beta`. **NOT** an InstanceNorm and **NOT** a GroupNorm.

## Weight-norm fold-in (already done by the cactus converter)

The converter (`python/src/converter_kokoro.py`) folds `weight_g` and `weight_v`
into a plain `weight` tensor before writing `text_encoder_cnn_*_conv_weight.weights`.
The cactus runtime sees a plain `Conv1d(512, 512, 5, padding=2)` per block.

Folded weight formula (PyTorch `weight_norm`, dim=0 default):
```
w = weight_g * weight_v / ||weight_v||  # norm computed over dims (1, 2)
```

## Input / output contracts

- **Input**:
  - `x: LongTensor[B, T]` — phoneme IDs in `[1, 177]` (vocab).
  - `input_lengths: LongTensor[B]` — actual sequence lengths (used by pack/pad).
  - `m: BoolTensor[B, T]` — text mask (`True` where padding).
- **Output**:
  - `FloatTensor[B, 512, T]` — per-phoneme hidden states, channel-first.
  - Padded positions are zeroed.

For batch-size-1 inference (cactus's normal case) `m` is all False
(no padding), `input_lengths == [T]`. The pack/unpack pair becomes a no-op
on values; cactus can just run a plain bidirectional LSTM over the full sequence.

## Forward pseudocode (10 lines, B=1, no padding)

```
x  = embedding[x_ids]           # [T, 512]
x  = x.T                        # [512, T]
for block in (cnn0, cnn1, cnn2):
    w   = block.weight_g * block.weight_v / norm(block.weight_v)  # (512,512,5)
    x   = conv1d(x, w, bias=block.bias, padding=2)                # [512, T]
    x   = layer_norm_C(x, gamma, beta)                            # along C
    x   = leaky_relu(x, 0.2)
fwd = lstm_forward (x.T, weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd)  # [T, 256]
bwd = lstm_reverse(x.T, weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd)   # [T, 256]
out = concat([fwd, bwd], dim=-1).T                                # [512, T]
return out
```

## Notable details for the C++ port

1. **No final projection** — the bidirectional LSTM's concatenated 512 dims
   are the final output. (Predictor and decoder consume this directly.)
2. **No pooling** — output is per-time-step, length T (same as input).
3. **No sinusoidal/positional encoding** — text_encoder uses none.
4. **CNN padding** is `padding=2` for `kernel_size=5` (i.e. SAME padding;
   output T equals input T).
5. **Dropout is OFF** in eval/inference (we always run in eval mode).
6. **Mask zeroing is a no-op** for B=1 unpadded inputs but the C++
   implementation should still zero pad positions if it ever batches.
7. **LSTM gate order**: PyTorch packs `[i, f, g, o]` — cactus's
   `cactus_lstm_cell` expects the same ordering (verified by converter
   `repack_lstm_*` helpers).
