# Kokoro `predictor` — Reference Architecture (StyleTTS2 ProsodyPredictor)

This is the canonical PyTorch architecture of the Kokoro-82M `predictor`
submodule, extracted from the upstream reference implementation. It is the
spec the cactus C++ port (Plan 2 Task 7b) must reproduce bit-for-bit
(modulo float rounding).

## Reference source

- Package: `kokoro` (PyPI, v0.9.4), maintained by hexgrad.
- Files:
  - `kokoro/modules.py` — `ProsodyPredictor` (lines **91–134**),
    `DurationEncoder` (lines **137–176**), `AdaLayerNorm` (lines **72–88**).
  - `kokoro/istftnet.py` — `AdaIN1d` (lines **20–31**),
    `AdainResBlk1d` (lines **340–381**), `UpSample1d` (lines **328–337**).
  - `kokoro/model.py` — instantiation (lines **54–57**) and
    inference forward pass (`forward_with_tokens`, lines **86–119**).
- Header comment in modules.py: `# https://github.com/yl4579/StyleTTS2/blob/main/models.py`
  — i.e. the architecture is inherited unchanged from StyleTTS2 (Li et al. 2023).

Local install path on this dev machine:
`/opt/homebrew/lib/python3.11/site-packages/kokoro/{modules.py,istftnet.py,model.py}`.

## Instantiation (from `kokoro/model.py`, lines 54–57)

```python
self.predictor = ProsodyPredictor(
    style_dim = config['style_dim'],   # 128
    d_hid     = config['hidden_dim'],  # 512
    nlayers   = config['n_layer'],     # 3
    max_dur   = config['max_dur'],     # 50
    dropout   = config['dropout'],     # 0.2 (eval-time: identity)
)
```

Config source: `hexgrad/Kokoro-82M/config.json`
(local cache: `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/...`).

## Top-level inputs / outputs (inference)

The predictor is **not** invoked by a single `forward(...)` call in
inference. Instead `KModel.forward_with_tokens` orchestrates four
sub-stages by hand. The reference snippet below is the full predictor
slice extracted verbatim from `model.py:102-115`:

```python
bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())   # [1, T,  768]
d_en     = self.bert_encoder(bert_dur).transpose(-1, -2)             # [1, 512, T]
s        = ref_s[:, 128:]                                            # [1, 128]   (predictor half)
d        = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)  # [1, T, 640]
x, _     = self.predictor.lstm(d)                                    # [1, T, 512]
duration = self.predictor.duration_proj(x)                           # [1, T, 50]
duration = torch.sigmoid(duration).sum(axis=-1) / speed              # [1, T]
pred_dur = torch.round(duration).clamp(min=1).long().squeeze()       # [T] int64
indices  = torch.repeat_interleave(torch.arange(T, device=...), pred_dur)
pred_aln_trg          = torch.zeros((T, T_mel))
pred_aln_trg[indices, torch.arange(T_mel)] = 1                       # [T, T_mel]
en       = d.transpose(-1, -2) @ pred_aln_trg                        # [1, 640, T_mel]
F0_pred, N_pred = self.predictor.F0Ntrain(en, s)                     # [1, T_mel], [1, T_mel]
```

So the predictor produces three outputs that flow downstream:

| Output    | Shape          | Consumer                       |
|-----------|----------------|--------------------------------|
| pred_dur  | `int64[T]`     | alignment matrix construction  |
| F0_pred   | `f32[1, T_mel]`| `decoder(asr, F0_pred, N, s)`  |
| N_pred    | `f32[1, T_mel]`| `decoder(asr, F0_pred, N, s)`  |

with `T_mel = sum(pred_dur)` and `T = len(input_ids) = phoneme count + 2`
(KModel pads with two `[0]` tokens, see `model.py:131`).

### Style vector handling

`ref_s` is `voices.bin[voice][T_idx]` of shape `[1, 256]`
(see `kokoro_reference.py` and Plan 2 Task 5). Kokoro splits it:

- `ref_s[:, :128]` → decoder style (passed to `decoder(...)` only).
- `ref_s[:, 128:]` → **predictor style `s`** (used everywhere below).

Inside the predictor, `s` is reused without modification by:
- `DurationEncoder` (concatenated as a per-time-step feature in every LSTM input)
- Every `AdaLayerNorm` (in `DurationEncoder`)
- Every `AdaIN1d` inside `AdainResBlk1d` blocks of `F0` and `N`

## Sub-modules

### 1. `DurationEncoder` (modules.py:137-176) — produces `d`

Constructor:
```python
DurationEncoder(sty_dim=128, d_model=512, nlayers=3, dropout=0.2)
```

Builds an alternating stack of:
- 3× `nn.LSTM(input=512+128=640, hidden=256, num_layers=1, bidirectional=True)`
- 3× `AdaLayerNorm(style_dim=128, channels=512)`

(So `self.lstms` has length 6: indices 0,2,4 are LSTMs, 1,3,5 are AdaLayerNorms.)

Forward (B=1 unpadded view):
1. Input `x: [1, 512, T]` (the `d_en` from `bert_encoder` transposed to channel-first).
2. Permute to `[T, 1, 512]`, broadcast-cat style → `[T, 1, 640]`. Then transpose back
   to `[1, T, 640]`, then to `[1, 640, T]`.
3. For each block in `self.lstms`:
   - **LSTM (layers 0/2/4)**: transpose to `[1, T, 640]`, run BiLSTM → `[1, T, 512]`,
     dropout (eval=identity), transpose back to `[1, 512, T]`.
   - **AdaLayerNorm (layers 1/3/5)**: input is `[1, 512, T]`. Run AdaLayerNorm with
     style `s` → `[1, 512, T]`. Re-concat style → `[1, 640, T]`.
4. Final transpose to `[1, T, 640]`.

Output: `d: [1, T, 640]` (style is appended on the way out — note this is *with* style
because the loop ended on an AdaLayerNorm step that re-concatenated style).

### 2. `lstm` + `duration_proj` — produces `pred_dur`

```python
self.lstm          = nn.LSTM(input=640, hidden=256, bidirectional=True)
self.duration_proj = LinearNorm(in=512, out=50)   # max_dur=50
```

Forward (`model.py:106-109`):
```
x, _      = predictor.lstm(d)                               # [1, T, 512]
duration  = predictor.duration_proj(x)                      # [1, T, 50]
duration  = sigmoid(duration).sum(dim=-1) / speed           # [1, T]
pred_dur  = round(duration).clamp(min=1).long().squeeze()   # [T]
```

The interpretation: `duration_proj` outputs a 50-dim vector per phoneme;
each entry `i` is sigmoid-pressed and summed → expected duration in mel
frames. `clamp(min=1)` guarantees every phoneme gets at least one frame.

### 3. Alignment matrix → `en`

```
indices       = torch.repeat_interleave(torch.arange(T), pred_dur)
pred_aln_trg  = zeros[T, T_mel]
pred_aln_trg[indices, arange(T_mel)] = 1
en            = d.transpose(-1, -2) @ pred_aln_trg          # [1, 640, T_mel]
```

`pred_aln_trg` is a 0/1 alignment matrix; right-multiplying `d^T` by it
"upsamples" each phoneme's 640-dim feature into `pred_dur[i]` consecutive
mel-frame positions.

### 4. `F0Ntrain` (modules.py:124-134) — produces `F0_pred`, `N_pred`

```python
self.shared  = nn.LSTM(input=640, hidden=256, bidirectional=True)   # post-alignment LSTM
self.F0      = ModuleList([                                          # 3 blocks
    AdainResBlk1d(512, 512, style_dim=128, dropout_p=0.2),
    AdainResBlk1d(512, 256, style_dim=128, upsample='nearest', dropout_p=0.2),
    AdainResBlk1d(256, 256, style_dim=128, dropout_p=0.2),
])
self.N       = ModuleList([                                          # identical shape to F0
    AdainResBlk1d(512, 512, style_dim=128, dropout_p=0.2),
    AdainResBlk1d(512, 256, style_dim=128, upsample='nearest', dropout_p=0.2),
    AdainResBlk1d(256, 256, style_dim=128, dropout_p=0.2),
])
self.F0_proj = nn.Conv1d(256, 1, 1, 1, 0)
self.N_proj  = nn.Conv1d(256, 1, 1, 1, 0)
```

Forward:
```
x, _ = shared(en.transpose(-1, -2))      # in: [1, T_mel, 640]  → out: [1, T_mel, 512]
x    = x.transpose(-1, -2)               # [1, 512, T_mel]   ← branch point

F0   = x
for block in self.F0:
    F0 = block(F0, s)                    # AdainResBlk1d, see below
F0   = self.F0_proj(F0)                  # [1, 1, 2*T_mel]   (one upsample doubled length)
F0   = F0.squeeze(1)                     # [1, 2*T_mel]

N    = x                                 # restart from same x
for block in self.N:
    N  = block(N, s)
N    = self.N_proj(N)
N    = N.squeeze(1)                      # [1, 2*T_mel]
```

**Important length detail**: block index 1 has `upsample='nearest'`. Its
`UpSample1d` does `F.interpolate(x, scale_factor=2, mode='nearest')` and its
`pool` is a `weight_norm(ConvTranspose1d(d, d, k=3, s=2, groups=d, padding=1, output_padding=1))`
(also stride=2). So the per-mel-frame T axis is **doubled** through `F0Ntrain`.

The final `F0_pred` and `N_pred` therefore have length `2 * T_mel`. The decoder
expects this; see `decoder(asr, F0_pred, N, s)` on `model.py:118`.

## AdaIN / AdaLayerNorm semantics

There are **two** distinct style-conditioned norm flavours used in the predictor.
Both consume the same 128-dim style `s` but produce different normalisations.

### `AdaLayerNorm` (modules.py:72-88) — used in `DurationEncoder`

```python
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels):
        self.fc = nn.Linear(style_dim, channels*2)   # 128 → 1024

    def forward(self, x, s):                         # x: [1, 512, T] (effectively)
        # internal transposes route x to [1, T, 512] for F.layer_norm
        h            = self.fc(s)                    # [1, 1024]
        gamma, beta  = chunk(h, 2, dim=1)            # each [1, 512]
        x_n          = F.layer_norm(x, (512,))       # over the channel dim, no affine
        out          = (1 + gamma) * x_n + beta      # broadcast over T
        return out
```

i.e. plain LayerNorm over the channel dim with **no learned affine of its own**,
then a per-channel `(1+γ(s)) · x_n + β(s)` modulation. The "+ 1" on gamma is
load-bearing — it's the StyleTTS2 convention so that an all-zero `fc` weights
init gives the identity scale.

### `AdaIN1d` (istftnet.py:20-31) — used inside `AdainResBlk1d` (F0 / N)

```python
class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        self.norm = nn.InstanceNorm1d(num_features, affine=True)   # (sic: see comment in src)
        self.fc   = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):                          # x: [1, C, T_mel]
        h           = self.fc(s)                      # [1, 2*C]
        gamma, beta = chunk(h.view(B, 2*C, 1), 2, dim=1)   # each [1, C, 1]
        return (1 + gamma) * self.norm(x) + beta
```

i.e. **InstanceNorm1d** (per-(B,C) normalisation across T) with its own
learnable affine (`.norm.weight`, `.norm.bias`), *then* the same
`(1 + γ(s)) · · + β(s)` modulation as `AdaLayerNorm`.

The source-code comment confirms `affine=True` is intentional (InstanceNorm
with `affine=False` had an old ONNX-export bug that dropped a dim).

### `AdainResBlk1d` (istftnet.py:340-381) — F0/N building block

```
def _residual(x, s):
    x = norm1(x, s)                # AdaIN1d(C_in)
    x = LeakyReLU(0.2)(x)
    x = pool(x)                    # Identity if upsample='none'; else group-ConvTranspose1d(stride=2)
    x = conv1(x)                   # Conv1d(C_in → C_out, k=3, p=1, weight_norm)
    x = norm2(x, s)                # AdaIN1d(C_out)
    x = LeakyReLU(0.2)(x)
    x = conv2(x)                   # Conv1d(C_out → C_out, k=3, p=1, weight_norm)
    return x

def _shortcut(x):
    x = upsample(x)                # nn.Identity if 'none'; else F.interpolate(scale=2, nearest)
    if learned_sc:                 # True iff dim_in != dim_out
        x = conv1x1(x)             # Conv1d(C_in → C_out, k=1, weight_norm, no bias)
    return x

def forward(x, s):
    return (residual(x, s) + shortcut(x)) * rsqrt(2)
```

Block 1 (the only `upsample='nearest'` one in F0/N) doubles T via:
- residual path: `pool` = ConvTranspose1d(stride=2) doubles T
- shortcut path: `upsample` = F.interpolate(scale=2, nearest) doubles T

So both branches end at `[1, C_out, 2T]` and sum.

## Layer-by-layer (concrete dims for v0_19, B=1)

| # | Layer | Op | Param shapes | Notes |
|---|-------|----|--------------|-------|
| 0 | Input `d_en` | `f32[1, 512, T]` | — | from `bert_encoder` |
| 0 | Input `s` | `f32[1, 128]` | — | from `voices.bin[voice][T_idx, 0, 128:]` |
| 1 | DurEnc cat-style | concat | — | `[1, 640, T]` |
| 2 | DurEnc LSTM 0 | BiLSTM(640→512) | `weight_ih_l0:(1024,640)`, `weight_hh_l0:(1024,256)`, biases, `_reverse` | `predictor_text_encoder_lstms_0_*` |
| 3 | DurEnc AdaLN 1 | AdaLayerNorm(128, 512) | `fc.weight:(1024,128)`, `fc.bias:(1024)` | `predictor_text_encoder_lstms_1_fc_*`. Re-cat style after → `[1, 640, T]` |
| 4 | DurEnc LSTM 2 | BiLSTM(640→512) | same shapes as layer 2 | `predictor_text_encoder_lstms_2_*` |
| 5 | DurEnc AdaLN 3 | AdaLayerNorm(128, 512) | same as layer 3 | `predictor_text_encoder_lstms_3_fc_*` |
| 6 | DurEnc LSTM 4 | BiLSTM(640→512) | same shapes as layer 2 | `predictor_text_encoder_lstms_4_*` |
| 7 | DurEnc AdaLN 5 | AdaLayerNorm(128, 512) | same as layer 3 | `predictor_text_encoder_lstms_5_fc_*`. Output `d:[1, T, 640]` |
| 8 | top LSTM | BiLSTM(640→512) | `weight_ih_l0:(1024,640)`, etc. | `predictor_lstm_*`. Output `[1, T, 512]` |
| 9 | duration_proj | `LinearNorm(512→50)` | `linear_layer.weight:(50,512)`, `bias:(50)` | `predictor_duration_proj_linear_layer_*` |
|10 | sigmoid+sum+round | — | — | `pred_dur:[T] int64`, `T_mel = sum(pred_dur)` |
|11 | aln matrix build | scatter | — | `[T, T_mel]` |
|12 | en | matmul | — | `d.T @ aln = [1, 640, T_mel]` |
|13 | shared LSTM | BiLSTM(640→512) | same shapes as layer 2 | `predictor_shared_*`. Output `[1, T_mel, 512]` (then transposed) |
|14 | F0 block 0 | AdainResBlk1d(512→512) | conv1/2 weight_norm, norm1/2 AdaIN1d | `predictor_F0_0_*` |
|15 | F0 block 1 | AdainResBlk1d(512→256, upsample) | + `pool` ConvTranspose1d (groups=512), `conv1x1` (256←512) | `predictor_F0_1_*` (note `_pool_*` and `_conv1x1_*`). Doubles T |
|16 | F0 block 2 | AdainResBlk1d(256→256) | conv1/2 (256), norms (256) | `predictor_F0_2_*` |
|17 | F0_proj | Conv1d(256→1, k=1) | `weight:(1,256,1)`, `bias:(1)` | `predictor_F0_proj_*`. Output `[1, 1, 2*T_mel]` → squeeze → `[1, 2*T_mel]` |
|18 | N branch | mirror of layers 14–17 | identical shapes | `predictor_N_*`. Output `[1, 2*T_mel]` |

Total trainable tensors: 122 raw → **106 after weight-norm fold** (16 weight_norm
pairs collapsed: 2 per AdainResBlk1d × 6 blocks = 12, plus the 2 pool layers and
the 2 conv1x1 in the upsample blocks). Counts match `assets/kokoro/weights/predictor_*.weights`.

## Forward pseudocode (top-to-bottom, B=1, no padding)

```
# Inputs
d_en : [1, 512, T]      # from bert_encoder.transpose(-1,-2)
s    : [1, 128]         # from voices.bin[voice][T_idx][0, 128:]

# (1) Duration encoder
x = cat([d_en, broadcast(s, T)], dim=1)           # [1, 640, T]
for k in 0..2:
    x = bilstm(x.transpose(-1,-2),                # [1, T, 640]
               lstms[2k]).transpose(-1,-2)         # [1, 512, T]
    x = ada_layer_norm(x, s, lstms[2k+1])         # [1, 512, T]
    x = cat([x, broadcast(s, T)], dim=1)          # [1, 640, T]
d  = x.transpose(-1,-2)                            # [1, T, 640]

# (2) Duration prediction
h, _      = bilstm(d, predictor.lstm)              # [1, T, 512]
duration  = duration_proj(h)                       # [1, T, 50]
pred_dur  = round(sigmoid(duration).sum(-1)/speed) # [T]
pred_dur  = clip(pred_dur, min=1).long()
T_mel     = sum(pred_dur)

# (3) Alignment upsample
aln              = zeros[T, T_mel]                 # 0/1 expansion matrix
aln[repeat_interleave(arange(T), pred_dur),
    arange(T_mel)] = 1
en               = d.transpose(-1,-2) @ aln        # [1, 640, T_mel]

# (4) F0 / N branches
x, _ = bilstm(en.transpose(-1,-2),                 # [1, T_mel, 640]
              predictor.shared)                    # [1, T_mel, 512]
x    = x.transpose(-1,-2)                          # [1, 512, T_mel]

F0 = x
for blk in predictor.F0:                           # 3 AdainResBlk1d
    F0 = blk(F0, s)                                # last block's T == 2*T_mel
F0 = F0_proj(F0).squeeze(1)                        # [1, 2*T_mel]

N  = x
for blk in predictor.N:
    N = blk(N, s)
N  = N_proj(N).squeeze(1)                          # [1, 2*T_mel]

return pred_dur, F0, N
```

## Notable details for the C++ port

1. **Style is split, not the whole `ref_s`**. Predictor sees only the upper
   128-dim half (`ref_s[:, 128:]`); decoder sees only the lower half
   (`ref_s[:, :128]`). `voices.bin` has shape `(511, 1, 256)` per voice
   (one slice per phoneme-count bucket); inference uses the slice indexed
   by `T - 1` (cf. kokoro-onnx).
2. **DurationEncoder appends style to LSTM inputs but resets and re-appends
   it after every AdaLayerNorm step**. Don't try to "share" the cat across
   the loop — the source explicitly re-cats.
3. **Two distinct norm flavours**:
   - `AdaLayerNorm` (in DurationEncoder) is **LayerNorm over channels, no
     affine** + `(1+γ(s))·x + β(s)`.
   - `AdaIN1d` (in F0/N) is **InstanceNorm1d with `affine=True`** + the same
     `(1+γ(s))·x + β(s)` post-modulation. *The InstanceNorm's own affine
     params (`norm.weight`, `norm.bias`) are real trained parameters that
     must be loaded.* See the source comment about an ONNX-export bug.
4. **F0 block 1 doubles T** via two parallel stride-2 paths:
   - residual: `pool` = depthwise `ConvTranspose1d(stride=2, output_padding=1)`
   - shortcut: `F.interpolate(scale_factor=2, mode='nearest')`
5. **Final F0/N length = 2 * T_mel**, not `T_mel`. The decoder consumes this.
6. **All weight_norm Conv1d layers are folded** by the converter — the cactus
   runtime sees plain `Conv1d` kernels.
7. **Dropout is OFF** in eval/inference (we always run `eval()`).
8. **LSTM gate order**: PyTorch packs `[i, f, g, o]`. The Kokoro converter
   uses `repack_lstm_*` so the cactus runtime sees the same ordering.
9. **`pred_dur.clamp(min=1)`**: the clamp is part of the contract; without it
   a phoneme could get 0 frames and the alignment matrix would scatter to
   nowhere. Implement the clamp before the cumulative-sum.
10. **`AdaLayerNorm` LayerNorm has no learned affine**: the call is
    `F.layer_norm(x, (channels,), eps=1e-5)` with no weight/bias args.
    All channel-wise modulation comes from `(1 + gamma(s)) * x + beta(s)`.
