# Kokoro `decoder` — Reference Architecture (ISTFTNet / StyleTTS2 Hi-Fi-GAN)

This is the canonical PyTorch architecture of the Kokoro-82M `decoder`
submodule, extracted from the upstream reference implementation. It is
the spec the cactus C++ port (Plan 2 Task 8b) must reproduce
bit-for-bit (modulo float rounding and PRNG choices — see
"Determinism" below).

## Reference source

- Package: `kokoro` (PyPI, v0.9.4), maintained by hexgrad.
- Files:
  - `kokoro/istftnet.py` — `Decoder` (lines **384–421**),
    `Generator` (lines **257–325**),
    `SineGen` (lines **108–209**),
    `SourceModuleHnNSF` (lines **212–254**),
    `AdainResBlk1d` (lines **340–381**),
    `AdaIN1d` (lines **20–31**),
    `AdaINResBlock1` (lines **34–77**),
    `TorchSTFT` (lines **80–105**).
  - `kokoro/custom_stft.py` — `CustomSTFT` (full file). Used when
    `disable_complex=True` (the path the cactus port mirrors via
    `cactus_stft` / `cactus_istft`).
  - `kokoro/model.py` — instantiation (lines **62–65**) and
    inference forward pass (`forward_with_tokens`, lines **86–119**;
    decoder call is at line **118**).
- Header comment: `# ADAPTED from https://github.com/yl4579/StyleTTS2/
  blob/main/Modules/istftnet.py`. The architecture is inherited
  unchanged from StyleTTS2 (Li et al. 2023), which itself adapts
  Hi-Fi-GAN (Kong et al. 2020) with an iSTFT vocoder head from
  iSTFTNet (Kaneko et al. 2022) and an HnNSF harmonic source from
  the NSF family (Wang et al. 2019).

Local install path on this dev machine:
`/opt/homebrew/lib/python3.11/site-packages/kokoro/{istftnet.py,custom_stft.py,model.py}`.

## Instantiation (from `kokoro/model.py`, lines 62–65)

```python
self.decoder = Decoder(
    dim_in            = config['hidden_dim'],   # 512
    style_dim         = config['style_dim'],    # 128
    dim_out           = config['n_mels'],       # 80  (unused at inference)
    disable_complex   = disable_complex,
    **config['istftnet'],
)
```

`config['istftnet']` (from `hexgrad/Kokoro-82M/config.json`):

```json
{
  "gen_istft_n_fft":           20,
  "gen_istft_hop_size":         5,
  "resblock_kernel_sizes":     [3, 7, 11],
  "resblock_dilation_sizes":  [[1,3,5],[1,3,5],[1,3,5]],
  "upsample_initial_channel":  512,
  "upsample_kernel_sizes":     [20, 12],
  "upsample_rates":            [10,  6]
}
```

So inside the generator: `num_upsamples = 2`, `num_kernels = 3`,
total upsample factor `prod(upsample_rates) * gen_istft_hop_size =
10 * 6 * 5 = 300` audio samples per input "frame" of the
post-upsampled feature.

## Top-level inputs / outputs (inference)

Forward signature (`Decoder.forward`, istftnet.py lines 407–421):

```python
audio = decoder(asr, F0_curve, N, s)
```

| name       | shape                | dtype     | source                                                    |
|------------|----------------------|-----------|-----------------------------------------------------------|
| `asr`      | `(1, 512, T_mel)`    | float32   | `t_en @ pred_aln_trg` (text_encoder output, aligned).      |
| `F0_curve` | `(1, 2*T_mel)`       | float32   | `predictor.F0_proj(...).squeeze(1)` (already 2× upsampled).|
| `N`        | `(1, 2*T_mel)`       | float32   | `predictor.N_proj(...).squeeze(1)`.                        |
| `s`        | `(1, 128)`           | float32   | `ref_s[:, :128]` — note: the **first** half of the voice  |
|            |                      |           | embedding (the predictor uses the **second** half).        |
| → `audio`  | `(1, 1, audio_len)`  | float32   | 24kHz mono PCM. `audio_len = 600 * T_mel` samples (see    |
|            |                      |           | "Audio length" derivation below).                          |

`T_mel = sum(pred_dur)` from the predictor — the number of mel-rate
frames covering the utterance, ~80 frames/second nominal (24kHz / 300).

`audio_len` derivation: F0 enters at `2*T_mel` samples. `f0_upsamp` is
`nn.Upsample(scale_factor=300)` (`upsample_scale = prod([10,6])*5`),
so the harmonic source runs at `2*T_mel * 300 / 2 = 300*T_mel`?
**No** — `f0_upsamp` is applied to `F0_curve[:, None]` of shape
`(B, 1, 2*T_mel)`, giving `(B, 1, 600*T_mel)`. Then iSTFT collapses
back to `600*T_mel - 0` audio samples (the iSTFT window is built so
the output length matches the input the m_source produced via
`length=x.shape[-1]` — see `Generator.forward` line 325 / `TorchSTFT.
inverse`). Net: **`audio_len = 600 * T_mel` samples = 25 ms/frame ×
T_mel frames at 24 kHz**, i.e. ~12.5 ms/phoneme on average for the
KModel default `speed=1`.

For the reference phrase **p01** ("Hello, my name is Cactus.") with
the misaki phonemizer, T_mel = 112 → `audio_len = 67 200` samples =
**2.80 s** of 24 kHz audio. (kokoro-onnx, which uses espeak-ng, gets
T_mel = 64 → 1.60 s. Same model, different phonemizer.)

## Tensor inventory (post-fold)

After Plan 2 Task 5's converter folds `weight_norm` (`weight_g`,
`weight_v` → fused `weight`) and drops InstanceNorm `affine` no-op
params, the decoder produces **305 tensors** organized as:

| group          | tensors | notes                                                               |
|----------------|---------|---------------------------------------------------------------------|
| `encode/`      |       6 | one AdainResBlk1d (no upsample, learned shortcut).                  |
| `decode.0/`    |       8 | AdainResBlk1d (no upsample).                                        |
| `decode.1/`    |       8 | AdainResBlk1d (no upsample).                                        |
| `decode.2/`    |       8 | AdainResBlk1d (no upsample).                                        |
| `decode.3/`    |      10 | AdainResBlk1d **with upsample** (extra `pool` ConvT1d, groups=1090).|
| `F0_conv/`     |       2 | weight-normed Conv1d, groups=1, kernel=3, stride=2.                 |
| `N_conv/`      |       2 | weight-normed Conv1d, groups=1, kernel=3, stride=2.                 |
| `asr_res/`     |       2 | weight-normed Conv1d 1×1, 512 → 64.                                 |
| `generator/`   |     ~259| see breakdown below.                                                |

Generator breakdown (counting post-fold):

| sub-module                     | tensors | shape highlights                                |
|--------------------------------|---------|--------------------------------------------------|
| `m_source.l_linear/`           |       2 | `weight: (1, 9)`, `bias: (1,)` — see SourceModule |
| `noise_convs.0/` (plain conv1d)|       2 | `(256, 22, 12)` weight (no weight_norm here).    |
| `noise_convs.1/`               |       2 | `(128, 22,  1)`.                                 |
| `noise_res.0/` AdaINResBlock1  |      30 | 3×{conv1, conv2}+3×{adain1, adain2}+2×3 alpha. ch=256, k=7 |
| `noise_res.1/` AdaINResBlock1  |      30 | ch=128, k=11                                     |
| `ups.0/` ConvT1d (weight-normed)|      2 | `(512→256, k=20, s=10)` (weight stored as 512×256×20). |
| `ups.1/` ConvT1d                |      2 | `(256→128, k=12, s=6)`.                          |
| `resblocks.0..2/` (3 kernels)  |    3×30 | After ups.0, channels=256. Kernels {3,7,11}, all dilation {1,3,5}. |
| `resblocks.3..5/`              |    3×30 | After ups.1, channels=128. Same kernel/dilation set. |
| `conv_post/` weight-normed     |       2 | Conv1d 128 → 22 (= n_fft/2+1 mag + n_fft/2+1 phase = 11+11), kernel=7. |

(The exact `weight_norm`-folded parameter count totals to 305 once you
sum these. The pre-fold checkpoint had 375 tensors due to `weight_g` /
`weight_v` pairs; see `assets/kokoro/converter_subset_inspection.txt`
for the equivalent text_encoder + bert audit.)

## Snake1d (per-channel learnable)

The exact code (istftnet.py line 71, repeated at 74):

```python
xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
```

where `a1` (`alpha1[i]`) has shape `(1, channels, 1)` and is broadcast
over batch and time. `a2` (`alpha2[i]`) is the symmetric activation
inside the same residual sub-block. **It is only used inside
`AdaINResBlock1`** (the Hi-Fi-GAN-style ResBlock used in the
`generator.noise_res` and `generator.resblocks`). The decoder's
`encode` and `decode.*` blocks (`AdainResBlk1d`, distinct class!)
use plain `nn.LeakyReLU(0.2)` instead. Don't confuse the two; both
classes exist in `istftnet.py` with confusingly similar names.

## Harmonic source module (`SourceModuleHnNSF`)

Operates on `F0_curve` (the predictor's raw F0 prediction). Code:
istftnet.py lines 212–254 (module) and 108–209 (`SineGen`).

Pipeline (no PyTorch parameters apart from `l_linear`):

1. **Upsample F0 to audio rate.** `f0 = self.f0_upsamp(F0_curve[:, None])
   .transpose(1, 2)` → shape `(1, 600*T_mel, 1)` after upsample by
   `prod(upsample_rates) * gen_istft_hop_size = 300` and the extra
   factor of 2 already present in `F0_curve` (it's at `2*T_mel`).
2. **Generate harmonic comb.** `SineGen.forward(f0)` returns
   `sine_waves` of shape `(1, 600*T_mel, 9)` — **9 harmonics** =
   fundamental (`harmonic_num=8` + 1). Each harmonic `h` is
   `sin(2π · cumsum(((h+1)·f0 / 24000) % 1))`, voicing-masked so
   `f0 ≤ voiced_threshold (=10 Hz)` produces noise instead of a sine.
   - `_f02sine` (lines 142–183) does the cumulative phase. For
     numerical stability with 600x upsampling, it downsamples
     `rad_values` by `1/upsample_scale` *before* `cumsum`, then
     re-upsamples the cumulative phase by `upsample_scale` (lines
     155–157). The cactus port can either replicate this trick or
     accumulate at audio rate directly (slightly different rounding;
     test against fixture).
   - **PRNG dependency:** line 150 calls `torch.rand` for an initial
     phase offset (`rand_ini`) which is then zeroed for the
     fundamental (`rand_ini[:, 0] = 0`), and line 205 calls
     `torch.randn_like(sine_waves)` for additive Gaussian noise.
     For deterministic fixtures we MUST seed the RNG before each
     forward — see "Determinism" below.
3. **Merge harmonics.** `sine_merge = tanh(l_linear(sine_waves))` —
   linear `9 → 1` (the `(1, 9)` weight + `(1,)` bias is the only
   trainable parameter in the harmonic source). Output shape
   `(1, 600*T_mel, 1)`. **This is `har_source` (after
   `transpose(1,2).squeeze(1)` in Generator: `(1, 600*T_mel)`).**
4. **Compute STFT of harmonic source.** `stft.transform(har_source)`
   → magnitude + phase, each shape `(1, n_fft/2 + 1, frames) =
   (1, 11, 120*T_mel)`. Concatenated along channel → `har: (1, 22,
   120*T_mel)`. This is the conditioning signal injected into each
   upsample block via `noise_convs[i]`.

The "noise" in the SourceModule (line 253: `noise = torch.randn_like(uv)
* sine_amp / 3`) is computed but **not used downstream** — only
`sine_merge` is consumed by the generator (it goes into `har_source`
above). That dead-code branch is a StyleTTS2 inheritance.

## Top-level forward (`Decoder.forward`, istftnet.py lines 407–421)

```python
def forward(self, asr, F0_curve, N, s):                     # asr: (1,512,T_mel)
    F0 = self.F0_conv(F0_curve.unsqueeze(1))                # (1, 1, T_mel)
    N  = self.N_conv (N.unsqueeze(1))                       # (1, 1, T_mel)
    x = torch.cat([asr, F0, N], axis=1)                     # (1, 514, T_mel)
    x = self.encode(x, s)                                   # (1, 1024, T_mel)
    asr_res = self.asr_res(asr)                             # (1, 64, T_mel)
    res = True
    for block in self.decode:                               # 4 AdainResBlk1d
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)      # +64+1+1 = +66 ch
        x = block(x, s)
        if block.upsample_type != "none":                   # only decode.3 upsamples
            res = False                                     # → after decode.3, no more cat
    x = self.generator(x, s, F0_curve)                      # → audio (1, 1, 600*T_mel)
    return x
```

`encode` is `AdainResBlk1d(dim_in + 2, 1024, style_dim) =
AdainResBlk1d(514, 1024, 128)`. `decode.{0,1,2}` are
`AdainResBlk1d(1090, 1024, 128)` (no upsample); `decode.3` is
`AdainResBlk1d(1090, 512, 128, upsample=True)` (depth-wise ConvT1d
`pool` doubles T_mel dimension).

The `res = True/False` flag gates re-concatenation of `asr_res, F0, N`
**only at the input of each decode block until the first upsample
fires**. Since `decode.3` is the only block with `upsample=True`, the
sequence is:

1. `x: (1, 1024, T_mel)` ← encode output.
2. cat→ `(1, 1090, T_mel)`, decode.0 → `(1, 1024, T_mel)`.
3. cat→ `(1, 1090, T_mel)`, decode.1 → `(1, 1024, T_mel)`.
4. cat→ `(1, 1090, T_mel)`, decode.2 → `(1, 1024, T_mel)`.
5. cat→ `(1, 1090, T_mel)`, decode.3 (upsample) → `(1, 512, 2*T_mel)`.
6. (no more cat; res = False) → into generator.

So the generator receives `x: (1, 512, 2*T_mel)` and produces audio at
`600*T_mel` samples (a further 300× temporal upsample inside the
generator).

## Generator forward (istftnet.py lines 299–325)

```python
def forward(self, x, s, f0):
    # ---- Harmonic source (no autograd; no parameters in upsample) ----
    f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)        # (1, 600*T_mel, 1)
    har_source, _, _ = self.m_source(f0)                    # (1, 600*T_mel, 1)
    har_source = har_source.transpose(1, 2).squeeze(1)      # (1, 600*T_mel)
    har_spec, har_phase = self.stft.transform(har_source)   # (1, 11, frames)
    har = torch.cat([har_spec, har_phase], dim=1)           # (1, 22, frames)

    # ---- Upsample loop ----
    for i in range(self.num_upsamples):                     # 2 iters
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = self.noise_convs[i](har)                 # (1, ch_i, T_i)
        x_source = self.noise_res[i](x_source, s)           # (1, ch_i, T_i)
        x = self.ups[i](x)                                  # ConvT1d: (1, ch_i, T_i)
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)                      # ReflectionPad1d((1,0))
        x = x + x_source                                    # MRF input

        # MRF (multi-receptive-field) sum across 3 kernels.
        xs = None
        for j in range(self.num_kernels):                   # 3 iters
            block_out = self.resblocks[i*3 + j](x, s)       # AdaINResBlock1
            xs = block_out if xs is None else xs + block_out
        x = xs / self.num_kernels                           # (1, ch_i, T_i)

    # ---- iSTFT vocoder head ----
    x = F.leaky_relu(x)                                     # default 0.01 slope
    x = self.conv_post(x)                                   # (1, 22, frames)
    spec  = torch.exp(x[:, : self.post_n_fft//2 + 1, :])    # (1, 11, frames) magnitude
    phase = torch.sin(x[:,    self.post_n_fft//2 + 1:, :])  # (1, 11, frames) phase (in radians, but applied directly)
    return self.stft.inverse(spec, phase)                   # (1, 1, audio_len)
```

Key shape pegs (with `T_mel = 112` for p01, verified by the fixture
script `kokoro_decoder_reference.py`):

| stage                 | shape                | T-axis size for p01 |
|-----------------------|----------------------|---------------------|
| Generator input `x`   | `(1, 512, 2*T_mel)`  | 224                 |
| `f0` after upsamp     | `(1, 600*T_mel, 1)`  | 67200               |
| `har_source`          | `(1, 600*T_mel)`     | 67200               |
| `har` after STFT      | `(1, 22, frames)`    | 13441               |
| After `ups.0` (pre-resblock) | `(1, 256, 20*T_mel)` | 2240         |
| After resblocks[0..2] | `(1, 256, 20*T_mel)` | 2240                |
| After `ups.1` (pre-pad) | `(1, 128, 120*T_mel)` | 13440           |
| After reflection_pad  | `(1, 128, 120*T_mel + 1)` | 13441         |
| After resblocks[3..5] | `(1, 128, 120*T_mel + 1)` | 13441         |
| After `conv_post`     | `(1,  22, 120*T_mel + 1)` | 13441         |
| spec, phase           | each `(1, 11, 120*T_mel + 1)` | 13441    |
| After iSTFT (audio)   | `(1,   1, 600*T_mel)` | 67200             |

STFT frame count for `har` (with `center=True`, pad_len = n_fft/2 = 10,
hop = 5): `(600*T_mel + 2*10)/5 + 1 = 120*T_mel + 5`. Wait — that's
13445, not 13441. The discrepancy: PyTorch's iSTFT length crop
(`waveform[..., pad_len:-pad_len]`) of CustomSTFT (custom_stft.py
181-182) is what brings the audio back to exactly 600*T_mel. The
13441 figure for the "frames" axis after `conv_post` is the
post-`reflection_pad((1,0))` count (one extra frame appended to
13440). That count must match the STFT-frame count of `har` so
`x + x_source` aligns at the second upsample step — and indeed
`conv1d(har, kernel=stride_f0_2*2, stride=stride_f0_2)` with
`stride_f0_2 = 1` and `padding=1` (see Generator.__init__ line 286 —
this is `noise_convs[1]`, kernel 1, stride 1) preserves the length.
The matching arithmetic:

- `har_source` length = 600*T_mel = 67200.
- CustomSTFT pads to 67200 + 20 = 67220, then strides by 5:
  `(67220 - 20) / 5 + 1 = 13441` frames.

So `noise_convs[1]` outputs `(1, 128, 13441)` directly, and the
`reflection_pad((1, 0))` after `ups.1` (which would otherwise give
13440) brings the upsampled-x to 13441 to match. Confirmed.

## AdaIN1d, AdainResBlk1d, AdaINResBlock1 — same as predictor

These are the same classes used by the predictor's F0/N branches
(documented in `predictor_architecture.md`). Quick recap:

- **AdaIN1d** (lines 20–31): `InstanceNorm1d(num_features,
  affine=True)` whose affine params (1.0 / 0.0) are no-ops, then
  `style → linear(128 → 2C)` → split into γ, β →
  `(1 + γ) * normed + β`.
- **AdainResBlk1d** (lines 340–381): two `Conv1d(C_in→C_out, k=3,
  pad=1)` interleaved with AdaIN1d and `LeakyReLU(0.2)`. Optional
  upsample uses a depthwise `ConvTranspose1d(C_in, C_in, k=3, s=2,
  groups=C_in, output_padding=1)` (`pool`). Residual is scaled by
  `1/√2`. Output: `(residual + shortcut) * rsqrt(2)`.
- **AdaINResBlock1** (lines 34–77): three parallel residual paths
  with dilations `(1, 3, 5)` (the second conv in each path uses
  dilation 1). Snake1d activation `(x + (1/α)·sin²(αx))` per
  channel.

The decoder's `encode` and `decode.*` blocks use **AdainResBlk1d**
(LeakyReLU). The generator's `noise_res.*` and `resblocks.*` use
**AdaINResBlock1** (Snake1d). Same AdaIN1d under the hood.

## STFT / iSTFT

Two paths, controlled by `disable_complex`:

| `disable_complex` | STFT module                              | filter_length | hop_length |
|-------------------|-------------------------------------------|---------------|------------|
| `False` (default) | `TorchSTFT` (uses `torch.stft`/`istft`)   | 20            | 5          |
| `True`            | `CustomSTFT` (uses `Conv1d`/`ConvT1d`)    | 20            | 5          |

The cactus port uses `cactus_stft` / `cactus_istft` (Plan 1) which
mirror the **CustomSTFT** code path: real conv + window-sum
overlap-add. This is the path the C++ port (Task 8b) must validate
against. For the reference fixtures we set `disable_complex=True`
so the recorded `stft_real`, `stft_imag`, and `audio` tensors
match the path the port will reproduce.

Note: CustomSTFT is **not** numerically identical to TorchSTFT in
edge cases:
- **Padding mode**: CustomSTFT uses `replicate` (or `constant`),
  TorchSTFT uses `reflect`. With our 67 200-sample input and a
  10-sample pad on each side, this affects only the very first /
  last STFT frame.
- **`atan2` correction** at line 137 of custom_stft.py to handle the
  `imag==0, real<0` case (TorchSTFT returns +π via complex angle).

For the cactus port, the operative spec is **CustomSTFT**.

## Determinism

Two PRNG sites in `SineGen` (lines 150 and 205) make the
harmonic-source output non-deterministic by default:

1. `rand_ini = torch.rand(B, dim, device=...)` — initial phase
   offsets for harmonics 2..9 (the fundamental is masked to 0).
2. `noise = noise_amp * torch.randn_like(sine_waves)` — additive
   white noise, scaled by `sine_amp/3` (≈ 0.033) in voiced regions
   and `sine_amp` (≈ 0.1) in unvoiced regions.

To make fixtures reproducible the reference script calls
`torch.manual_seed(<deterministic seed>)` immediately before each
phrase's forward pass. The C++ port should accept the recorded
`har_source` as ground truth (or seed its PRNG identically — but
matching the Mersenne-Twister state across PyTorch and our C++
RNG is not worth it). **Recommendation:** the cactus port should
take an optional pre-generated `har_source` input for golden tests
(matching the fixture), and use a `std::mt19937` (or any fast PRNG)
for production inference. Audible differences from PRNG choice are
imperceptible — the voiced-region noise is at -29 dB relative to
the harmonic comb.

## Cross-check vs kokoro-onnx Task 2 reference

The Task 2 reference audio (`tests/fixtures/data/kokoro_p<NN>_audio.bin`)
was generated by **kokoro-onnx**, which uses **espeak-ng** as its
phonemizer. The PyTorch reference uses **misaki**. These produce
different phoneme strings (e.g. p01: misaki = `həlˈO, mI nˈAm ɪz
kˈæktəs.` len 26; espeak-ng = `həlˈoʊ, maɪ nˈeɪm ɪz kˈæktəs.` len 29),
which leads to different phoneme IDs, different predicted durations,
and different `T_mel`. Therefore the **end-to-end audio buffers do
NOT match across phonemizers** (p01: 67 200 samples PyTorch vs
38 400 samples kokoro-onnx).

This is **not a checkpoint reconstruction error** — it's an
expected pipeline divergence. The decoder fixtures generated by
this script are still the correct ground truth for the C++ port,
because the cactus port also uses misaki phonemes (or, more
precisely, ingests the same predictor outputs that this fixture
script ingests).

Cross-validation strategy used in the script:

1. The script's `asr`, `F0_curve`, `N`, `s` inputs are identical to
   the saved predictor fixtures (same misaki phonemes, same
   pre-validated predictor).
2. After generating per-stage fixtures via forward hooks, the script
   re-runs the official `Decoder.forward` end-to-end with the same
   seeded RNG and asserts max|Δ| < 1e-5 between the hook-recorded
   path and the official forward — proving the per-stage capture is
   the verbatim official reference path.
3. The audio output of step 2 IS the canonical PyTorch reference for
   the cactus port. (It is NOT compared against `kokoro_p<NN>_audio.bin`,
   which is a different pipeline.)
