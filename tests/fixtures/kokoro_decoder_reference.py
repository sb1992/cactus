"""Run the PyTorch reference Kokoro decoder on the 5 reference phrases
and dump per-stage intermediate outputs for cactus C++ port validation.

Plan 2 Task 8a fixture generator. The cactus implementation (Task 8b)
will load the same .weights files and must reproduce these stage-by-
stage.

Reference: kokoro PyPI v0.9.4
  - kokoro/istftnet.py    Decoder (lines 384-421), Generator (257-325),
                          SineGen (108-209), SourceModuleHnNSF (212-254),
                          AdainResBlk1d (340-381), AdaINResBlock1 (34-77).
  - kokoro/custom_stft.py CustomSTFT (full file). Used because cactus
                          mirrors this code path via cactus_stft / cactus_istft.
  - kokoro/model.py       KModel.forward_with_tokens (86-119); decoder call
                          at line 118 uses ref_s[:, :128] (NOT [:, 128:]).

Inputs are constructed from the already-validated predictor + text_encoder
PyTorch references (see kokoro_predictor_reference.py and
kokoro_text_encoder_reference.py). We rebuild `asr` from text_encoder +
predictor's alignment matrix; everything else (F0, N, T, T_mel) we read
from the predictor fixture .bin files.

Determinism: SineGen calls torch.rand and torch.randn_like internally
(istftnet.py lines 150 and 205). We set torch.manual_seed(SEED=42)
immediately before each phrase's forward pass so the recorded harmonic
source is reproducible. See decoder_architecture.md "Determinism" for
the ground-truth contract.

Outputs (in tests/fixtures/data/, gitignored):
  decoder_p<NN>_input_asr.bin              — float32 [T_mel, 512]
  decoder_p<NN>_input_F0.bin               — float32 [2*T_mel]
  decoder_p<NN>_input_N.bin                — float32 [2*T_mel]
  decoder_p<NN>_input_style128.bin         — float32 [128]   (ref_s[:, :128])
  decoder_p<NN>_F0_after_conv.bin          — float32 [T_mel] (after F0_conv)
  decoder_p<NN>_N_after_conv.bin           — float32 [T_mel]
  decoder_p<NN>_after_encode.bin           — float32 [T_mel, 1024]
  decoder_p<NN>_asr_res.bin                — float32 [T_mel, 64]
  decoder_p<NN>_after_decode_block_0.bin   — float32 [T_mel, 1024]
  decoder_p<NN>_after_decode_block_1.bin   — float32 [T_mel, 1024]
  decoder_p<NN>_after_decode_block_2.bin   — float32 [T_mel, 1024]
  decoder_p<NN>_after_decode_block_3.bin   — float32 [2*T_mel, 512]
  decoder_p<NN>_har_source.bin             — float32 [600*T_mel]
  decoder_p<NN>_har_spec.bin               — float32 [11, frames]
  decoder_p<NN>_har_phase.bin              — float32 [11, frames]
  decoder_p<NN>_after_ups0.bin             — float32 [256, 20*T_mel]
  decoder_p<NN>_after_ups1.bin             — float32 [128, ~120*T_mel]
  decoder_p<NN>_after_resblocks0.bin       — float32 [256, 20*T_mel]
  decoder_p<NN>_after_resblocks1.bin       — float32 [128, 120*T_mel]
  decoder_p<NN>_after_conv_post.bin        — float32 [22, 120*T_mel]
  decoder_p<NN>_stft_real.bin              — float32 [11, 120*T_mel]  (= spec * cos(phase))
  decoder_p<NN>_stft_imag.bin              — float32 [11, 120*T_mel]  (= spec * sin(phase))
  decoder_p<NN>_audio.bin                  — float32 [600*T_mel]      (canonical PyTorch reference)

Each .bin layout: [u32 length][f32 ...]    ("length" = element count, not bytes.)

Run:  /opt/homebrew/bin/python3.11 tests/fixtures/kokoro_decoder_reference.py
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "kokoro"
PTH_PATH = ASSETS / "kokoro-v0_19.pth"
VOICES_PATH = ASSETS / "voices.bin"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

REFERENCE_PHRASES = [
    ("p01", "Hello, my name is Cactus."),
    ("p02", "The quick brown fox jumps over the lazy dog."),
    ("p03", "What time is it in Tokyo right now?"),
    ("p04", "I can run entirely on your device, no cloud needed."),
    ("p05", "Speech synthesis is the inverse of speech recognition."),
]

SEED = 42  # Master RNG seed used before every phrase forward.


# ---------------------------------------------------------------------------
# Binary I/O — same layout as kokoro_predictor_reference.py.
# ---------------------------------------------------------------------------

def write_f32(path: Path, arr) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


def read_f32(path: Path, shape=None) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        a = np.frombuffer(f.read(), dtype=np.float32).copy()
    assert a.size == n, f"{path}: header={n}, payload={a.size}"
    if shape is not None:
        a = a.reshape(shape)
    return a


def read_i64(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        a = np.frombuffer(f.read(), dtype=np.int64).copy()
    assert a.size == n
    return a


# ---------------------------------------------------------------------------
# Locate config.json (for vocab) and instantiate Decoder + TextEncoder.
# ---------------------------------------------------------------------------

HF_CONFIG_HINTS = [
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M",
]
config_path = None
for root in HF_CONFIG_HINTS:
    for cfg in root.rglob("config.json"):
        try:
            with open(cfg) as f:
                obj = json.load(f)
            if "vocab" in obj and obj.get("n_token") == 178:
                config_path = cfg
                break
        except Exception:
            continue
    if config_path:
        break
if config_path is None:
    sys.exit(
        "Could not locate Kokoro config.json. "
        "Run `python3.11 -c 'from kokoro import KModel; KModel(repo_id=\"hexgrad/Kokoro-82M\")'` "
        "once to populate the HF cache."
    )

print(f"Building Decoder + TextEncoder skeletons (config={config_path}) ...")
torch.manual_seed(SEED)
with open(config_path) as f:
    cfg = json.load(f)

from kokoro.istftnet import Decoder
from kokoro.modules import TextEncoder

# disable_complex=True → uses CustomSTFT (which mirrors cactus_stft /
# cactus_istft). This is the path the C++ port (Task 8b) reproduces.
decoder = Decoder(
    dim_in=cfg["hidden_dim"], style_dim=cfg["style_dim"],
    dim_out=cfg["n_mels"], disable_complex=True,
    **cfg["istftnet"],
)
text_encoder = TextEncoder(
    channels=cfg["hidden_dim"], kernel_size=cfg["text_encoder_kernel_size"],
    depth=cfg["n_layer"], n_symbols=cfg["n_token"],
)

print(f"Loading weights from {PTH_PATH} ...")
ckpt = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
net = ckpt["net"]
for sub_name, module in (("decoder", decoder), ("text_encoder", text_encoder)):
    sd = net[sub_name]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    # Accepted "missing" categories:
    #   - fused weight_norm `.weight` (we load split _g/_v).
    #   - InstanceNorm1d affine no-op params (`*.norm.weight/bias`).
    #   - generator.stft.* buffers (computed in CustomSTFT.__init__).
    real_missing = [
        k for k in missing
        if not k.endswith(".weight")
        and not k.endswith(".norm.weight")
        and not k.endswith(".norm.bias")
        and not k.startswith("generator.stft.")
    ]
    real_unexpected = [k for k in unexpected if "position_ids" not in k]
    if real_missing or real_unexpected:
        print(f"  [{sub_name}] missing={real_missing}  unexpected={real_unexpected}")
        if real_missing:
            sys.exit(f"unexpected missing tensors in {sub_name}; aborting.")
decoder.eval()
text_encoder.eval()
print("Decoder + TextEncoder loaded OK")


# ---------------------------------------------------------------------------
# Voices.bin (af_bella).
# ---------------------------------------------------------------------------

print(f"Loading voices.bin from {VOICES_PATH} ...")
voices = np.load(str(VOICES_PATH))
voice_emb = np.array(voices["af_bella"], dtype=np.float32)   # [511, 1, 256]
print(f"  voice af_bella: shape={voice_emb.shape}")


# ---------------------------------------------------------------------------
# Forward hooks: capture every major sub-module output.
#
# Hook design notes:
# - Hooks fire on Module.__call__, AFTER forward() returns. They see
#   the post-forward output but cannot intercept the *inside* of a
#   forward (e.g. between F.leaky_relu and ups[i] in Generator). For
#   the few stages that live inside Generator.forward() (har_source,
#   stft_real/imag), we monkeypatch the Generator forward instead.
# - We use named registration so we can read back per-stage tensors
#   in a deterministic order.
# ---------------------------------------------------------------------------

def install_hooks(decoder, text_encoder):
    captures = {}

    def hook(name):
        def fn(_module, _inp, out):
            # AdainResBlk1d / AdaINResBlock1 forward returns a single tensor.
            if isinstance(out, tuple):
                out = out[0]
            captures[name] = out.detach().cpu().numpy().copy()
        return fn

    handles = [
        decoder.F0_conv.register_forward_hook(hook("F0_after_conv")),
        decoder.N_conv.register_forward_hook(hook("N_after_conv")),
        decoder.encode.register_forward_hook(hook("after_encode")),
        decoder.asr_res.register_forward_hook(hook("asr_res")),
        decoder.decode[0].register_forward_hook(hook("after_decode_block_0")),
        decoder.decode[1].register_forward_hook(hook("after_decode_block_1")),
        decoder.decode[2].register_forward_hook(hook("after_decode_block_2")),
        decoder.decode[3].register_forward_hook(hook("after_decode_block_3")),
        # Generator-internal modules:
        decoder.generator.ups[0].register_forward_hook(hook("after_ups0_raw")),
        decoder.generator.ups[1].register_forward_hook(hook("after_ups1_raw")),
        decoder.generator.conv_post.register_forward_hook(hook("after_conv_post")),
    ]
    return captures, handles


# ---------------------------------------------------------------------------
# Monkeypatch Generator.forward so we can capture intermediates that
# live INSIDE the forward — har_source, har_spec/phase, MRF outputs,
# spec/phase fed into iSTFT — without rewriting the model.
# ---------------------------------------------------------------------------

import torch.nn.functional as F

def patched_generator_forward(self, x, s, f0):
    """Verbatim copy of Generator.forward (istftnet.py 299-325) with
    extra capture points written into self._captures.
    """
    cap = self._captures
    with torch.no_grad():
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _noi, _uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        cap["har_source"] = har_source.detach().cpu().numpy().copy()
        har_spec, har_phase = self.stft.transform(har_source)
        cap["har_spec"] = har_spec.detach().cpu().numpy().copy()
        cap["har_phase"] = har_phase.detach().cpu().numpy().copy()
        har = torch.cat([har_spec, har_phase], dim=1)
    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = self.noise_convs[i](har)
        x_source = self.noise_res[i](x_source, s)
        x = self.ups[i](x)
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i * self.num_kernels + j](x, s)
            else:
                xs += self.resblocks[i * self.num_kernels + j](x, s)
        x = xs / self.num_kernels
        cap[f"after_resblocks{i}"] = x.detach().cpu().numpy().copy()
    x = F.leaky_relu(x)
    x = self.conv_post(x)
    spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
    cap["stft_spec"] = spec.detach().cpu().numpy().copy()
    cap["stft_phase"] = phase.detach().cpu().numpy().copy()
    cap["stft_real"] = (spec * torch.cos(phase)).detach().cpu().numpy().copy()
    cap["stft_imag"] = (spec * torch.sin(phase)).detach().cpu().numpy().copy()
    return self.stft.inverse(spec, phase)


# Bind the captures dict onto the generator and patch.
decoder.generator._captures = {}
import types
decoder.generator.forward = types.MethodType(patched_generator_forward, decoder.generator)


# ---------------------------------------------------------------------------
# Per-phrase driver. Reads predictor fixtures, runs decoder, writes outputs.
# ---------------------------------------------------------------------------

def run_and_dump(tag: str) -> None:
    print(f"\n[{tag}]")

    # Read predictor fixtures (already validated against PyTorch reference).
    ids = read_i64(OUT / f"predictor_{tag}_input_phonemes.bin")
    T = ids.size
    input_ids = torch.from_numpy(ids).long().unsqueeze(0)            # (1, T)
    input_lengths = torch.LongTensor([T])
    text_mask = torch.zeros((1, T), dtype=torch.bool)

    # The alignment matrix is stored as [T, T_mel] (column-major in the
    # T_mel axis). Re-shape via the recorded T.
    aln_flat = read_f32(OUT / f"predictor_{tag}_alignment.bin")
    T_mel = aln_flat.size // T
    aln = aln_flat.reshape(T, T_mel)
    pred_aln_b = torch.from_numpy(aln).float().unsqueeze(0)          # (1, T, T_mel)

    F0 = read_f32(OUT / f"predictor_{tag}_F0_pred.bin")              # (2*T_mel,)
    N  = read_f32(OUT / f"predictor_{tag}_N_pred.bin")               # (2*T_mel,)
    F0_t = torch.from_numpy(F0).float().unsqueeze(0)                 # (1, 2*T_mel)
    N_t  = torch.from_numpy(N ).float().unsqueeze(0)

    # Voice slice: kokoro-onnx convention is voices[voice][len(input_ids)-1].
    ref_s = voice_emb[T - 1]                                          # (1, 256)
    s_dec_np = ref_s[:, :128]                                         # decoder uses FIRST half
    s_pre_np = ref_s[:, 128:]                                         # predictor uses SECOND half
    s_dec = torch.from_numpy(s_dec_np.copy()).float()                 # (1, 128)

    # Sanity: predictor fixture's style_s should match s_pre.
    s_pre_fixture = read_f32(OUT / f"predictor_{tag}_style_s.bin", (128,))
    diff_style = float(np.abs(s_pre_fixture - s_pre_np.squeeze(0)).max())
    assert diff_style < 1e-6, f"style_s mismatch with predictor fixture: {diff_style}"

    # Compute asr = text_encoder(input_ids) @ pred_aln_b.
    with torch.no_grad():
        torch.manual_seed(SEED)  # text_encoder has no PRNG ops, but be safe.
        t_en = text_encoder(input_ids, input_lengths, text_mask)     # (1, 512, T)
        asr = t_en @ pred_aln_b                                       # (1, 512, T_mel)

    print(f"  T={T}  T_mel={T_mel}  asr={tuple(asr.shape)}  "
          f"F0={tuple(F0_t.shape)}  N={tuple(N_t.shape)}")

    # Install hooks (fresh per-phrase to keep captures clean).
    captures, handles = install_hooks(decoder, text_encoder)
    decoder.generator._captures = {}

    # Forward — seed RIGHT BEFORE this call so SineGen's torch.rand and
    # torch.randn_like calls draw from a deterministic state.
    torch.manual_seed(SEED)
    with torch.no_grad():
        audio = decoder(asr, F0_t, N_t, s_dec)                        # (1, 1, 600*T_mel)

    for h in handles:
        h.remove()

    # Pull generator-internal captures.
    captures.update(decoder.generator._captures)

    # ---- Cross-check: re-run decoder with the SAME seed; result must match
    # the recorded audio byte-for-byte (or float-EQ). This proves the hooks
    # didn't perturb the forward.
    torch.manual_seed(SEED)
    with torch.no_grad():
        audio_again = decoder(asr, F0_t, N_t, s_dec)
    diff_audio = float((audio - audio_again).abs().max().item())
    if diff_audio > 0:
        # Accept tiny rounding from non-deterministic intra-op order, but flag.
        print(f"  WARN: re-run audio max|Δ|={diff_audio:.2e} (expected 0.0)")

    # ---- Write input + per-stage fixtures ----------------------------------
    write_f32(OUT / f"decoder_{tag}_input_asr.bin",
              asr[0].transpose(0, 1).cpu().numpy())                   # [T_mel, 512]
    write_f32(OUT / f"decoder_{tag}_input_F0.bin", F0)                # [2*T_mel]
    write_f32(OUT / f"decoder_{tag}_input_N.bin",  N)                 # [2*T_mel]
    write_f32(OUT / f"decoder_{tag}_input_style128.bin",
              s_dec_np.squeeze(0))                                    # [128]

    # F0/N after the strided Conv1d (T axis halved: 2*T_mel → T_mel)
    write_f32(OUT / f"decoder_{tag}_F0_after_conv.bin",
              captures["F0_after_conv"][0, 0])                        # [T_mel]
    write_f32(OUT / f"decoder_{tag}_N_after_conv.bin",
              captures["N_after_conv"][0, 0])                         # [T_mel]

    # Encode + decode blocks (T-major storage: [Ti, Ci])
    write_f32(OUT / f"decoder_{tag}_after_encode.bin",
              captures["after_encode"][0].transpose(1, 0))            # [T_mel, 1024]
    write_f32(OUT / f"decoder_{tag}_asr_res.bin",
              captures["asr_res"][0].transpose(1, 0))                 # [T_mel, 64]
    write_f32(OUT / f"decoder_{tag}_after_decode_block_0.bin",
              captures["after_decode_block_0"][0].transpose(1, 0))    # [T_mel, 1024]
    write_f32(OUT / f"decoder_{tag}_after_decode_block_1.bin",
              captures["after_decode_block_1"][0].transpose(1, 0))    # [T_mel, 1024]
    write_f32(OUT / f"decoder_{tag}_after_decode_block_2.bin",
              captures["after_decode_block_2"][0].transpose(1, 0))    # [T_mel, 1024]
    write_f32(OUT / f"decoder_{tag}_after_decode_block_3.bin",
              captures["after_decode_block_3"][0].transpose(1, 0))    # [2*T_mel, 512]

    # Generator internals
    write_f32(OUT / f"decoder_{tag}_har_source.bin",
              captures["har_source"][0])                              # [600*T_mel]
    write_f32(OUT / f"decoder_{tag}_har_spec.bin",
              captures["har_spec"][0])                                # [11, frames]
    write_f32(OUT / f"decoder_{tag}_har_phase.bin",
              captures["har_phase"][0])                               # [11, frames]
    write_f32(OUT / f"decoder_{tag}_after_ups0.bin",
              captures["after_ups0_raw"][0])                          # [256, ~20*T_mel]
    write_f32(OUT / f"decoder_{tag}_after_ups1.bin",
              captures["after_ups1_raw"][0])                          # [128, ~120*T_mel - 1]
    write_f32(OUT / f"decoder_{tag}_after_resblocks0.bin",
              captures["after_resblocks0"][0])                        # [256, 20*T_mel]
    write_f32(OUT / f"decoder_{tag}_after_resblocks1.bin",
              captures["after_resblocks1"][0])                        # [128, 120*T_mel]
    write_f32(OUT / f"decoder_{tag}_after_conv_post.bin",
              captures["after_conv_post"][0])                         # [22, 120*T_mel]
    write_f32(OUT / f"decoder_{tag}_stft_real.bin",
              captures["stft_real"][0])                               # [11, 120*T_mel]
    write_f32(OUT / f"decoder_{tag}_stft_imag.bin",
              captures["stft_imag"][0])                               # [11, 120*T_mel]

    # Final audio (canonical PyTorch reference).
    audio_np = audio.flatten().cpu().numpy()
    write_f32(OUT / f"decoder_{tag}_audio.bin", audio_np)             # [600*T_mel]

    # ---- Cross-validation against Task 2 (kokoro-onnx) reference -----------
    onnx_audio_path = OUT / f"kokoro_{tag}_audio.bin"
    if onnx_audio_path.exists():
        onnx_audio = read_f32(onnx_audio_path)
        if onnx_audio.size != audio_np.size:
            print(f"  cross-check vs kokoro-onnx: SHAPE MISMATCH "
                  f"(pytorch={audio_np.size}, onnx={onnx_audio.size}) — "
                  f"expected: phonemizer differs (misaki vs espeak-ng). "
                  f"See decoder_architecture.md.")
        else:
            diff = float(np.abs(audio_np - onnx_audio).max())
            print(f"  cross-check vs kokoro-onnx: max|Δ|={diff:.4e}")
    else:
        print(f"  cross-check vs kokoro-onnx: skipped ({onnx_audio_path.name} missing)")

    print(
        f"  shapes: F0_conv={captures['F0_after_conv'].shape[2]}  "
        f"encode={captures['after_encode'].shape}  "
        f"decode_3={captures['after_decode_block_3'].shape}  "
        f"har_source={captures['har_source'].shape}  "
        f"ups0={captures['after_ups0_raw'].shape}  "
        f"ups1={captures['after_ups1_raw'].shape}  "
        f"resblocks1={captures['after_resblocks1'].shape}  "
        f"audio={audio_np.shape}  "
        f"audio min/max={audio_np.min():.3f}/{audio_np.max():.3f}"
    )


for tag, _text in REFERENCE_PHRASES:
    run_and_dump(tag)

written = sum(
    1 for f in os.listdir(OUT)
    if f.startswith("decoder_") and f.endswith(".bin")
)
print(f"\nWrote {written} decoder fixtures to {OUT}")
