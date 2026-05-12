"""Run the PyTorch reference Kokoro predictor on the 5 reference phrases
and dump per-stage intermediate outputs for cactus C++ port validation.

Plan 2 Task 7a fixture generator. The cactus implementation (Task 7b) will
load the same .weights files and must reproduce these stage-by-stage.

Reference: kokoro PyPI v0.9.4
  - kokoro/modules.py     ProsodyPredictor (lines 91–134), DurationEncoder, AdaLayerNorm
  - kokoro/istftnet.py    AdainResBlk1d, AdaIN1d, UpSample1d
  - kokoro/model.py       KModel.forward_with_tokens (lines 86–119)

Phonemizer: misaki.en.G2P (the same phonemizer Kokoro/StyleTTS2 use upstream).
Voice: af_bella, slice indexed by len(input_ids)-1 (the kokoro-onnx convention).

Outputs (in tests/fixtures/data/, gitignored):
  predictor_p<NN>_input_phonemes.bin       — int64 phoneme IDs (with KModel pad) [T]
  predictor_p<NN>_style_s.bin              — float32 style vector              [128]
  predictor_p<NN>_input_d_en.bin           — float32 bert_encoder output       [T, 512]
                                              (predictor sub-input; channel-last for storage)
  predictor_p<NN>_dur_enc_after_lstm0.bin  — float32 DurEnc LSTM 0 out         [T, 512]
  predictor_p<NN>_dur_enc_after_aln1.bin   — float32 DurEnc AdaLN 1 out        [T, 512]
  predictor_p<NN>_dur_enc_after_lstm2.bin  — float32 DurEnc LSTM 2 out         [T, 512]
  predictor_p<NN>_dur_enc_after_aln3.bin   — float32 DurEnc AdaLN 3 out        [T, 512]
  predictor_p<NN>_dur_enc_after_lstm4.bin  — float32 DurEnc LSTM 4 out         [T, 512]
  predictor_p<NN>_dur_enc_after_aln5.bin   — float32 DurEnc AdaLN 5 out        [T, 512]
  predictor_p<NN>_d.bin                    — float32 DurEnc final              [T, 640]
  predictor_p<NN>_after_top_lstm.bin       — float32 predictor.lstm out        [T, 512]
  predictor_p<NN>_duration_proj.bin        — float32 raw 50-dim per-phoneme    [T, 50]
  predictor_p<NN>_durations.bin            — int64 pred_dur                    [T]
  predictor_p<NN>_alignment.bin            — float32 0/1 alignment matrix      [T, T_mel]
  predictor_p<NN>_en.bin                   — float32 d.T @ aln                 [T_mel, 640]
  predictor_p<NN>_after_shared_lstm.bin    — float32 shared(en) out            [T_mel, 512]
  predictor_p<NN>_F0_block0.bin            — float32 AdainResBlk1d F0[0] out   [T_mel, 512]
  predictor_p<NN>_F0_block1.bin            — float32 AdainResBlk1d F0[1] out   [2*T_mel, 256]
  predictor_p<NN>_F0_block2.bin            — float32 AdainResBlk1d F0[2] out   [2*T_mel, 256]
  predictor_p<NN>_F0_pred.bin              — float32 final F0_proj squeeze     [2*T_mel]
  predictor_p<NN>_N_block0.bin             — float32 AdainResBlk1d N[0]        [T_mel, 512]
  predictor_p<NN>_N_block1.bin             — float32 AdainResBlk1d N[1]        [2*T_mel, 256]
  predictor_p<NN>_N_block2.bin             — float32 AdainResBlk1d N[2]        [2*T_mel, 256]
  predictor_p<NN>_N_pred.bin               — float32 final N_proj squeeze      [2*T_mel]

Each .bin layout: [u32 length][f32 ...]    (or [u32 length][i64 ...] for ints).
The "length" header is the total element count, NOT byte count.

Run:  /opt/homebrew/bin/python3.11 tests/fixtures/kokoro_predictor_reference.py
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Use the upstream PyTorch reference — same install Task 6a used.
from misaki import en as misaki_en

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


def write_f32(path: Path, arr) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


def write_i64(path: Path, arr) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int64).flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Locate config.json (for vocab) and load the official KModel.
# ---------------------------------------------------------------------------

# Vocab from config.json (the source of truth for ID assignments).
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

print(f"Building KModel skeleton (config={config_path}) ...")
torch.manual_seed(0)
with open(config_path) as f:
    cfg = json.load(f)
vocab = cfg["vocab"]

# KModel.__init__ insists on a checkpoint with top-level keys matching its
# attributes (bert / bert_encoder / predictor / text_encoder / decoder). The
# bundled v0_19 .pth wraps everything in {"net": {...}}, so we instantiate the
# skeleton (passing a dummy checkpoint path is awkward) and load weights by
# hand. This matches how kokoro_text_encoder_reference.py does it.
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from kokoro.istftnet import Decoder
from transformers import AlbertConfig

class _Skeleton(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = CustomAlbert(AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"]))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, cfg["hidden_dim"])
        self.predictor = ProsodyPredictor(
            style_dim=cfg["style_dim"], d_hid=cfg["hidden_dim"],
            nlayers=cfg["n_layer"], max_dur=cfg["max_dur"], dropout=cfg["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=cfg["hidden_dim"], kernel_size=cfg["text_encoder_kernel_size"],
            depth=cfg["n_layer"], n_symbols=cfg["n_token"],
        )
        self.decoder = Decoder(
            dim_in=cfg["hidden_dim"], style_dim=cfg["style_dim"],
            dim_out=cfg["n_mels"], **cfg["istftnet"],
        )

print(f"Loading weights from {PTH_PATH} ...")
model = _Skeleton(cfg)
ckpt = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
net = ckpt["net"]
for key, sd in net.items():
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    sub = getattr(model, key)
    missing, unexpected = sub.load_state_dict(sd, strict=False)
    # Fused .weight is missing for every weight_norm Conv* — that's expected.
    # InstanceNorm1d affine params (`*.norm.weight/bias`) are also missing —
    # the v0_19 .pth was trained with `affine=False`, hexgrad later flipped it
    # to True for an ONNX export bug fix (see comment in istftnet.py:23).
    # PyTorch's default init for those params is `weight=1, bias=0`, which
    # makes the affine a no-op — i.e. exactly equivalent to `affine=False`.
    real_missing = [
        k for k in missing
        if not k.endswith(".weight")
        and not k.endswith(".norm.weight")
        and not k.endswith(".norm.bias")
    ]
    real_unexpected = [k for k in unexpected if "position_ids" not in k]
    if real_missing or real_unexpected:
        if key == "predictor":
            assert not real_missing, f"predictor missing: {real_missing}"
            assert not real_unexpected, f"predictor unexpected: {real_unexpected}"
        else:
            print(f"  [{key}] missing={len(real_missing)} unexpected={len(real_unexpected)}")
model.eval()
predictor = model.predictor
print("Predictor loaded OK")


# ---------------------------------------------------------------------------
# Load voices.bin and the af_bella voice embedding.
# ---------------------------------------------------------------------------

print(f"Loading voices.bin from {VOICES_PATH} ...")
voices = np.load(str(VOICES_PATH))
voice_name = "af_bella"
voice_emb = np.array(voices[voice_name], dtype=np.float32)   # [511, 1, 256]
print(f"  voice {voice_name}: shape={voice_emb.shape}")


# ---------------------------------------------------------------------------
# Phonemizer.
# ---------------------------------------------------------------------------

g2p = misaki_en.G2P(trf=False, british=False)


def phonemize(text: str) -> list[int]:
    phonemes, _tokens = g2p(text)
    ids: list[int] = []
    skipped: list[str] = []
    for ch in phonemes:
        if ch in vocab:
            ids.append(int(vocab[ch]))
        else:
            skipped.append(ch)
    if skipped:
        print(f"  WARN: dropped {len(skipped)} unmapped chars: {skipped!r}")
    return ids


# ---------------------------------------------------------------------------
# Manual forward that mirrors KModel.forward_with_tokens exactly but captures
# every per-stage intermediate. We reproduce model.py:86-119 line-for-line so
# the cactus port can validate sub-stage by sub-stage.
#
# We avoid forward_hooks because (a) DurationEncoder runs its blocks inside a
# Python loop with control flow that hooks would only capture at module-level
# granularity, and (b) we want both pre- and post-cat-with-style snapshots in
# a deterministic order. The manual replication is short and matches the
# upstream reference verbatim.
# ---------------------------------------------------------------------------

def run_and_dump(tag: str, text: str) -> None:
    print(f"\n[{tag}] text='{text}'")
    raw_ids = phonemize(text)
    if not raw_ids:
        sys.exit(f"phonemizer produced empty IDs for '{text}'")

    # KModel.forward pads with leading + trailing 0 (model.py:131).
    input_ids = torch.LongTensor([[0, *raw_ids, 0]])     # [1, T]
    T = input_ids.shape[-1]
    print(f"  T={T}, first IDs: {input_ids[0, :10].tolist()}")

    # Voice slice: kokoro-onnx convention is voices[voice][len(input_ids)-1].
    # Officially KPipeline does `pack[len(input_ids)-1]` (see kokoro/pipeline.py).
    ref_s_np = voice_emb[T - 1]                          # [1, 256]
    ref_s = torch.from_numpy(np.asarray(ref_s_np, dtype=np.float32))    # [1, 256]
    s_full = ref_s                                        # [1, 256]
    s_predictor = ref_s[:, 128:]                          # [1, 128]
    print(f"  ref_s shape={tuple(ref_s.shape)} (slice index {T-1})")

    input_lengths = torch.LongTensor([T])
    text_mask = torch.zeros((1, T), dtype=torch.bool)    # B=1, no padding

    with torch.no_grad():
        # ---- Stage A: BERT + bert_encoder -> d_en ------------------------
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())  # [1, T, 768]
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)                # [1, 512, T]

        # ---- Stage B: DurationEncoder (manual replica of modules.py:148-176)
        dur_enc = predictor.text_encoder
        masks = text_mask
        x = d_en                                                  # [1, 512, T]
        x = x.permute(2, 0, 1)                                    # [T, 1, 512]
        s = s_predictor.expand(x.shape[0], x.shape[1], -1)        # [T, 1, 128]
        x = torch.cat([x, s], axis=-1)                            # [T, 1, 640]
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)                                     # [1, T, 640]
        x = x.transpose(-1, -2)                                   # [1, 640, T]

        dur_enc_intermediates: list[np.ndarray] = []
        for i, block in enumerate(dur_enc.lstms):
            from kokoro.modules import AdaLayerNorm
            if isinstance(block, AdaLayerNorm):
                # AdaLayerNorm path (modules.py:158-160)
                x = block(x.transpose(-1, -2), s_predictor).transpose(-1, -2)  # [1, 512, T]
                # Capture the post-AdaLN tensor BEFORE re-cat with style so the
                # fixture exactly matches the named "after_aln*" stage.
                dur_enc_intermediates.append(
                    x[0].transpose(0, 1).cpu().numpy().copy()      # [T, 512]
                )
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)     # [1, 640, T]
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                # BiLSTM path (modules.py:161-174). For B=1 unpadded, pack/pad
                # is numerically identical to a plain unpacked forward.
                x_in = x.transpose(-1, -2)                          # [1, T, 640]
                block.flatten_parameters()
                x_lstm, _ = block(x_in)                             # [1, T, 512]
                # F.dropout(p=self.dropout, training=False) -> identity
                x = x_lstm.transpose(-1, -2)                        # [1, 512, T]
                # Capture the post-LSTM tensor in T-major form.
                dur_enc_intermediates.append(
                    x[0].transpose(0, 1).cpu().numpy().copy()       # [T, 512]
                )
        d_manual = x.transpose(-1, -2)                              # [1, T, 640]

        # Cross-check vs official DurationEncoder.forward.
        d_official = dur_enc(d_en, s_predictor, input_lengths, text_mask)   # [1, T, 640]
        diff_d = float((d_manual - d_official).abs().max().item())
        if diff_d > 1e-5:
            sys.exit(f"[{tag}] DurationEncoder manual vs official mismatch: {diff_d}")
        d = d_official                                              # [1, T, 640]

        # ---- Stage C: top-level predictor.lstm + duration_proj ----------
        x_top, _ = predictor.lstm(d)                                # [1, T, 512]
        # Note: model.py:107 does NOT apply masked_fill here — it just runs
        # the LSTM unpadded (B=1 inference). We follow exactly.
        duration_logits = predictor.duration_proj(x_top)            # [1, T, 50]
        duration = torch.sigmoid(duration_logits).sum(axis=-1) / 1.0  # speed=1
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        T_mel = int(pred_dur.sum().item())
        print(f"  T_mel = sum(pred_dur) = {T_mel}  (avg {T_mel/T:.2f} frames/phoneme)")

        # ---- Stage D: alignment matrix ---------------------------------
        indices = torch.repeat_interleave(torch.arange(T), pred_dur)
        pred_aln = torch.zeros((T, T_mel))
        pred_aln[indices, torch.arange(T_mel)] = 1
        pred_aln_b = pred_aln.unsqueeze(0)                          # [1, T, T_mel]
        en = d.transpose(-1, -2) @ pred_aln_b                       # [1, 640, T_mel]

        # ---- Stage E: F0 / N branches -----------------------------------
        # Replica of ProsodyPredictor.F0Ntrain (modules.py:124-134).
        x_shared, _ = predictor.shared(en.transpose(-1, -2))        # [1, T_mel, 512]
        x_branch = x_shared.transpose(-1, -2)                       # [1, 512, T_mel]

        F0 = x_branch
        F0_blocks = []
        for blk in predictor.F0:
            F0 = blk(F0, s_predictor)
            # Capture in T-major form so storage is consistent: [Ti, Ci]
            F0_blocks.append(
                F0[0].transpose(0, 1).cpu().numpy().copy()
            )
        F0_pred = predictor.F0_proj(F0).squeeze(1)                  # [1, 2*T_mel]

        N = x_branch
        N_blocks = []
        for blk in predictor.N:
            N = blk(N, s_predictor)
            N_blocks.append(
                N[0].transpose(0, 1).cpu().numpy().copy()
            )
        N_pred = predictor.N_proj(N).squeeze(1)                     # [1, 2*T_mel]

        # Cross-check vs official F0Ntrain.
        F0_official, N_official = predictor.F0Ntrain(en, s_predictor)
        diff_F0 = float((F0_official - F0_pred).abs().max().item())
        diff_N = float((N_official - N_pred).abs().max().item())
        if diff_F0 > 1e-5 or diff_N > 1e-5:
            sys.exit(f"[{tag}] F0Ntrain manual vs official mismatch: F0={diff_F0}, N={diff_N}")

        # We've already cross-checked the DurationEncoder and F0Ntrain sub-stages
        # against the official module forwards. The .pth checkpoint format
        # ({"net": {...}}) doesn't match KModel.__init__'s expectations, so we
        # can't easily run forward_with_tokens end-to-end here, but every
        # predictor sub-stage above is the verbatim upstream PyTorch reference.
        print(f"  cross-checks pass: d max|Δ|={diff_d:.2e}, "
              f"F0 max|Δ|={diff_F0:.2e}, N max|Δ|={diff_N:.2e}")

    # ---- Write fixtures ---------------------------------------------------
    write_i64(OUT / f"predictor_{tag}_input_phonemes.bin", input_ids[0].cpu().numpy())
    write_f32(OUT / f"predictor_{tag}_style_s.bin", s_predictor[0].cpu().numpy())     # [128]
    write_f32(OUT / f"predictor_{tag}_input_d_en.bin",
              d_en[0].transpose(0, 1).cpu().numpy())                                  # [T, 512]

    # DurationEncoder per-stage intermediates (alternating LSTM/AdaLN).
    stage_names = [
        "dur_enc_after_lstm0", "dur_enc_after_aln1",
        "dur_enc_after_lstm2", "dur_enc_after_aln3",
        "dur_enc_after_lstm4", "dur_enc_after_aln5",
    ]
    for name, arr in zip(stage_names, dur_enc_intermediates):
        write_f32(OUT / f"predictor_{tag}_{name}.bin", arr)

    write_f32(OUT / f"predictor_{tag}_d.bin", d[0].cpu().numpy())                     # [T, 640]
    write_f32(OUT / f"predictor_{tag}_after_top_lstm.bin", x_top[0].cpu().numpy())    # [T, 512]
    write_f32(OUT / f"predictor_{tag}_duration_proj.bin", duration_logits[0].cpu().numpy())  # [T, 50]
    write_i64(OUT / f"predictor_{tag}_durations.bin", pred_dur.cpu().numpy())         # [T]
    write_f32(OUT / f"predictor_{tag}_alignment.bin", pred_aln.cpu().numpy())         # [T, T_mel]
    write_f32(OUT / f"predictor_{tag}_en.bin",
              en[0].transpose(0, 1).cpu().numpy())                                    # [T_mel, 640]
    write_f32(OUT / f"predictor_{tag}_after_shared_lstm.bin",
              x_shared[0].cpu().numpy())                                              # [T_mel, 512]

    write_f32(OUT / f"predictor_{tag}_F0_block0.bin", F0_blocks[0])
    write_f32(OUT / f"predictor_{tag}_F0_block1.bin", F0_blocks[1])
    write_f32(OUT / f"predictor_{tag}_F0_block2.bin", F0_blocks[2])
    write_f32(OUT / f"predictor_{tag}_F0_pred.bin", F0_pred[0].cpu().numpy())         # [2*T_mel]

    write_f32(OUT / f"predictor_{tag}_N_block0.bin", N_blocks[0])
    write_f32(OUT / f"predictor_{tag}_N_block1.bin", N_blocks[1])
    write_f32(OUT / f"predictor_{tag}_N_block2.bin", N_blocks[2])
    write_f32(OUT / f"predictor_{tag}_N_pred.bin", N_pred[0].cpu().numpy())           # [2*T_mel]

    print(
        f"  shapes: d_en={tuple(d_en.shape)}  d={tuple(d.shape)}  "
        f"top_lstm={tuple(x_top.shape)}  duration_proj={tuple(duration_logits.shape)}  "
        f"pred_dur={tuple(pred_dur.shape)}  T_mel={T_mel}  "
        f"en={tuple(en.shape)}  shared={tuple(x_shared.shape)}\n"
        f"          F0[0]={F0_blocks[0].shape}  F0[1]={F0_blocks[1].shape}  "
        f"F0[2]={F0_blocks[2].shape}  F0_pred={tuple(F0_pred.shape)}  "
        f"N_pred={tuple(N_pred.shape)}"
    )


for tag, text in REFERENCE_PHRASES:
    run_and_dump(tag, text)

# Count fixtures written for the last tag (one constant per phrase).
written = sum(
    1 for f in os.listdir(OUT) if f.startswith("predictor_") and f.endswith(".bin")
)
print(f"\nWrote {written} predictor fixtures to {OUT}")
