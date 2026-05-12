"""Run the PyTorch reference Kokoro `text_encoder` on the 5 reference phrases
and dump per-layer intermediate outputs for cactus C++ port validation.

Plan 2 Task 6a fixture generator. The cactus implementation (Task 6b) will
load the same .weights files and must reproduce these intermediates layer-by-layer.

Reference: kokoro PyPI v0.9.4, kokoro/modules.py:35-69 (TextEncoder).
Phonemizer: misaki.en.G2P (the same phonemizer Kokoro/StyleTTS2 use upstream).

Outputs (in tests/fixtures/data/, gitignored):
  text_encoder_p<NN>_input_phonemes.bin     — int64 phoneme IDs    [T]
  text_encoder_p<NN>_after_embedding.bin    — float32              [T, 512]
  text_encoder_p<NN>_after_cnn0.bin         — float32              [T, 512]   (transposed back to T-major)
  text_encoder_p<NN>_after_cnn1.bin         — float32              [T, 512]
  text_encoder_p<NN>_after_cnn2.bin         — float32              [T, 512]
  text_encoder_p<NN>_after_lstm.bin         — float32              [T, 512]

Each .bin layout: [u32 length][f32 ...]   (or [u32 length][i64 ...] for phonemes).
The "length" header is the total element count, NOT byte count.

Run:  /opt/homebrew/bin/python3.11 tests/fixtures/kokoro_text_encoder_reference.py
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure we use the upstream PyTorch reference, not anything local.
from kokoro.modules import TextEncoder
from misaki import en as misaki_en

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "kokoro"
PTH_PATH = ASSETS / "kokoro-v0_19.pth"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

# The 5 reference phrases (must match kokoro_reference.py).
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
# Build the reference text_encoder and load weights from the .pth.
# ---------------------------------------------------------------------------

# Config matches hexgrad/Kokoro-82M config.json:
#   hidden_dim = 512, text_encoder_kernel_size = 5, n_layer = 3, n_token = 178
torch.manual_seed(0)
model = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178)

ckpt = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
te_state_raw = ckpt["net"]["text_encoder"]
# Strip "module." prefix (DDP artifact).
te_state = {k[7:] if k.startswith("module.") else k: v for k, v in te_state_raw.items()}

# weight_norm parameterization expects weight_g + weight_v in the state dict;
# we have those plus `bias`. Load in non-strict mode so PyTorch can attach the
# parameters without complaining about the absent fused `.weight` (it will be
# computed on first forward pass from g and v).
missing, unexpected = model.load_state_dict(te_state, strict=False)
# We expect `weight` (the fused tensor) to be missing for each conv — that's fine.
expected_missing = {f"cnn.{i}.0.weight" for i in range(3)}
extra_missing = set(missing) - expected_missing
assert not extra_missing, f"unexpected missing keys: {extra_missing}"
assert not unexpected, f"unexpected extra keys: {unexpected}"

model.eval()


# ---------------------------------------------------------------------------
# Build vocab + phonemizer.
# ---------------------------------------------------------------------------

# Vocab from config.json (the source of truth for ID assignments).
HF_CONFIG_HINTS = [
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M",
]
vocab = None
for root in HF_CONFIG_HINTS:
    for cfg in root.rglob("config.json"):
        try:
            with open(cfg) as f:
                obj = json.load(f)
            if "vocab" in obj and obj.get("n_token") == 178:
                vocab = obj["vocab"]
                print(f"Loaded vocab from {cfg}")
                break
        except Exception:
            continue
    if vocab:
        break
if vocab is None:
    sys.exit(
        "Could not locate Kokoro config.json (need vocab map). "
        "Run `python3.11 -c 'from kokoro import KModel; KModel(repo_id=\"hexgrad/Kokoro-82M\")'` "
        "once to populate the HF cache."
    )

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
# Manual forward that mirrors TextEncoder.forward but captures intermediates.
# We avoid forward_hooks because the LayerNorm in `nn.Sequential` would also
# transpose/un-transpose and we want a single canonical "after the block" tensor.
# ---------------------------------------------------------------------------

def run_and_dump(tag: str, text: str) -> None:
    print(f"\n[{tag}] text='{text}'")
    ids = phonemize(text)
    if not ids:
        sys.exit(f"phonemizer produced empty IDs for '{text}'")
    # Match kokoro/model.py: prepend a 0 token (the "BOS" pad token kokoro
    # uses for inference) so the encoder sees an explicit boundary.
    # We DO NOT prepend here — the upstream kokoro.KModel only prepends in
    # the pipeline. The text_encoder itself just consumes whatever IDs it
    # gets, so for fixture purposes we use the raw misaki output. Cactus
    # must match this convention.
    T = len(ids)
    print(f"  T={T}, first few IDs: {ids[:10]}")

    x_ids = torch.tensor([ids], dtype=torch.long)                # [1, T]
    input_lengths = torch.tensor([T], dtype=torch.long)
    text_mask = torch.zeros((1, T), dtype=torch.bool)            # all valid

    # --- Step 1: embedding ---
    with torch.no_grad():
        emb = model.embedding(x_ids)        # [1, T, 512]
    after_embedding = emb[0].cpu().numpy()  # [T, 512]

    # --- Step 2: transpose + mask, then 3 CNN blocks ---
    x = emb.transpose(1, 2)                 # [1, 512, T]
    m = text_mask.unsqueeze(1)              # [1, 1, T]
    x = x.masked_fill(m, 0.0)

    cnn_outputs = []
    with torch.no_grad():
        for i, block in enumerate(model.cnn):
            x = block(x)                     # Conv1d(weight_norm) -> LayerNorm -> LeakyReLU -> Dropout(eval=identity)
            x = x.masked_fill(m, 0.0)
            # Capture in T-major form to be consistent with embedding output.
            cnn_outputs.append(x[0].transpose(0, 1).cpu().numpy())  # [T, 512]

    # --- Step 3: LSTM ---
    x_lstm_in = x.transpose(1, 2)           # [1, T, 512]
    with torch.no_grad():
        # Unbatched B=1 unpadded path is identical in numerics to pack/unpack.
        lstm_out, _ = model.lstm(x_lstm_in)  # [1, T, 512]
    after_lstm = lstm_out[0].cpu().numpy()   # [T, 512]

    # --- Cross-check against the official forward (full mask path) ---
    with torch.no_grad():
        full_out = model(x_ids, input_lengths, text_mask)   # [1, 512, T]
    full_T_major = full_out[0].transpose(0, 1).cpu().numpy()  # [T, 512]
    diff = np.max(np.abs(full_T_major - after_lstm))
    if diff > 1e-5:
        sys.exit(f"[{tag}] reference path mismatch vs official forward: max|Δ|={diff}")
    print(f"  matches official forward (max|Δ|={diff:.2e})")

    # --- Write fixtures ---
    write_i64(OUT / f"text_encoder_{tag}_input_phonemes.bin", ids)
    write_f32(OUT / f"text_encoder_{tag}_after_embedding.bin", after_embedding)
    write_f32(OUT / f"text_encoder_{tag}_after_cnn0.bin", cnn_outputs[0])
    write_f32(OUT / f"text_encoder_{tag}_after_cnn1.bin", cnn_outputs[1])
    write_f32(OUT / f"text_encoder_{tag}_after_cnn2.bin", cnn_outputs[2])
    write_f32(OUT / f"text_encoder_{tag}_after_lstm.bin",  after_lstm)
    print(
        f"  shapes: emb={after_embedding.shape}  "
        f"cnn0={cnn_outputs[0].shape}  cnn1={cnn_outputs[1].shape}  "
        f"cnn2={cnn_outputs[2].shape}  lstm={after_lstm.shape}"
    )


for tag, text in REFERENCE_PHRASES:
    run_and_dump(tag, text)

print(f"\nWrote {len(REFERENCE_PHRASES)*6} fixtures to {OUT}")
