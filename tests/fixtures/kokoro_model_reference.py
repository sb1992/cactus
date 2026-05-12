"""Generate reference fixtures for the cactus C++ Kokoro top-level KokoroModel.

Plan 2 Task 9 fixture generator. Produces:

  bert_p<NN>_input_ids.bin           — int64 KModel-padded phoneme IDs   [T]
  bert_p<NN>_after_emb_norm.bin      — fp32  embeddings + LayerNorm      [T, 128]
  bert_p<NN>_after_emb_proj.bin      — fp32  emb_hidden_mapping output    [T, 768]
  bert_p<NN>_after_layer_0.bin       — fp32  one albert layer applied 1x [T, 768]
  bert_p<NN>_after_layer_5.bin       — fp32  one albert layer applied 6x [T, 768]
  bert_p<NN>_last_hidden_state.bin   — fp32  full bert output (12 layers) [T, 768]
  bert_p<NN>_pooled_output.bin       — fp32  pooler output                [768]
  bert_p<NN>_d_en.bin                — fp32  bert_encoder output          [T, 512]

Plus the af_bella voice as a flat raw binary the C++ KokoroModel can mmap:

  voice_af_bella_raw.bin             — fp32  (511, 256) flattened         [130816]

Run:  /opt/homebrew/bin/python3.11 tests/fixtures/kokoro_model_reference.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertLayer

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


def write_f32(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int64).flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


# Locate the Kokoro config.json (cached HF repo).
HF_HINTS = [Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M"]
config_path = None
for root in HF_HINTS:
    for cfg in root.rglob("config.json"):
        try:
            obj = json.load(open(cfg))
        except Exception:
            continue
        if "vocab" in obj and obj.get("n_token") == 178:
            config_path = cfg
            break
    if config_path:
        break
if config_path is None:
    sys.exit("Could not locate Kokoro config.json. Run KModel(repo_id='hexgrad/Kokoro-82M') once.")
print(f"config: {config_path}")
cfg = json.load(open(config_path))


# ---------------------------------------------------------------------------
# Build a CustomAlbert + bert_encoder skeleton and load weights from kokoro
# v0_19.pth.
# ---------------------------------------------------------------------------
from kokoro.modules import CustomAlbert

print(f"Loading weights from {PTH_PATH} ...")
torch.manual_seed(0)
albert_cfg = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
bert = CustomAlbert(albert_cfg).eval()
bert_encoder = torch.nn.Linear(albert_cfg.hidden_size, cfg["hidden_dim"]).eval()
ckpt = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
net = ckpt["net"]
sd_bert = {k[7:] if k.startswith("module.") else k: v for k, v in net["bert"].items()}
sd_be   = {k[7:] if k.startswith("module.") else k: v for k, v in net["bert_encoder"].items()}
m1, u1 = bert.load_state_dict(sd_bert, strict=False)
m2, u2 = bert_encoder.load_state_dict(sd_be, strict=False)
real_missing = [k for k in m1 if "position_ids" not in k]
real_unexpected = [k for k in u1 if "position_ids" not in k]
if real_missing or real_unexpected:
    print(f"  WARN bert load: missing={real_missing[:5]} unexpected={real_unexpected[:5]}")
print("OK")


# ---------------------------------------------------------------------------
# Voice embedding (af_bella) as a flat raw binary.
# ---------------------------------------------------------------------------
print(f"Loading voices.bin from {VOICES_PATH} ...")
voices = np.load(str(VOICES_PATH))
voice_emb = np.array(voices["af_bella"], dtype=np.float32)   # [511, 1, 256]
print(f"  voice af_bella shape={voice_emb.shape}")
voice_flat = voice_emb.reshape(-1).astype(np.float32)        # [511*256]
voice_path_out = OUT / "voice_af_bella_raw.bin"
voice_flat.tofile(voice_path_out)                            # NO header — flat
print(f"  wrote {voice_path_out} ({voice_flat.nbytes} B, {voice_flat.size} floats)")


# ---------------------------------------------------------------------------
# Phonemizer (use the same misaki-derived IDs the predictor reference dumps).
# We REUSE the existing predictor_p<NN>_input_phonemes.bin fixtures so the
# bert input_ids are byte-identical to what the predictor was tested with.
# ---------------------------------------------------------------------------
def load_predictor_input_ids(tag):
    path = OUT / f"predictor_{tag}_input_phonemes.bin"
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(), dtype=np.int64)
    assert arr.size == n
    return arr


# ---------------------------------------------------------------------------
# Build a per-stage capture of the BERT pipeline. We reproduce KModel.bert
# manually to capture intermediates. The ALBERT weights have a single shared
# layer applied num_hidden_layers (=12) times.
# ---------------------------------------------------------------------------
def manual_bert_forward(input_ids):
    """Capture every per-stage intermediate from CustomAlbert.forward.

    We hook the embeddings module + each application of the shared
    albert_layer to capture (T, 128) post-LN embeddings, (T, 768) post-proj
    embeddings, and the (T, 768) output after each of 12 layer applications.

    Returns dict with: after_emb_norm (T, 128), after_emb_proj (T, 768),
    per_layer (list of 12 (T, 768)), last_hidden, pooled (768,).
    """
    captures = {"after_emb_norm": None, "after_emb_proj": None,
                "per_layer": [], "pooled": None}

    def hook_emb(module, inputs, output):
        # output: (1, T, 128) — embeddings + LayerNorm
        captures["after_emb_norm"] = output[0].detach().cpu().numpy().copy()

    def hook_proj(module, inputs, output):
        # output: (1, T, 768) — encoder.embedding_hidden_mapping_in
        captures["after_emb_proj"] = output[0].detach().cpu().numpy().copy()

    def hook_layer(module, inputs, output):
        # output: tuple — first element is hidden (1, T, 768)
        h = output[0] if isinstance(output, tuple) else output
        captures["per_layer"].append(h[0].detach().cpu().numpy().copy())

    h_emb  = bert.embeddings.register_forward_hook(hook_emb)
    h_proj = bert.encoder.embedding_hidden_mapping_in.register_forward_hook(hook_proj)
    layer  = bert.encoder.albert_layer_groups[0].albert_layers[0]
    h_layer = layer.register_forward_hook(hook_layer)

    try:
        with torch.no_grad():
            T = input_ids.shape[-1]
            text_mask = torch.zeros((1, T), dtype=torch.bool)
            # Run the full bert with attention_mask=all-ones (matches KModel).
            last_hidden = bert(input_ids, attention_mask=(~text_mask).int())  # (1, T, 768)
            pooled = bert.pooler_activation(bert.pooler(last_hidden[:, 0, :]))  # (1, 768)
    finally:
        h_emb.remove()
        h_proj.remove()
        h_layer.remove()

    captures["last_hidden"] = last_hidden[0].cpu().numpy().copy()
    captures["pooled"] = pooled[0].cpu().numpy().copy()
    assert len(captures["per_layer"]) == albert_cfg.num_hidden_layers, \
        f"expected {albert_cfg.num_hidden_layers} layer captures, got {len(captures['per_layer'])}"
    return captures


def run_phrase(tag, text):
    print(f"\n[{tag}] {text!r}")
    raw_ids = load_predictor_input_ids(tag)  # np.int64 (T,) — already padded
    T = raw_ids.size
    print(f"  T={T}, first IDs: {raw_ids[:10].tolist()}")

    input_ids = torch.from_numpy(raw_ids.copy()).reshape(1, -1).long()
    caps = manual_bert_forward(input_ids)

    # bert_encoder runs on the captured last_hidden_state.
    with torch.no_grad():
        d_en = bert_encoder(torch.from_numpy(caps["last_hidden"][None])).cpu().numpy()[0].copy()

    # ---- write fixtures ----
    write_i64(OUT / f"bert_{tag}_input_ids.bin",        raw_ids)
    write_f32(OUT / f"bert_{tag}_after_emb_norm.bin",   caps["after_emb_norm"])
    write_f32(OUT / f"bert_{tag}_after_emb_proj.bin",   caps["after_emb_proj"])
    write_f32(OUT / f"bert_{tag}_after_layer_0.bin",    caps["per_layer"][0])
    write_f32(OUT / f"bert_{tag}_after_layer_5.bin",    caps["per_layer"][5])
    write_f32(OUT / f"bert_{tag}_last_hidden_state.bin", caps["last_hidden"])
    write_f32(OUT / f"bert_{tag}_pooled_output.bin",    caps["pooled"])
    write_f32(OUT / f"bert_{tag}_d_en.bin",             d_en)

    print(f"  wrote bert_{tag}_*.bin (last_hidden shape {caps['last_hidden'].shape}, "
          f"d_en shape {d_en.shape})")


for tag, text in REFERENCE_PHRASES:
    run_phrase(tag, text)

print("\nDone.")
