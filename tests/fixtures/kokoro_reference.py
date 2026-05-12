"""Generate kokoro-onnx reference outputs for cactus Kokoro tests (Plan 2).

Synthesizes 5 reference phrases with kokoro-onnx (Apache 2.0) and saves both
raw audio (WAV for human listening) and float32 .bin (for cactus tests to
compare against). Also dumps the voice embedding tensor for the converter.

Run: python3.11 tests/fixtures/kokoro_reference.py
Outputs: tests/fixtures/data/kokoro_*.{wav,bin} (gitignored, regenerate locally)

First run downloads ~330MB (kokoro-v0_19.onnx + voices.json) into the
huggingface_hub cache (~/.cache/huggingface/...). Subsequent runs reuse it.
"""
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)


def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


REFERENCE_PHRASES = [
    ("p01", "Hello, my name is Cactus."),
    ("p02", "The quick brown fox jumps over the lazy dog."),
    ("p03", "What time is it in Tokyo right now?"),
    ("p04", "I can run entirely on your device, no cloud needed."),
    ("p05", "Speech synthesis is the inverse of speech recognition."),
]

# Locate model + voices file. Prefer ./assets/kokoro/ if pre-downloaded;
# otherwise download from the kokoro-onnx HF release.
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "assets", "kokoro")
os.makedirs(ASSETS, exist_ok=True)
model_path = os.path.join(ASSETS, "kokoro-v0_19.onnx")
voices_path = os.path.join(ASSETS, "voices.bin")

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
# kokoro-onnx 0.5.x calls np.load() on the voices file, so we need the
# numpy-archive .bin (zip of npy arrays), not the legacy voices.json.
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  using cached {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
        return
    print(f"  downloading {url} -> {dest}")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  done ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")


print("Locating Kokoro model files...")
_download(KOKORO_MODEL_URL, model_path)
_download(VOICES_URL, voices_path)

print(f"Loading Kokoro from {model_path} ...")
kokoro = Kokoro(model_path, voices_path)
voice = "af_bella"

print(f"Synthesizing {len(REFERENCE_PHRASES)} reference phrases (voice={voice}):")
for tag, text in REFERENCE_PHRASES:
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    if sample_rate != 24000:
        sys.exit(f"unexpected sample rate {sample_rate} (expected 24000)")
    sf.write(os.path.join(OUT, f"kokoro_{tag}.wav"), samples, sample_rate)
    write_f32(os.path.join(OUT, f"kokoro_{tag}_audio.bin"), samples)
    print(f"  {tag}: {len(samples) / sample_rate:.2f}s  '{text}'")

# Save the voice embedding for the converter's later use (Plan 2 Task 5)
voices = np.load(voices_path)
voice_emb = np.array(voices[voice], dtype=np.float32)
write_f32(os.path.join(OUT, f"kokoro_voice_{voice}.bin"), voice_emb)

print(f"\nWrote fixtures to {OUT}")
print(f"Voice {voice} embedding shape: {voice_emb.shape}")
print(f"Total .wav + .bin files: {len([f for f in os.listdir(OUT) if f.startswith('kokoro_')])}")
