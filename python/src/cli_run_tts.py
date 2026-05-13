"""cactus run-tts -- synthesize text to a WAV file using Kokoro.

Thin Python ctypes wrapper around the cactus_tts C FFI. The C++ KokoroModel
runs the encoders, predictor, decoder, generator, and iSTFT. Phonemization
lives on the Python side via misaki (the official Kokoro phonemizer) so that
the C++ runtime stays Python-free and we sidestep the imperfect ARPAbet->IPA
mapping baked into the legacy text path.

Pipeline:
    text -> misaki.en.G2P -> IPA phonemes -> Kokoro vocab IDs (i64)
                                                 |
                                                 v
                                cactus_tts_synthesize_phonemes
                                                 |
                                                 v
                              KokoroModel::synthesize_with_phonemes
"""

from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import sys
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_lib() -> str:
    """Locate libcactus shared library produced by `cactus build`."""
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    env = os.environ.get("CACTUS_LIB")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        repo / "cactus" / "build" / "libcactus.dylib",
        repo / "cactus" / "build" / "libcactus.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    sys.exit(
        "libcactus shared library not found. Run `cactus build` first. "
        f"Looked in: {', '.join(str(c) for c in candidates)}"
    )


_lib = ctypes.CDLL(_find_lib())

_lib.cactus_tts_create.restype = ctypes.c_void_p
_lib.cactus_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

_lib.cactus_tts_destroy.restype = None
_lib.cactus_tts_destroy.argtypes = [ctypes.c_void_p]

_lib.cactus_tts_synthesize.restype = ctypes.c_int
_lib.cactus_tts_synthesize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_size_t),
]

_lib.cactus_tts_synthesize_phonemes.restype = ctypes.c_int
_lib.cactus_tts_synthesize_phonemes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_size_t),
]

_lib.cactus_tts_sample_rate.restype = ctypes.c_int
_lib.cactus_tts_sample_rate.argtypes = [ctypes.c_void_p]


# ---------------------------------------------------------------------------
# Default asset locations (resolved relative to the repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_WEIGHTS = _REPO_ROOT / "assets" / "kokoro" / "weights"
_DEFAULT_DICT    = _REPO_ROOT / "assets" / "kokoro" / "kokoro-g2p.dict"
# voices.bin in assets/ is a multi-voice container; the single-voice raw
# float bin that KokoroModel::load expects (511 * 256 floats) lives under
# tests/fixtures/data/. Either file can be overridden via --voice.
_DEFAULT_VOICE   = _REPO_ROOT / "tests" / "fixtures" / "data" / "voice_af_bella_raw.bin"


# ---------------------------------------------------------------------------
# Phonemization (misaki + Kokoro vocab)
# ---------------------------------------------------------------------------

# Search hints for the Kokoro config.json that holds the IPA -> token-ID map.
# Currently shipped by the `hexgrad/Kokoro-82M` HF repo (n_token=178, vocab
# has 114 entries; the gap covers reserved IDs).
_HF_CONFIG_HINTS = [
    Path.home() / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M",
]


def _load_kokoro_vocab() -> dict:
    """Locate Kokoro's IPA-char -> token-ID dict from the HF config cache."""
    for root in _HF_CONFIG_HINTS:
        if not root.exists():
            continue
        for cfg in root.rglob("config.json"):
            try:
                with open(cfg) as f:
                    obj = json.load(f)
            except Exception:
                continue
            if "vocab" in obj and obj.get("n_token") == 178:
                return obj["vocab"]
    sys.exit(
        "Could not locate Kokoro config.json with vocab map. Populate the "
        "HF cache via `python -c 'from kokoro import KModel; "
        "KModel(repo_id=\"hexgrad/Kokoro-82M\")'` once."
    )


def _phonemes_to_ids(phonemes: str, vocab: dict) -> list[int]:
    """Map an IPA string to Kokoro vocab IDs, wrapping with KModel pad tokens.

    KModel.forward (model.py:131) prepends + appends a 0 token. We mirror that
    here so the C++ side can take ids verbatim.
    """
    ids: list[int] = [0]
    skipped: list[str] = []
    for ch in phonemes:
        if ch in vocab:
            ids.append(int(vocab[ch]))
        else:
            skipped.append(ch)
    ids.append(0)
    if skipped:
        sys.stderr.write(
            f"warn: dropped {len(skipped)} unmapped phoneme chars: "
            f"{''.join(skipped)!r}\n"
        )
    return ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cactus run-tts",
                                description="Synthesize text to a WAV file with Kokoro.")
    p.add_argument("text", help="Text to synthesize (ignored if --phonemes is set)")
    p.add_argument("--out", default="out.wav", help="Output WAV path (default: out.wav)")
    p.add_argument("--weights", default=str(_DEFAULT_WEIGHTS),
                   help=f"Kokoro flat weights directory (default: {_DEFAULT_WEIGHTS})")
    p.add_argument("--dict", default=str(_DEFAULT_DICT),
                   help=f"G2P dictionary path (default: {_DEFAULT_DICT}). "
                        "Unused now that phonemization runs in Python via misaki, "
                        "but KokoroModel::load still requires a valid dict file.")
    p.add_argument("--voice", default=str(_DEFAULT_VOICE),
                   help=f"Single-voice raw f32 file, 511*256 floats "
                        f"(default: {_DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=1.0, help="Speech rate (default: 1.0)")
    p.add_argument("--phonemes", default=None,
                   help="IPA phoneme string (bypass misaki). Useful for testing.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # 1) Phonemize on the Python side (misaki) unless the caller passed IPA directly.
    if args.phonemes is not None:
        phonemes = args.phonemes
    else:
        try:
            from misaki import en as misaki_en
        except ImportError:
            sys.exit(
                "misaki is required for the text path. Install with "
                "`pip install misaki[en]` or pass --phonemes <IPA-string>."
            )
        # Enable misaki's espeak-ng-based letter-to-sound fallback so
        # out-of-vocabulary words (proper nouns, loanwords, etc.) get a
        # phoneme approximation instead of misaki's "❓" unk marker, which
        # would later be dropped by _phonemes_to_ids and produce silence
        # at the corresponding position in the audio.
        fallback = None
        try:
            from misaki import espeak as misaki_espeak
            fallback = misaki_espeak.EspeakFallback(british=False)
        except Exception as e:
            sys.stderr.write(
                f"warn: misaki[espeak] L2S fallback unavailable ({e}); "
                "OOV words will be dropped from synthesis. Install with "
                "`pip install misaki[espeak]` (and ensure espeak-ng is "
                "available) for full coverage.\n"
            )
        g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback)
        phonemes, _tokens = g2p(args.text)

    vocab = _load_kokoro_vocab()
    ids = _phonemes_to_ids(phonemes, vocab)
    if len(ids) <= 2:
        sys.exit("phonemizer produced no usable IDs (only the pad tokens)")

    print(f"Phonemes: {phonemes}")
    preview = ids[:10] + (["..."] + ids[-3:] if len(ids) > 13 else [])
    print(f"IDs ({len(ids)}): {preview}")

    handle = _lib.cactus_tts_create(args.weights.encode(),
                                    args.dict.encode(),
                                    args.voice.encode())
    if not handle:
        sys.exit(
            "Failed to load Kokoro model. Verify --weights, --dict, --voice paths "
            "(and that `cactus build` has been run)."
        )

    try:
        ids_arr = (ctypes.c_int64 * len(ids))(*ids)

        # Size query: NULL out + n=0 -> required size in *out_n.
        n = ctypes.c_size_t(0)
        rc = _lib.cactus_tts_synthesize_phonemes(
            handle, ids_arr, len(ids), ctypes.c_float(args.speed),
            None, ctypes.byref(n),
        )
        # rc == -3 is the expected "buffer too small / size returned" code.
        if n.value == 0:
            sys.exit(f"synthesize_phonemes size query returned 0 (rc={rc})")

        buf = (ctypes.c_float * n.value)()
        rc = _lib.cactus_tts_synthesize_phonemes(
            handle, ids_arr, len(ids), ctypes.c_float(args.speed),
            buf, ctypes.byref(n),
        )
        if rc != 0:
            sys.exit(f"synthesize_phonemes failed (rc={rc})")

        sr = _lib.cactus_tts_sample_rate(handle)

        # Convert f32 [-1, 1] -> int16 PCM and write a mono WAV.
        f32 = array.array("f", buf)
        i16 = array.array(
            "h",
            [max(-32768, min(32767, int(s * 32767.0))) for s in f32],
        )
        with wave.open(args.out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(i16.tobytes())

        print(f"Wrote {args.out}: {n.value} samples, "
              f"{n.value / sr:.2f}s @ {sr} Hz")
        return 0
    finally:
        _lib.cactus_tts_destroy(handle)


if __name__ == "__main__":
    sys.exit(main())
