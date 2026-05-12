"""cactus run-tts -- synthesize text to a WAV file using Kokoro.

Thin Python ctypes wrapper around the cactus_tts C FFI. No model logic
lives here; the C++ KokoroModel does everything (G2P, encoders, predictor,
decoder, generator, iSTFT). This module just loads libcactus.dylib/.so,
calls the four FFI entry points, converts f32 -> int16, and writes a
mono 16-bit WAV at 24 kHz.
"""

from __future__ import annotations

import argparse
import array
import ctypes
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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cactus run-tts",
                                description="Synthesize text to a WAV file with Kokoro.")
    p.add_argument("text", help="Text to synthesize")
    p.add_argument("--out", default="out.wav", help="Output WAV path (default: out.wav)")
    p.add_argument("--weights", default=str(_DEFAULT_WEIGHTS),
                   help=f"Kokoro flat weights directory (default: {_DEFAULT_WEIGHTS})")
    p.add_argument("--dict", default=str(_DEFAULT_DICT),
                   help=f"G2P dictionary path (default: {_DEFAULT_DICT})")
    p.add_argument("--voice", default=str(_DEFAULT_VOICE),
                   help=f"Single-voice raw f32 file, 511*256 floats "
                        f"(default: {_DEFAULT_VOICE})")
    p.add_argument("--speed", type=float, default=1.0, help="Speech rate (default: 1.0)")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    handle = _lib.cactus_tts_create(args.weights.encode(),
                                    args.dict.encode(),
                                    args.voice.encode())
    if not handle:
        sys.exit(
            "Failed to load Kokoro model. Verify --weights, --dict, --voice paths "
            "(and that `cactus build` has been run)."
        )

    try:
        # Size query: NULL out + n=0 -> required size in *out_n.
        n = ctypes.c_size_t(0)
        rc = _lib.cactus_tts_synthesize(
            handle, args.text.encode(), ctypes.c_float(args.speed),
            None, ctypes.byref(n),
        )
        # rc == -3 is the expected "buffer too small / size returned" code.
        if n.value == 0:
            sys.exit(f"synthesize size query returned 0 (rc={rc})")

        buf = (ctypes.c_float * n.value)()
        rc = _lib.cactus_tts_synthesize(
            handle, args.text.encode(), ctypes.c_float(args.speed),
            buf, ctypes.byref(n),
        )
        if rc != 0:
            sys.exit(f"synthesize failed (rc={rc})")

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
