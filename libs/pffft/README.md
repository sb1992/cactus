# PFFFT (vendored)

Pretty Fast FFT — single-precision real & complex 1D FFT. Originally written by
Julien Pommier; vendored from the rtoy/pffft GitHub mirror (canonical single-file
form of the Pommier distribution).

## Source

- **Upstream:** https://github.com/rtoy/pffft (mirror of https://bitbucket.org/jpommier/pffft)
- **Vendored at upstream commit:** 74d7261be17cf659d5930d4830609406bd7553e3
- **Vendored on:** 2026-05-12

## Why

cactus needs an FFT for the ISTFTNet vocoder in Kokoro TTS (voice loop, stage 1).
PFFFT is small (~70KB total source), NEON-optimized on ARM64, header+source with
no external deps, and BSD-3 licensed. SSE on x86, NEON on ARM, scalar fallback
otherwise.

## Files

- `pffft.h` — public API
- `pffft.c` — main implementation
- `fftpack.h`, `fftpack.c` — fallback FFT used internally by pffft.c for sizes
  the SIMD path cannot handle. Required — do not delete.
- `README.txt` — original upstream README, contains BSD-3-like license text
  (license is also in the source file headers).

## License

BSD-3-like. License text is in the headers of `pffft.c` and in `README.txt`.
Distribution permits commercial use, modification, and redistribution under the
same license terms.

## Modifications

None. To update, re-download from the upstream URLs above and re-record the
upstream commit SHA.
