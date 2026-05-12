# Plan 1 — TTS Kernel Foundation: COMPLETE

**Date completed:** 2026-05-12
**Branch:** `voice-loop`
**Final commit:** `638e0eb6`
**Plan file:** [`2026-05-12-voice-loop-plan-1-tts-kernels.md`](./2026-05-12-voice-loop-plan-1-tts-kernels.md)
**Spec:** [`docs/superpowers/specs/2026-05-12-voice-loop-stage-1-design.md`](../specs/2026-05-12-voice-loop-stage-1-design.md)

## Shipped

- **PFFFT vendored** at `libs/pffft/` (5 source files: `pffft.h`, `pffft.c`, `fftpack.h`, `fftpack.c`, `README.txt` — pristine, BSD-3-like NCAR/UCAR license verified). Plus `libs/pffft/README.md` documenting upstream commit `74d7261b…` from `rtoy/pffft`. NEON path live on ARM64 macOS.
- **PFFFT wired into the cactus library build** as a separate OBJECT library (`cactus_pffft_obj`) mirroring the existing `cactus_i8mm_obj` pattern. Linked into both `cactus` (STATIC) and `cactus_ffi` (SHARED). Project `LANGUAGES` upgraded from `CXX` to `C CXX` to compile pffft.c. Third-party warnings suppressed with `-w` (annotated).
- **`cactus_fft_r2c` / `cactus_fft_c2r`** at `cactus/kernel/kernel_fft.{h,cpp}`. Wraps PFFFT ordered transforms with **scipy.fft.rfft compatible layout** on input/output (interleaved `[re, im]` of length N+2 with explicit zeros for `im_0` and `im_{N/2}`). Validated against scipy reference at N=256 (5-cycle sine), N=512 (deterministic random), single-bin spectrum reconstruction, and full R2C→C2R round-trip — all within 1e-3 (forward) / 1e-4 (inverse + round-trip) tolerance.
- **`cactus_stft` / `cactus_istft`** at `cactus/kernel/kernel_stft.{h,cpp}`. Hann window + per-frame R2C/C2R + overlap-add + per-sample window-sum normalization (COLA-correct). Validated by ISTFT round-trip on a 24kHz chirp signal vs scipy reference within 1e-3 on interior samples (edges leak from windowing as expected).
- **`cactus_lstm_cell`** at `cactus/kernel/kernel_lstm.{h,cpp}` (added alongside the pre-existing `cactus_lstm_cell_f16` — additive, no regression). Scalar fp32 PyTorch-compatible LSTMCell with gate order `[i, f, g, o]`. Validated against torch reference at I=64, H=128, tolerance 1e-4.
- **`cactus_conv1d_transpose`** at `cactus/kernel/kernel_conv1d_transpose.{h,cpp}`. Scalar fp32 PyTorch-compatible ConvTranspose1d with weight order `(C_in, C_out, K)` (note: distinct from regular Conv1d's `(C_out, C_in, K)`). Validated against torch reference at C_in=C_out=4, K=4, stride=2, padding=1 (typical ISTFTNet 2x temporal upsample) within 1e-4.
- **5 new test binaries**, all passing: `test_pffft_smoke`, `test_kernel_fft` (4 cases), `test_kernel_stft`, `test_kernel_lstm`, `test_kernel_conv1d_transpose`. Cactus's existing `test_kernel` (19 cases) still passes — zero regressions.
- **3 reference-fixture generators** at `tests/fixtures/{fft_reference,lstm_reference,conv_reference}.py`. Generated `.bin` files live in gitignored `tests/fixtures/data/`. Deterministic seeds (`np.random.seed(42)`, `torch.manual_seed(0)`, `torch.manual_seed(1)`).

## Numbers

- **13 commits** on `voice-loop` ahead of `main` (2 docs + 11 kernel/build/test commits).
- **8 new test cases** in 5 binaries, all green.
- **0 regressions** (existing `test_kernel` 19/19 still passes).
- **PFFFT vendored:** ~190 KB total (5 files), pristine.
- **New cactus code:** ~600 lines across kernel + test files (excluding fixture generators).

## Deferred (deliberate, per spec)

- **NEON optimization** of LSTM cell and ConvTranspose1d — defer until profiling proves they matter. They're called at hundreds-per-second in Kokoro's predictor stage, not the hot path. If a future profile flags them, replace inner accumulation loops with cactus's existing `cactus_gemv_*` kernels.
- **INT4/INT8 quantized kernel variants** — Plan 2 quantizes Kokoro weights first; if profile shows kernel quantization buys more than the model quantization already does, add then.
- **Streaming STFT/ISTFT API** — not needed by Kokoro (operates on full sequences). Add only if a future model needs it.

## Known minor items (not blocking)

- **PFFFT_Setup leak on `std::bad_alloc`** — if the temp `std::vector<float>` allocation throws between `pffft_new_setup` and `pffft_destroy_setup`, the setup leaks. Realistically harmless (n is small) but a trivial RAII guard would close it. Deferred until a Plan 2 cleanup pass.
- **Hann window recomputed per call** in `cactus_stft`/`cactus_istft`. Can be cached if profiled hot. Not on the critical path of any current call site.
- **Git committer identity** auto-resolves to `Shraey <shraey@Shraeys-MBP-2.modem>` (no `git config --global user.email` set on this machine). Harmless locally; PRs to upstream will look mildly off until configured.

## Test verification commands

```bash
cd /Users/shraey/.superset/worktrees/cactus/spec-decoding
cd cactus && ./build.sh                          # rebuild library
cd ../tests/build && cmake .. >/dev/null && make -j4
for t in test_pffft_smoke test_kernel_fft test_kernel_stft test_kernel_lstm \
         test_kernel_conv1d_transpose test_kernel; do
  echo "=== $t ==="; ./$t
done
```

## Next: Plan 2 — Kokoro Model Integration

Kokoro forward pass (text encoder, predictor, ISTFTNet decoder) on top of these kernels. KokoroConverter (.pth → INT8 ~85MB), bundled CMUdict G2P, cactus_tts FFI, `cactus run-tts` CLI. End-to-end gate: 5 phrases match kokoro-onnx within 2dB spectral diff.

See: `docs/superpowers/plans/2026-05-12-voice-loop-plan-2-kokoro-integration.md`
