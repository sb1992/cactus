# PR-prep design: getting `voice-loop` ready for upstream cactus

**Date:** 2026-05-13
**Branch (start):** `voice-loop` at `c2e78d6f` (31 commits ahead of `upstream/main`, all on user's fork `sb1992/cactus`)
**Branches (end):** `tts-kernels` (PR#1) + `kokoro-tts` (PR#2, stacked)
**Status:** Design approved, awaiting spec self-review + user review before writing implementation plan
**Related:** [Plan 1 complete](../plans/2026-05-12-voice-loop-plan-1-tts-kernels-COMPLETE.md), [Plan 2 complete](../plans/2026-05-12-voice-loop-plan-2-kokoro-integration-COMPLETE.md)

---

## Goal

Take the working `voice-loop` branch (Plan 1 + Plan 2 fully shipped: TTS kernels + Kokoro-82M end-to-end) and prepare it for two stacked pull requests against `cactus-compute/cactus:main`, following cactus's conventions exactly. Stop at "ready to open" — do not actually open the PRs.

## Why

The branch currently violates several upstream conventions:
- No DCO sign-offs on any of 31 commits (cactus's `dco.yml` will hard-fail)
- Author email `shraey@Shraeys-MBP-2.modem` (local hostname; needs real GitHub email)
- Internal docs in commits (`docs/superpowers/specs/` + `docs/superpowers/plans/` artifacts shouldn't ship upstream)
- 9 headers under `cactus/models/kokoro/` (CONTRIBUTING.md says "one header per folder")
- Comments throughout new C++ (CONTRIBUTING.md says "avoid comments — make code read like English")
- 56 new voice tests invisible to upstream CI (`cpp.yml` only builds `test_stt`)
- Conventional-commits style (`feat:`, `fix:`) not used upstream — minor noise
- 17k+ line monolithic submission would exceed cactus's PR-size norms (median 294 lines, p90 2260)

## Decisions locked in (from clarifying-question round)

| Question | Decision |
|---|---|
| Opening move | **Open both PRs at once** (stacked). Skip the issue #568 pre-comment. Faster, accept higher upfront cleanup cost. |
| Commit granularity | **Heavy squash to ~5 + ~6 commits** across the two PRs. Matches upstream patterns. |
| Header consolidation | **Two headers**: `kokoro.h` (public surface — G2P, KokoroModel) + `kokoro_internal.h` (everything else). |
| Comment stripping | **Selective**: strip `// what` (narration of code) comments; keep `// why` (decisions, workarounds, layout quirks). |
| CI integration | **PR#1**: wire `test_kernel_*` + `test_pffft_smoke` into `cpp.yml`. **PR#2**: document in body, leave CI gating to maintainers (450 MB asset pipeline is too heavy for `cpp.yml`). |
| Branch organization | **Two branches stacked**: `tts-kernels` off `upstream/main`; `kokoro-tts` off `tts-kernels`. Standard GitHub stacked-PR pattern. |
| Working branch | **Stay on `voice-loop`** with backup branches at each phase. No separate working branch. |
| Email | **User-provided** at rewrite time (not yet given). Spec uses `<YOUR_EMAIL>` placeholder. |
| PR opening | **NOT automated**. Spec writes PR body markdown files; user runs `gh pr create` manually. |

## End-state branch shape

```
upstream/main
   │
   ├─── tts-kernels (PR #1 → cactus-compute/cactus:main)
   │      ~5 commits, signed-off, real email, no docs/superpowers/
   │      Adds:
   │        libs/pffft/ (vendored BSD-3)
   │        cactus/kernel/kernel_fft.{h,cpp}
   │        cactus/kernel/kernel_stft.{h,cpp}
   │        cactus/kernel/kernel_lstm.{h,cpp}        (alongside existing fp16)
   │        cactus/kernel/kernel_conv1d_transpose.{h,cpp}
   │        tests/test_pffft_smoke.cpp + test_kernel_*.cpp
   │        tests/fixtures/data/{fft,stft,lstm,convT}_*.bin (committed; ~80 KB)
   │        .github/workflows/cpp.yml (extended to build + run new tests)
   │
   └─── kokoro-tts (PR #2 → cactus-compute/cactus:main, stacked on tts-kernels)
          ~6 commits, signed-off, real email, no docs/superpowers/
          Adds (on top of PR #1's kernels):
            cactus/models/kokoro/kokoro.h           (public)
            cactus/models/kokoro/kokoro_internal.h  (private, all 7+ classes)
            cactus/models/kokoro/*.cpp              (per-class .cpp files preserved)
            cactus/ffi/cactus_tts.{h,cpp}
            python/src/converter_kokoro.py
            python/src/cli_run_tts.py + cli.py dispatch
            python/pyproject.toml (kokoro-onnx + soundfile dev deps)
            assets/kokoro/build_dict.py + kokoro-g2p.dict (2.1 MB CMUdict)
            tests/test_voice_*.cpp + test_cactus_tts_ffi.cpp
            tests/fixtures/kokoro_*_reference.py (regenerator scripts only;
                                                   .bin outputs gitignored)
```

## Workflow — six phases

Each phase ends with `git branch voice-loop-backup-phase<N>` so we can revert if anything later breaks.

### Phase 1: Branch hygiene (DCO + drop docs)

```bash
git branch voice-loop-backup-original
git config user.email "<YOUR_EMAIL>"
git config user.name "Shraey Bhatia"
git fetch upstream
git rebase --signoff upstream/main
# Interactive rebase: drop the docs/superpowers/ commits.
# Confirmed list at time of writing (4 commits — verify with
# `git log --oneline | grep -E "docs:.*(voice loop|Plan [12])"` on your
# rebased branch in case the count shifted):
git rebase -i upstream/main
#   change 'pick' → 'drop' for:
#     d36a079e docs: voice loop stage 1 design
#     a1a23e9f docs: voice loop implementation plans 1 and 2
#     54d81c12 docs: Plan 1 (TTS kernel foundation) complete
#     c2e78d6f docs: Plan 2 (Kokoro integration) complete
# Verify all 56 voice tests + Plan 1 kernel tests still pass
git branch voice-loop-backup-phase1
```

End state: `voice-loop` has ~26 commits, all DCO-signed, real author email.

### Phase 2: Code hygiene (headers + comments)

**2a. Header consolidation (9 → 2)**

Move all class definitions from the 9 individual `.h` files in `cactus/models/kokoro/` into:
- `kokoro.h` — `cactus::kokoro::G2P`, `cactus::kokoro::KokoroModel` (everything a public consumer of the model needs)
- `kokoro_internal.h` — `TextEncoder`, `Bert`, `Predictor`, `DecoderEncodeStage`, `Generator`, `weights_loader` namespace, all helpers (AdaIN1d, AdaLayerNorm, AdainResBlk1d, BiLSTM, conv helpers, Snake1d)

Mechanical merge: cat the 9 headers, dedupe `#pragma once` and `#include`s, organize to respect declaration order (forward-declare cross-referencing classes if needed). Per-class `.cpp` files stay split — the rule is "one header per folder", not "one file per folder."

Update each `.cpp`'s includes:
- `text_encoder.cpp`, `bert.cpp`, `predictor.cpp`, `decoder.cpp`, `generator.cpp` → `#include "kokoro_internal.h"`
- `model.cpp`, `g2p.cpp` → `#include "kokoro.h"` plus `#include "kokoro_internal.h"` for internals it composes
- `cactus/ffi/cactus_tts.cpp` → `#include "../models/kokoro/kokoro.h"`
- Test files: update accordingly

Delete the 9 old header files.

Verify: `cd cactus && ./build.sh` succeeds. All 56 voice tests pass.

**2b. Selective comment stripping**

Pass through every new `.h` and `.cpp` file (kokoro.h, kokoro_internal.h, all `cactus/models/kokoro/*.cpp`, `cactus/ffi/cactus_tts.cpp`, `cactus/kernel/kernel_{fft,stft,lstm,conv1d_transpose}.{h,cpp}`).

**Strip** (`// what` style):
- `// Loop over time frames`
- `// Apply Hann window`
- `// Compute matrix multiply`
- Section headers like `// ----- helpers -----` (unless they materially help readability)
- Restating-the-name comments: `// load_weights — loads weights from path`

**Keep** (`// why` style):
- PFFFT-vs-scipy layout explanations in `kernel_fft.cpp`
- AdaIN1d non-affine InstanceNorm note (v0_19.pth lacks .norm.weight/.bias)
- Snake1d formula reference
- Weight-norm folding rationale
- Reflection-pad asymmetry in generator
- LSTM gate-order note (PyTorch [i,f,g,o] vs cuDNN [i,f,o,g])
- Any inline comment citing a file:line in the kokoro reference (`# istftnet.py:340-381`)

After both 2a + 2b, run the full test suite: 56 voice + 19 existing `test_kernel` + new kernel tests. Fix any breakage. `git branch voice-loop-backup-phase2`.

### Phase 3: Branch split (`tts-kernels` + `kokoro-tts`)

**3a. Create `tts-kernels` off `upstream/main`**

```bash
git checkout -b tts-kernels upstream/main
git cherry-pick <Plan 1 commit range>
# 11 cherry-picks (the kernel work)
```

Squash to ~5 commits via `git rebase -i upstream/main`:
1. `vendor: add PFFFT (BSD-3) + wire into cactus library` — combines PFFFT vendor + CMake wiring + smoke test
2. `feat(kernel): cactus_fft_r2c / cactus_fft_c2r — real ↔ complex 1D FFT` — fixture generator + r2c + c2r + tests
3. `feat(kernel): cactus_stft / cactus_istft — Hann window + overlap-add` — STFT/ISTFT op + chirp test
4. `feat(kernel): cactus_lstm_cell — PyTorch-compatible scalar fp32 LSTM` — fixture + op + test
5. `feat(kernel): cactus_conv1d_transpose — PyTorch ConvTranspose1d` — fixture + op + test

Verify: build cactus + run `test_pffft_smoke`, `test_kernel_fft`, `test_kernel_stft`, `test_kernel_lstm`, `test_kernel_conv1d_transpose`. All pass. Existing `test_kernel` (19 tests) still passes.

**3b. Create `kokoro-tts` stacked on `tts-kernels`**

```bash
git checkout -b kokoro-tts tts-kernels
git cherry-pick <Plan 2 commit range>
# 17 cherry-picks (Kokoro work)
```

Squash to ~6 commits:
1. `build+assets: kokoro-onnx dev dep + bundled CMUdict G2P dictionary` — pyproject.toml + voices + dictionary
2. `feat(kokoro): G2P dictionary lookup + L2S fallback` — g2p.cpp + tests
3. `feat(converter): KokoroConverter — .pth → cactus binary with weight-norm folding` — full converter + round-trip tests
4. `feat(kokoro): text encoder + predictor + bert (ALBERT-tiny PLBERT pattern)` — three model classes + per-stage tests
5. `feat(kokoro): decoder (encode + decode blocks + Hi-Fi-GAN generator with Snake1d + iSTFT)` — decoder.cpp + generator.cpp + per-stage tests
6. `feat(kokoro+ffi): KokoroModel + cactus_tts FFI + run-tts CLI` — top-level model + FFI + Python CLI

Verify: build cactus + run all 56 voice tests + 7 ffi smoke. All pass.

`git branch voice-loop-backup-phase3`.

### Phase 4: CI wiring (PR#1 only)

Edit `.github/workflows/cpp.yml` on the `tts-kernels` branch.

Current (approximate — verify against actual file):
```yaml
- run: cmake -B build-ci && cmake --build build-ci --target test_stt
- run: ./tests/build-ci/test_stt
```

Change to:
```yaml
- run: cmake -B build-ci && cmake --build build-ci --target test_stt test_pffft_smoke test_kernel_fft test_kernel_stft test_kernel_lstm test_kernel_conv1d_transpose
- run: ./tests/build-ci/test_stt
- run: ./tests/build-ci/test_pffft_smoke
- run: ./tests/build-ci/test_kernel_fft
- run: ./tests/build-ci/test_kernel_stft
- run: ./tests/build-ci/test_kernel_lstm
- run: ./tests/build-ci/test_kernel_conv1d_transpose
```

**Fixture committing decision**: The kernel tests reference `tests/fixtures/data/*.bin` files which are currently gitignored. Total size for kernel-only fixtures (`fft_*`, `stft_*`, `lstm_*`, `convT_*`) is ~80 KB. Commit them (with .gitignore exception):
- Add to `.gitignore`: keep blanket `tests/fixtures/data/` ignore but add `!tests/fixtures/data/fft_*`, `!tests/fixtures/data/stft_*`, etc.
- Or simpler: commit them explicitly via `git add -f tests/fixtures/data/{fft,stft,lstm,convT}_*.bin` in the kernel commit.

Note in the commit message: "fixtures regenerable via `python tests/fixtures/{fft,lstm,conv}_reference.py`."

This is a separate `ci: add TTS kernel tests to cpp.yml + commit reference fixtures` commit on `tts-kernels` only. Do NOT cherry-pick this commit onto `kokoro-tts`.

Test the YAML by sanity-checking syntax (use `actionlint` if available; otherwise visual review).

### Phase 5: Push + write PR body files

```bash
git push origin tts-kernels
git push origin kokoro-tts
```

Write PR body files to `docs/superpowers/notes/` (workspace, NOT committed):
- `PR1_body.md` — content for the `tts-kernels` PR (template in Section 4 of design)
- `PR2_body.md` — content for the `kokoro-tts` PR (template in Section 4 of design)

Both bodies must include:
- Summary
- "Why this is its own PR" (PR#1)
- Stacked-on note (PR#2)
- Test summary
- DCO sign-off note
- Issue link (#568 in PR#2)
- Maintainer cc (@HenryNdubuaku, @yujonglee)

PR#2 specifically must address:
- Why Kokoro vs the Gemma 4 / Qwen TTS asked for in #568
- The 4-of-5 sanity-check vs strict-match audio comparison (INT8 round-boundary nuance)
- The CI gap (450 MB asset pipeline; recommend extending verify-pr.yml in follow-up)

### Phase 6: Open PRs (manual, NOT automated)

User runs:

```bash
gh pr create \
  --repo cactus-compute/cactus \
  --base main \
  --head sb1992:tts-kernels \
  --title "feat(kernel): TTS kernel foundation — FFT, STFT, LSTM cell, transposed conv1d" \
  --body-file docs/superpowers/notes/PR1_body.md \
  --draft

gh pr create \
  --repo cactus-compute/cactus \
  --base main \
  --head sb1992:kokoro-tts \
  --title "feat(kokoro): Kokoro-82M TTS in cactus C++ (stacked on #<PR1>)" \
  --body-file docs/superpowers/notes/PR2_body.md \
  --draft
```

Both opened as draft initially. Mark ready for review when satisfied.

## Risk mitigations

| Risk | Mitigation |
|---|---|
| Cherry-pick conflicts during branch split | Do them in small batches; verify build after each batch. Backup branch before each phase. |
| Header merge introduces compile errors (forward-declaration order, missing decls) | Build after the merge; fix incrementally. Reference original 9 headers in the backup branch. |
| Comment stripping accidentally removes load-bearing `// why` comments | Diff-review the strip pass; restore anything that documents an upstream-reference or non-obvious decision. |
| CI workflow YAML syntax error | actionlint locally if available; otherwise GitHub will surface it on push. Easy to fix follow-up commit. |
| Kernel test fixtures committed are large enough to bother reviewers | Total ~80 KB; precedent exists (`hotword.wav` 3.4 MB). Acceptable. |
| INT8 quantization "1-frame off" PR#2 phrases get questioned | PR body addresses this directly with table + explanation. Documented inline in `test_voice_predictor.cpp`. |
| Maintainer rejects Kokoro-vs-Gemma/Qwen mismatch | Address in PR#2 body; offer to discuss model choice. Worst case: close PR#2, leave PR#1 standing. |
| 17k-line PR#2 sticker shock | PR body acknowledges size, references precedent (Karen/needle PR). Open as draft to signal "feedback welcome before formal review." |
| Lost work during git rewrite | `voice-loop-backup-original` + per-phase backups. Worst case: `git reset --hard voice-loop-backup-original` and start over. |

## Pre-PR checklist (gate Phase 6)

- [ ] Both branches push cleanly to fork
- [ ] `git log upstream/main..tts-kernels --format='%(trailers:key=Signed-off-by)' | grep -c .` equals commit count
- [ ] `git log tts-kernels..kokoro-tts --format='%(trailers:key=Signed-off-by)' | grep -c .` equals commit count
- [ ] `git log kokoro-tts --format='%ae'` shows real email (not local hostname)
- [ ] All 56 voice tests pass on `kokoro-tts`
- [ ] Kernel tests + cpp.yml build target update verified locally on `tts-kernels`
- [ ] No `docs/superpowers/` files in either branch's diff vs `upstream/main`
- [ ] Exactly 2 headers under `cactus/models/kokoro/` (kokoro.h + kokoro_internal.h)
- [ ] No `// what` comments left in new C++ (sample-grep)
- [ ] PR body files written and re-read for typos
- [ ] `voice-loop-backup-original` exists in case of recovery need

## What's explicitly out of scope for this work

- Opening the PRs (manual step at the end; user does it)
- Pushing to upstream `cactus-compute/cactus` (only push to `sb1992/cactus` fork)
- Extending `verify-pr.yml` to cover Kokoro (mention as follow-up in PR#2 body; out of scope)
- Adding misaki phonemizer to the runtime path (current G2P+ARPAbet→IPA mapping is acceptable for v1)
- Speed parameter support (`speed != 1.0`) — out of scope, mention as follow-up
- INT4 quantization variant — out of scope
- Resolving the 1-frame off-by-one duration prediction — documented as INT8 quantization edge case
- iOS app target / voice session orchestrator (Plan 3 / Plan 4 from original spec) — separate future work

## Estimated effort

Single focused session: ~3-5 hours of implementer time.
- Phase 1 (branch hygiene): 30 min
- Phase 2 (header consolidation + comment strip): 1.5 hours
- Phase 3 (branch split + squash): 45 min
- Phase 4 (CI wiring): 30 min
- Phase 5 (push + PR bodies): 30 min
- Phase 6 (manual PR open): 5 min, user-driven

If anything goes wrong during the rewrite, recovery via backup branch adds ~10-15 min per incident.

## Acceptance criteria for "PR-ready"

1. Both branches push cleanly to `sb1992/cactus`
2. Pre-PR checklist all green
3. `gh pr create` commands ready to copy-paste
4. PR body files written and reviewed
5. User confirms they're ready to open the PRs (this work stops at "ready", does not open)
