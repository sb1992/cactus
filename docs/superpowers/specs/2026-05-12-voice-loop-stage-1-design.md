# Voice Loop — Stage 1 Design (half-duplex push-to-talk)

**Date:** 2026-05-12
**Branch:** `spec-decoding` (placeholder; will move to a dedicated `voice-loop` branch on implementation)
**Status:** Design approved by user, pending spec review
**Related:** cactus-compute/cactus issue #568 (TTS support + unified model API)

---

## Goal

Close the on-device voice loop in cactus end-to-end: microphone → STT → LLM → TTS → speaker, with a working macOS demo that proves the whole loop runs without cloud dependencies.

Stage 1 is half-duplex push-to-talk only. Full-duplex (AEC, barge-in, always-on listening) is explicitly Stage 2.

## Why now

- Cactus already has the input half: streaming STT (Whisper / Parakeet / Moonshine), Silero VAD wired up, LLM token streaming via callback. What's missing is the output half: TTS model, audio I/O abstraction, and the orchestration that ties them together.
- No competing project ships a fully-on-device, fully-open-source, fully-conversational voice assistant for Apple Silicon. Apple Intelligence's LLM tier still routes to Private Cloud Compute. Sesame Maya is on-device-capable but a cloud product. This is a category gap cactus can fill.
- Issue #568 surfaces the user demand for a unified `CactusModel` interface and TTS support — partial overlap with this spec, not the same scope.

## What's in scope

A new `cactus_voice_session` C++ orchestrator in cactus core, exposed via FFI, plus a Kokoro-82M TTS model integration, plus a macOS SwiftUI demo app that proves the loop closes end-to-end.

## What's out of scope (deferred to later stages)

- iOS app target — Stage 1.5 (same Swift wrapper, second app target)
- AEC / barge-in / always-on listening — Stage 2
- Wake word detection — Stage 3
- Android + Kotlin wrapper — Stage N
- Cloud TTS fallback — never (would break the on-device promise)
- Voice cloning — out of scope until clear use case
- Multilingual TTS — Stage 2+ (would require switching G2P to espeak-ng with GPL-3 license cost or a learned phonemizer)
- Speech-to-speech models like Sesame CSM — different architecture, separate spec

## Architecture

```
┌─────────────────── macOS Demo (Swift, ~300 LOC) ──────────────────┐
│                                                                    │
│  AVAudioEngine (input)                  AVAudioEngine (output)     │
│  ▼ render callback (RT thread)          ▲ scheduleBuffer           │
│  PCM frames (Float32, 16kHz mono)       PCM frames                 │
│  ▼                                      ▲                          │
│  ┌── SPSC lock-free queue ──┐           ┌── SPSC lock-free queue ──┐
│  │  cactus_voice_feed_pcm() │           │  cactus_voice_read_pcm() │
│  └──────────────┬───────────┘           └─────────────▲────────────┘
│                 │                                     │             │
└─────────────────┼─────────────────────────────────────┼─────────────┘
                  ▼                                     │
   ┌──────────────────────  cactus core (C++)  ────────────────────┐
   │                                                                │
   │   cactus_voice_session                                         │
   │   ┌──────────────────────────────────────────────────────────┐ │
   │   │  State machine: idle → listening → transcribing →        │ │
   │   │                 thinking → speaking → idle                │ │
   │   │                                                           │ │
   │   │  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌────────┐ │ │
   │   │  │ VAD  │──▶│ STT  │──▶│ LLM  │──▶│ TTS  │──▶│ Output │ │ │
   │   │  │Silero│   │Whisp │   │LFM2  │   │Kokoro│   │ queue  │ │ │
   │   │  └──────┘   │  or  │   │ 1.2B │   │ 82M  │   └────────┘ │ │
   │   │             │Parakt│   └──────┘   └──────┘               │ │
   │   │             └──────┘                                      │ │
   │   │                                                           │ │
   │   │  Sentence-aware token aggregator between LLM and TTS      │ │
   │   └──────────────────────────────────────────────────────────┘ │
   │                                                                │
   │   Existing: kernel thread pool (NEON/SME2), KV cache, etc.     │
   └────────────────────────────────────────────────────────────────┘

Callbacks fired to wrapper:
  • on_state(state)           — state machine transitions
  • on_partial_transcript(s)  — STT streaming output
  • on_llm_token(s)           — LLM token (for live "subtitles")
  • on_error(code, msg)
```

**Key principles:**

- The wrapper owns mic and speaker; cactus owns inference and sequencing.
- Two SPSC lock-free queues (mic-in, speaker-out) are the only shared state between the wrapper's RT audio threads and the cactus session thread.
- Callbacks fire on the cactus session thread, never on the audio RT thread, so wrappers can update UI freely from them.
- Existing cactus kernel thread pool is reused for STT/LLM/TTS forward passes — no new pool, sequenced through the state machine.
- Push-to-talk semantics enforced by the wrapper: the wrapper feeds mic PCM only while the user holds the button, and calls `cactus_voice_user_turn_end()` on release to commit the turn.

## C++ session API surface

```c
// cactus/ffi/cactus_voice.h

typedef struct cactus_voice_session cactus_voice_session;

typedef enum {
    CACTUS_VOICE_IDLE,
    CACTUS_VOICE_LISTENING,      // VAD detecting speech
    CACTUS_VOICE_TRANSCRIBING,   // STT running (overlaps with LISTENING via streaming)
    CACTUS_VOICE_THINKING,       // LLM running
    CACTUS_VOICE_SPEAKING,       // TTS emitting PCM
} cactus_voice_state;

typedef struct {
    const char* stt_model_path;       // e.g. ".../whisper-small-q4.bin" or NULL → parakeet
    const char* llm_model_path;       // e.g. ".../lfm2.5-1.2b-instruct-q4.bin"
    const char* tts_model_path;       // e.g. ".../kokoro-82m-q8.bin"
    const char* system_prompt;        // optional; NULL → built-in default
    const char* tts_voice;            // e.g. "af_bella"; NULL → default voice
    int input_sample_rate;            // mic rate the wrapper will feed (e.g. 48000)
    int output_sample_rate;           // speaker rate the wrapper expects (e.g. 48000)
    bool enable_vad;                  // true = use Silero, false = always-on listening
    float vad_threshold;              // 0.0–1.0, default 0.5
} cactus_voice_config;

typedef struct {
    void (*on_state)(void* user, cactus_voice_state s);
    void (*on_partial_transcript)(void* user, const char* utf8);
    void (*on_final_transcript)(void* user, const char* utf8);
    void (*on_llm_token)(void* user, const char* utf8);
    void (*on_llm_done)(void* user, const char* full_text);
    void (*on_error)(void* user, int code, const char* msg);
} cactus_voice_callbacks;

// Lifecycle
cactus_voice_session* cactus_voice_session_create(
    const cactus_voice_config* cfg,
    const cactus_voice_callbacks* cbs,
    void* user);
void cactus_voice_session_destroy(cactus_voice_session*);

// Audio I/O — wait-free, RT-safe to call from audio render callbacks
int cactus_voice_feed_pcm(cactus_voice_session*, const float* samples, int n_frames);
int cactus_voice_read_pcm(cactus_voice_session*, float* out_samples, int n_frames);

// Control
void cactus_voice_session_start(cactus_voice_session*);
void cactus_voice_session_stop(cactus_voice_session*);    // graceful: drain TTS, then idle
void cactus_voice_session_reset(cactus_voice_session*);   // hard: drop everything
void cactus_voice_user_turn_end(cactus_voice_session*);   // commit current user turn (push-to-talk)

// Optional: text-only injection for testing without a mic
void cactus_voice_inject_user_turn(cactus_voice_session*, const char* utf8);
```

**API design rationale:**

- Sample rates configured once at session creation, not per call. Wrapper picks the rate matching its audio engine; cactus internally resamples to whatever each model needs (16 kHz for VAD/STT, 24 kHz native for Kokoro). One conversion per model boundary, none in the hot RT path.
- PCM as `float* + n_frames` matches `AVAudioPCMBuffer`'s `floatChannelData`, avoids int16↔float conversion on the wrapper side.
- Callbacks fire on the session thread, never on the audio RT thread. Wrappers can update UI, log, or do any work in callbacks.
- `inject_user_turn` is a back door for tests and for users who want to drive the LLM by typing instead of talking. Cheap to add, big debugging value.
- No `set_voice` or `set_system_prompt` mid-session — config is immutable for v1. To change voice or prompt: destroy + create. Avoids race conditions.

## TTS model integration (Kokoro-82M)

**Model overview:**
- StyleTTS2-derived: phonemizer (G2P) → text encoder → predictor (duration + F0 + N) → ISTFTNet decoder → 24 kHz mono PCM
- 82M params, ~85 MB INT8, Apache 2.0
- Reference impls: `mlx-audio` (MLX, Python), `kokoro-onnx` (ONNX runtime), `hexgrad/Kokoro-82M` (HF original)

**Op-by-op cactus support:**

| Stage | Op | Cactus today | New work needed |
|---|---|---|---|
| G2P | misaki phonemizer (Python) | No | Dictionary lookup + L2S fallback in C++ |
| Text encoder | Small transformer | Yes | Minimal |
| Style predictor | LSTM + small MLPs | Partial | LSTM cell op (NEON) |
| Duration / F0 / N predictors | conv1d + LSTM | Partial | Depthwise conv1d if missing |
| ISTFTNet decoder | Upsampling convs + ISTFT | No | 1D transposed conv (NEON), STFT/iSTFT (NEON FFT) |

**Two genuinely new pieces:**

1. **G2P (grapheme-to-phoneme).** Decision: ship a CMU-style pronunciation dictionary (~5 MB asset) plus simple letter-to-sound fallback for OOV words. Sufficient for English assistant phrases. Acknowledged compromise: weird/rare words (proper nouns, technical terms) may mispronounce. Documented as a known v1 limitation.
   - Rejected: embedding espeak-ng (GPL-3 license would taint cactus).
   - Rejected: requiring callers to provide phonemes (useless for end users).

2. **ISTFTNet vocoder kernels.** Decision: integrate PFFFT (single-file, BSD-3, NEON-optimized) for the FFT op. Wrap as a cactus FFT kernel. ISTFTNet ends with an inverse STFT — needs FFT.
   - Rejected: writing our own NEON FFT (premature optimization for v1).

**Model packaging:**
- Convert Kokoro `.pth` → cactus binary format via the existing `python/src/converter.py` pipeline. Add a `KokoroConverter` class. Quantize per-tensor groupwise to INT8.
- Final asset: ~85 MB `kokoro-82m-q8.bin` + ~5 MB `kokoro-g2p.dict` + ~1 MB voice embedding (one default voice bundled; additional voices ship as separate asset files users can load on demand).

**Streaming output design:**
- Sentence aggregator buffers LLM tokens until `.!?` followed by whitespace, OR 80 tokens, whichever first.
- On flush: synthesize that sentence end-to-end (Kokoro's StyleTTS2 architecture does not naturally support sub-sentence streaming), push 24 kHz Float32 PCM into a resampler (→ `output_sample_rate` from config), then into the output ring buffer.
- Resampler: linear interpolation is sufficient for 24 k → 48 k upsampling.

**Realistic timeline (TTS work alone, before any session/orchestration code):**
- PFFFT integration + cactus FFT op wrapper: 2 days
- ISTFTNet kernels (transposed conv1d if missing, ISTFT op): 4–5 days
- LSTM cell + style/duration/F0 predictors: 2–3 days
- Converter + INT8 quantization: 2 days
- G2P dictionary + L2S fallback: 2–3 days
- End-to-end synthesis test (text in, WAV out, validation against `kokoro-onnx` reference): 1 day
- **Subtotal: ~2.5 weeks for one engineer.**

**Risk:** ISTFT correctness is the most likely place for "it generates audio but it sounds wrong" debugging. Mitigation: validate against `kokoro-onnx` reference output frame-by-frame for a 5-phrase test set before declaring kernels done.

## State machine

```
                  cactus_voice_session_start()
                            │
                            ▼
                       ┌─────────┐
                       │  IDLE   │◀────────────────────────┐
                       └────┬────┘                          │
       VAD detects speech   │                               │
       (or always-on if     │                               │
        VAD disabled)       │                               │
                            ▼                               │
                       ┌────────────┐                       │
                       │ LISTENING  │                       │
                       └────┬───────┘                       │
       streaming STT emits  │                               │
       partial transcripts  │                               │
                            ▼                               │
                       ┌──────────────┐                     │
                       │TRANSCRIBING  │                     │
                       └────┬─────────┘                     │
       VAD silence /        │                               │
       user_turn_end()      │ on_final_transcript           │
                            ▼                               │
                       ┌──────────┐                         │
                       │ THINKING │                         │
                       └────┬─────┘                         │
       LLM emits tokens →   │                               │
       sentence aggregator  │                               │
       fires first sentence │ on_llm_token (per token)      │
                            ▼                               │
                       ┌──────────┐                         │
                       │ SPEAKING │                         │
                       └────┬─────┘                         │
       output queue        │                                │
       fully drained &     │ on_llm_done (when LLM finishes)│
       LLM done            │                                │
                           └────────────────────────────────┘
```

**Critical sequencing detail — `THINKING` and `SPEAKING` overlap:** Once the first sentence is ready from the aggregator, state transitions to `SPEAKING` and TTS starts synthesizing — but the LLM keeps generating in parallel. New sentences from the LLM are queued behind the current TTS job. State stays `SPEAKING` until both:
1. The LLM emits EOS (or hits max tokens), AND
2. The output ring buffer is fully drained AND no TTS job is in flight.

**Concurrency model:**

The session owns **one dedicated session thread** that runs the state machine and is where all wrapper callbacks fire from. Heavy compute (STT/LLM/TTS forward passes, VAD inference, kernel ops) is dispatched from the session thread to **cactus's existing kernel thread pool**. The session thread is event-driven: it wakes on (a) new mic PCM arriving in the input ring, (b) a kernel task completing, (c) the output ring needing more PCM, (d) a control API call.

Three logical workers, all multiplexed on the session thread, dispatching to the kernel pool for compute:

| Worker | Purpose | Active when |
|---|---|---|
| `audio_in_worker` | Drains mic ring buffer, dispatches Silero VAD on 30 ms windows, hands speech segments to STT | LISTENING / TRANSCRIBING |
| `inference_worker` | Dispatches whichever of STT or LLM is currently active | Sequentially — STT then LLM |
| `tts_worker` | Pulls completed sentences from aggregator, dispatches Kokoro synthesis, pushes PCM to output ring | Continuously while sentences in queue |

The `inference_worker` runs STT and LLM serially because they share kernel pool resources and overlapping them would slow both. The `tts_worker` can overlap with `inference_worker` because TTS at 82 M params is small enough to leave kernel-pool slack for ongoing LLM generation.

**Cancellation:** `cactus_voice_session_reset()` sets a `cancel_flag` checked at safe points: between LLM tokens, between TTS sentences, between STT chunks. Drops all in-flight work, drains both ring buffers, returns to `IDLE`. No mid-kernel-call cancellation — caller may wait up to one inference step (~50 ms LLM token, ~150 ms TTS sentence) for reset to complete.

**Error handling:**
- Model load failure during `_create()` — return `NULL`, log details.
- STT/LLM/TTS forward-pass failure — fire `on_error(code, msg)`, transition to `IDLE`, drop in-flight audio.
- Ring buffer overflow on input (wrapper feeding faster than VAD can drain) — drop oldest samples, fire `on_error` once per overflow event (rate-limited, not per-sample).
- Ring buffer underflow on output (wrapper draining faster than TTS produces) — wrapper plays silence; not an error.

## macOS demo app

Single-window SwiftUI app, ~300 LOC. Smallest thing that proves the loop closes end-to-end.

```
┌─────────────────────────────────────────────────┐
│  Cactus Voice Demo                       ─ ☐ ✕  │
├─────────────────────────────────────────────────┤
│  ╭─────────────────────────────────────────╮   │
│  │ User:                                    │   │
│  │ What's the weather like in Tokyo?        │   │
│  │                                          │   │
│  │ Assistant:                               │   │
│  │ I don't have real-time weather data,    │   │
│  │ but Tokyo in May is typically...        │   │
│  ╰─────────────────────────────────────────╯   │
│                                                 │
│  State: ● SPEAKING                              │
│                                                 │
│  ┌─────┐  ╔═══════════════════════════════╗     │
│  │ ⏺   │  ║ ▁▂▃▅▆█▆▅▃▂▁ (mic waveform) ║     │
│  └─────┘  ╚═══════════════════════════════╝     │
│  Hold to talk                                   │
└─────────────────────────────────────────────────┘
```

**Wrapper components (Swift):**

```swift
final class CactusVoiceSession {
    init(config: Config, callbacks: Callbacks)
    func feedPCM(_ buf: AVAudioPCMBuffer)
    func readPCM(into buf: AVAudioPCMBuffer)
    func start() / stop() / reset() / endUserTurn()
}

final class AudioPipeline {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    // input tap on engine.inputNode → session.feedPCM (RT thread, wait-free)
    // player scheduled buffer source → session.readPCM (RT thread, wait-free)
}

struct ContentView: View {
    @StateObject var vm: VoiceVM
    var body: some View { /* transcript, state indicator, push-to-talk button, waveform */ }
}

// VoiceVM @MainActor view model bridges callbacks → @Published state
```

**Lifecycle:**
1. App launch → load models from bundle (or download to `Application Support` on first run, ~500 MB total: STT + LLM + TTS).
2. `CactusVoiceSession.init(...)` with paths + callbacks.
3. Tap "Allow Microphone" if needed (`NSMicrophoneUsageDescription` in Info.plist).
4. Hold push-to-talk → `AudioPipeline` taps mic, feeds PCM via `session.feedPCM` at the AVAudioEngine render callback (RT thread, wait-free copy into C ring).
5. Release button → wrapper calls `cactus_voice_user_turn_end()`, state transitions to `THINKING`.
6. Callbacks fire → `VoiceVM` updates `@Published` → SwiftUI re-renders.

**What the demo proves:**
- C++ session state machine works end-to-end with real microphone audio.
- TTS kernels produce correct audio (you hear coherent speech).
- Latency is measurable in real conditions (`os_signpost` markers at each state transition; visible in Instruments).
- Lock-free ring buffers do not xrun under macOS's audio scheduler.
- Memory footprint over a 10-minute conversation stays bounded (no leaks in the session loop).

**Out of scope for the demo intentionally:** settings UI, model picker, voice picker, conversation persistence, markdown rendering. Hardcode for v1; anything that distracts from "does the loop close" is excluded.

## Testing strategy

**Unit tests (C++, GoogleTest in `tests/`):**
- `test_voice_kokoro.cpp` — synthesize 5 reference phrases, compare frame-by-frame against `kokoro-onnx` reference WAVs. Pass = ≤ 2 dB spectral diff. Catches kernel correctness regressions.
- `test_voice_g2p.cpp` — phonemizer dictionary lookup + L2S fallback against a 200-word fixture (mix of dictionary hits, OOV, punctuation, numbers).
- `test_voice_pffft.cpp` — round-trip FFT/iFFT on random buffers, ≤ 1e-5 max diff vs reference scipy output.
- `test_voice_ringbuf.cpp` — SPSC ring buffer under stress: producer + consumer threads, 10 s duration, no data loss, no torn reads.

**Integration tests (C++, `tests/test_voice_session.cpp`):**
- Inject synthetic user turn via `cactus_voice_inject_user_turn("What is two plus two?")`, expect: `THINKING` → `SPEAKING`, callbacks fire in correct order, output PCM has nonzero energy, completes within 5 s. No real mic, no real speaker.
- Same but feed a known WAV via `feed_pcm` → expect VAD/STT to produce expected transcript ("hello world" with ≤ 1 word edit distance), then proceed through the rest of the loop.
- Reset mid-stream test — start a turn, call `reset()` 100 ms in, verify state returns to `IDLE` within 200 ms and no leaked threads.
- Stress test — 100 user turns back-to-back, watch RSS, fail if memory grows > 10 MB.

**End-to-end test (manual, on macOS demo app):**
- Documented checklist in `apps/voice-demo/README.md`: launch app → grant mic → hold button → say each of 5 reference phrases → verify (a) transcript correct, (b) LLM response coherent, (c) audio plays back without glitches, (d) `os_signpost` traces in Instruments show state-transition latencies under target.
- **Latency target: mouth-to-ear (release-button → first speaker output) ≤ 800 ms median on M2/M3 Mac for "hello, what's your name?" prompt.** Demo's pass/fail criterion.

**Not tested at v1 (deferred):** cross-platform parity, long-running stability (> 10 min), adversarial input, perceptual MOS / quality eval.

**CI integration:**
- Unit tests in existing cactus CMake test suite — no new infra.
- Integration tests gated behind `CACTUS_TEST_VOICE=1` env var (avoids 500 MB model download on every CI run).
- Nightly job runs voice integration tests with cached models.

## File layout

```
cactus/
  ffi/
    cactus_voice.h               (NEW — public C API)
    cactus_voice.cpp             (NEW — FFI implementation, callback marshalling)
  voice/                         (NEW directory)
    session.h / session.cpp      (cactus_voice_session class, state machine, workers)
    sentence_aggregator.h/.cpp   (LLM-token → sentence chunker)
    ring_buffer.h                (SPSC lock-free, header-only)
  models/
    model_kokoro.h / model_kokoro.cpp   (NEW — Kokoro forward pass)
  kernel/
    kernel_fft.h / kernel_fft.cpp       (NEW — PFFFT wrapper)
    kernel_lstm.h / kernel_lstm.cpp     (NEW if not present — LSTM cell)
    kernel_conv1d_transpose.h/.cpp      (NEW if not present)

libs/
  pffft/                         (NEW — vendored single-file PFFFT, BSD-3)

python/src/
  converter.py                   (extend with KokoroConverter class)

tests/
  test_voice_session.cpp         (NEW)
  test_voice_kokoro.cpp          (NEW)
  test_voice_g2p.cpp             (NEW)
  test_voice_pffft.cpp           (NEW)
  test_voice_ringbuf.cpp         (NEW)

apps/                            (NEW directory)
  voice-demo/
    README.md
    Package.swift
    Sources/CactusVoice/
      CactusVoiceSession.swift
      AudioPipeline.swift
    Sources/VoiceDemo/
      VoiceDemoApp.swift
      ContentView.swift
      VoiceVM.swift

assets/                          (NEW or existing)
  kokoro-g2p.dict                (~5 MB)
```

## Known risks (carried into v1, accepted)

- **ISTFTNet kernel correctness.** Mitigation: frame-by-frame reference comparison against `kokoro-onnx`.
- **macOS demo doesn't expose iPhone latency risk.** Exposing iPhone latency is exactly Stage 1.5's job; surfacing it in Stage 1 would block shipping.
- **Dictionary-based G2P will mispronounce some proper nouns.** Acceptable for v1, documented in demo README.

## Estimated effort

- TTS kernels + Kokoro integration: ~2.5 weeks
- C++ session orchestrator + state machine + ring buffers: ~1.5 weeks
- macOS demo app + Swift wrapper: ~1 week
- Testing + integration + measurement: ~1 week
- **Total for one engineer: 5–6 weeks** to a shippable Stage 1.

## Decision log (what was considered and rejected)

| Decision | Picked | Rejected | Why |
|---|---|---|---|
| TTS model | Kokoro-82M | OuteTTS-1.0 | OuteTTS reuses LLM kernels (strategic) but quality below Kokoro and iPhone latency risk too high for Stage 2 path. |
| TTS streaming | Sentence-level | Token-level | Token-level requires crossfade engineering for ~150 ms gain; not worth the artifact risk. |
| Orchestration location | Cactus C++ core | Per-platform wrapper | Single implementation benefits all wrappers; ambitious choice but aligned with "category-creating product" framing. |
| G2P | Bundled dictionary + L2S fallback | espeak-ng wrapper | espeak-ng is GPL-3, taints cactus. |
| FFT | Vendored PFFFT | Custom NEON FFT | Premature optimization; PFFFT is BSD-3, single file, NEON-optimized. |
| Quantization | INT8 | INT4 | INT8 safer for TTS; INT4 deferred to Stage 1.5 if quality holds. |
| Demo platform | macOS first | iOS first | Faster dev loop; iOS in Stage 1.5. |
| Demo form | SwiftUI app | CLI binary | "Real product" feel for demos; CLI considered and rejected. |
| AEC | None (half-duplex only) | vpio integration | Stage 2 problem; half-duplex push-to-talk doesn't need AEC. |

## Acceptance criteria for "Stage 1 done"

1. `cactus_voice_session` C API compiles, links, and ships in cactus's existing build.
2. All unit + integration tests pass.
3. macOS demo app launches, runs the loop end-to-end through 5 reference phrases without crashes.
4. Median mouth-to-ear latency ≤ 800 ms on M2 / M3 Mac for "hello, what's your name?" prompt.
5. README documenting the API, the demo, and Stage 2 roadmap committed.
6. No regressions in existing cactus test suite.
