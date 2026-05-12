# Voice Loop — Plan 2: Kokoro Model Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Kokoro-82M end-to-end inside cactus — text → 24 kHz PCM, frame-by-frame matching `kokoro-onnx` reference, INT8 quantized, exposed as a `cactus_tts` FFI + `cactus run-tts` CLI.

**Architecture:** Use the kernels from Plan 1. Add Python `KokoroConverter` that ingests upstream `.pth` checkpoints + voice embedding tensors and emits cactus's binary format. Add `Model` subclass for Kokoro composed of (text encoder, style predictor, duration/F0/N predictors, ISTFTNet decoder). Bundle CMUdict-derived dictionary + L2S fallback for English G2P.

**Tech Stack:** C++17 (extends Plan 1's kernel set), Python 3.10+ (for the converter), reference impl is `kokoro-onnx` (Apache 2.0). Validation via `kokoro-onnx` reference outputs.

**Spec:** [`docs/superpowers/specs/2026-05-12-voice-loop-stage-1-design.md`](../specs/2026-05-12-voice-loop-stage-1-design.md)

**Prerequisite:** Plan 1 complete (PFFFT vendored, FFT/STFT/LSTM/conv1d_transpose ops + tests passing).

**Definition of Done:** `cactus run-tts "Hello, my name is Cactus."` produces a `.wav` file that an automated test confirms matches `kokoro-onnx` output within ≤2 dB spectral diff for 5 reference phrases. INT8 weights file ≤ 100 MB. Dictionary G2P passes 200-word fixture. Full cactus test suite passes.

---

## Honest scope note

Several tasks below are "study reference + port to cactus pattern." Fabricating the exact layer code for Kokoro from memory would create plausible-looking but wrong code — Kokoro's StyleTTS2-derived architecture has specific dim configs (text encoder hidden, predictor hidden, decoder upsample factors) that need to be read out of the actual checkpoint or from `kokoro-onnx`'s model graph. For those tasks I provide:

- The exact reference URL + which file to study
- The concrete test that defines "done"
- Estimated time
- Common pitfalls

Tasks I can fully specify (G2P, converter, FFI, CLI) get full code as in Plan 1.

---

## File Structure

**New files:**
- `assets/kokoro/build_dict.py` — script that builds `kokoro-g2p.dict` from CMUdict
- `assets/kokoro/kokoro-g2p.dict` — bundled binary dictionary (~5 MB, generated)
- `assets/kokoro/voices/af_bella.bin` — bundled default voice embedding
- `cactus/models/model_kokoro.h` — Model subclass declaration
- `cactus/models/model_kokoro.cpp` — top-level forward, weight loading
- `cactus/models/kokoro/g2p.h`, `g2p.cpp` — dictionary G2P + L2S fallback
- `cactus/models/kokoro/text_encoder.h`, `text_encoder.cpp`
- `cactus/models/kokoro/predictor.h`, `predictor.cpp` — style + duration + F0 + N predictors
- `cactus/models/kokoro/decoder.h`, `decoder.cpp` — ISTFTNet decoder
- `cactus/ffi/cactus_tts.h`, `cactus_tts.cpp` — TTS FFI surface
- `python/src/converter_kokoro.py` — KokoroConverter (.pth → cactus binary, INT8 quantize)
- `python/src/cli_run_tts.py` — `cactus run-tts` subcommand
- `tests/test_voice_g2p.cpp`
- `tests/test_voice_kokoro.cpp` — end-to-end vs kokoro-onnx
- `tests/fixtures/kokoro_reference.py` — generates kokoro-onnx reference outputs for 5 phrases
- `tests/fixtures/g2p_words.txt` — 200-word G2P fixture (mix of dictionary hits + OOV)

**Modified files:**
- `cactus/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `python/src/converter.py` — register KokoroConverter
- `tools/src/cli.py` — register run-tts subcommand
- `python/pyproject.toml` (or equivalent) — add `kokoro-onnx`, `numpy`, `soundfile` to dev deps for fixture generation

---

## Task 1: Set up reference environment

**Files:**
- Modify: `python/pyproject.toml` (or wherever python deps live)

- [ ] **Step 1: Locate the dev dependencies file**

```bash
find python -name "pyproject.toml" -o -name "setup.py" -o -name "requirements-dev.txt" 2>/dev/null
```

Pick the file that defines dev/test deps.

- [ ] **Step 2: Add reference deps**

Add to dev/test deps (syntax depends on the file, e.g. for `pyproject.toml`):

```toml
[project.optional-dependencies]
dev = [
    "kokoro-onnx>=0.4.0",
    "soundfile>=0.12",
    "scipy>=1.11",
    "torch>=2.1",
    "huggingface-hub>=0.20",
]
```

- [ ] **Step 3: Install + smoke-test kokoro-onnx**

```bash
pip install -e ".[dev]"
python -c "from kokoro_onnx import Kokoro; print('ok')"
```

Expected: `ok`. If install fails, check Python version (kokoro-onnx requires 3.10+).

- [ ] **Step 4: Commit**

```bash
git add python/pyproject.toml
git commit -m "build: add kokoro-onnx + scientific deps for TTS fixture generation"
```

---

## Task 2: Generate Kokoro reference outputs

**Files:**
- Create: `tests/fixtures/kokoro_reference.py`

- [ ] **Step 1: Write the reference generator**

Create `tests/fixtures/kokoro_reference.py`:

```python
"""Generate kokoro-onnx reference outputs for cactus Kokoro tests.

Synthesizes 5 reference phrases with kokoro-onnx (Apache 2.0) and saves
both raw audio (WAV) and intermediate tensors (text encoder output, predictor
outputs, decoder output) for cactus to validate against.

Run: python tests/fixtures/kokoro_reference.py
Outputs: tests/fixtures/data/kokoro_*.bin and tests/fixtures/data/kokoro_*.wav
"""
import os, struct
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

OUT = os.path.join(os.path.dirname(__file__), "data")
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

# kokoro-onnx will download the model on first run (~330 MB). Cached after.
kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
voice = "af_bella"

for tag, text in REFERENCE_PHRASES:
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    assert sample_rate == 24000, f"unexpected sample rate {sample_rate}"
    sf.write(os.path.join(OUT, f"kokoro_{tag}.wav"), samples, sample_rate)
    write_f32(os.path.join(OUT, f"kokoro_{tag}_audio.bin"), samples)
    print(f"  {tag}: {len(samples) / sample_rate:.2f}s  '{text}'")

# Also save the voice embedding tensor for the converter's later use
import json
with open("voices.json") as f:
    voices = json.load(f)
voice_emb = np.array(voices[voice], dtype=np.float32)
write_f32(os.path.join(OUT, f"kokoro_voice_{voice}.bin"), voice_emb)

print(f"\nWrote fixtures to {OUT}")
print(f"Voice {voice} embedding: shape {voice_emb.shape}")
```

- [ ] **Step 2: Run it**

```bash
python tests/fixtures/kokoro_reference.py
```

Expected: downloads ~330 MB on first run; then prints 5 phrases with durations; creates 5 `.wav` + 5 `_audio.bin` + 1 `voice_af_bella.bin`. Runtime ~10–30s after model download.

- [ ] **Step 3: Sanity-listen to one phrase**

```bash
afplay tests/fixtures/data/kokoro_p01.wav   # macOS
# or: play tests/fixtures/data/kokoro_p01.wav   # if sox installed
```

Expected: clearly intelligible English voice saying "Hello, my name is Cactus." If garbled, kokoro-onnx is broken in your env — debug before continuing.

- [ ] **Step 4: Commit (generator only — data is gitignored)**

```bash
git add tests/fixtures/kokoro_reference.py
git commit -m "test: kokoro-onnx reference output generator (5 phrases)"
```

---

## Task 3: Build the CMU dictionary asset

**Files:**
- Create: `assets/kokoro/build_dict.py`

- [ ] **Step 1: Write the dictionary builder**

Create `assets/kokoro/build_dict.py`:

```python
"""Build kokoro-g2p.dict from CMUdict.

Downloads CMU Pronouncing Dictionary (public domain), normalizes to ARPAbet
(no stress markers — Kokoro doesn't use them), and emits a binary trie-friendly
format optimized for fast lookup at inference time.

Format (little-endian):
  uint32 magic = 0xCACDCACD
  uint32 version = 1
  uint32 n_words
  for each word:
    uint16 word_len
    bytes  word (ASCII lowercase, no nulls)
    uint16 phoneme_count
    bytes  phonemes (one byte per phoneme, indexed into phoneme_alphabet)
  uint16 alphabet_count
  for each phoneme:
    uint8  len
    bytes  phoneme_string

Run: python assets/kokoro/build_dict.py
Output: assets/kokoro/kokoro-g2p.dict
"""
import os, struct, urllib.request, re
from collections import OrderedDict

OUT = os.path.join(os.path.dirname(__file__), "kokoro-g2p.dict")
SRC = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

# 1. Download
print(f"Downloading CMUdict from {SRC} ...")
raw = urllib.request.urlopen(SRC).read().decode("utf-8")

# 2. Parse: each line is "WORD(N)? ph1 ph2 ph3 ..."
#    Strip stress digits (AA1 → AA), drop alt prons (keep only first).
seen = OrderedDict()
phoneme_alphabet = OrderedDict()
for line in raw.splitlines():
    line = line.strip()
    if not line or line.startswith(";;;"): continue
    parts = line.split(None, 1)
    if len(parts) != 2: continue
    word, prons = parts
    word = re.sub(r"\(\d+\)$", "", word).lower()
    if word in seen: continue
    phs = [re.sub(r"\d$", "", p) for p in prons.split()]
    seen[word] = phs
    for p in phs:
        if p not in phoneme_alphabet:
            phoneme_alphabet[p] = len(phoneme_alphabet)

print(f"  {len(seen)} words, {len(phoneme_alphabet)} phonemes")
assert len(phoneme_alphabet) <= 255, "phoneme alphabet must fit in uint8"

# 3. Write binary
with open(OUT, "wb") as f:
    f.write(struct.pack("<II", 0xCACDCACD, 1))
    f.write(struct.pack("<I", len(seen)))
    for word, phs in seen.items():
        wb = word.encode("ascii", errors="ignore")
        f.write(struct.pack("<H", len(wb)))
        f.write(wb)
        f.write(struct.pack("<H", len(phs)))
        f.write(bytes(phoneme_alphabet[p] for p in phs))
    f.write(struct.pack("<H", len(phoneme_alphabet)))
    for ph in phoneme_alphabet:
        pb = ph.encode("ascii")
        f.write(struct.pack("<B", len(pb)))
        f.write(pb)

print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1024 / 1024:.2f} MB)")
```

- [ ] **Step 2: Run it**

```bash
python assets/kokoro/build_dict.py
```

Expected:
```
Downloading CMUdict from ... (about 3 MB)
  ~135000 words, 39 phonemes
Wrote .../kokoro-g2p.dict (~5 MB)
```

- [ ] **Step 3: Commit (script + generated dict — dict is small enough to vendor)**

```bash
git add assets/kokoro/build_dict.py assets/kokoro/kokoro-g2p.dict
git commit -m "assets: bundled CMUdict-derived G2P dictionary for Kokoro"
```

---

## Task 4: G2P implementation (dictionary lookup)

**Files:**
- Create: `cactus/models/kokoro/g2p.h`
- Create: `cactus/models/kokoro/g2p.cpp`
- Create: `tests/fixtures/g2p_words.txt`
- Create: `tests/test_voice_g2p.cpp`
- Modify: `cactus/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the 200-word fixture**

Create `tests/fixtures/g2p_words.txt`. Layout: one word per line, then `\t`, then expected phonemes (space-separated). Mix dictionary hits + OOV. (The first ~150 should be common dictionary words; last ~50 are deliberate OOV testing the L2S fallback.)

```
hello	HH AH L OW
world	W ER L D
cactus	K AE K T AH S
the	DH AH
quick	K W IH K
brown	B R AW N
fox	F AA K S
jumps	JH AH M P S
over	OW V ER
lazy	L EY Z IY
dog	D AO G
... (continue for ~150 hits)
... (then ~50 OOV — leave the phoneme column empty for these; they exercise the L2S fallback whose output we'll regenerate from the actual implementation in step 6)
```

For brevity, build the fixture programmatically:

```bash
python <<'EOF'
import struct
words = ["hello","world","cactus","the","quick","brown","fox","jumps","over","lazy",
         "dog","cat","sing","read","write","computer","phone","screen","keyboard","mouse",
         # ... (you can copy any 200 common English words for the hits)
        ]
# Look up actual pronunciations in the dictionary we just built
import os
DICT = "assets/kokoro/kokoro-g2p.dict"
with open(DICT, "rb") as f:
    magic, ver = struct.unpack("<II", f.read(8))
    n_words, = struct.unpack("<I", f.read(4))
    table = {}
    for _ in range(n_words):
        wlen, = struct.unpack("<H", f.read(2))
        word = f.read(wlen).decode("ascii")
        plen, = struct.unpack("<H", f.read(2))
        ph_idx = list(f.read(plen))
        table[word] = ph_idx
    alpha_n, = struct.unpack("<H", f.read(2))
    alpha = []
    for _ in range(alpha_n):
        l, = struct.unpack("<B", f.read(1))
        alpha.append(f.read(l).decode("ascii"))
with open("tests/fixtures/g2p_words.txt", "w") as f:
    for w in words:
        if w in table:
            f.write(w + "\t" + " ".join(alpha[i] for i in table[w]) + "\n")
        else:
            f.write(w + "\t\n")  # OOV — L2S fallback fills these in step 6
EOF
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_voice_g2p.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kokoro/g2p.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct WordCase {
    std::string word;
    std::vector<std::string> expected_phonemes;  // empty = OOV (L2S fallback used; just check non-empty result)
};

static std::vector<WordCase> load_fixture() {
    std::vector<WordCase> out;
    std::ifstream f("tests/fixtures/g2p_words.txt");
    std::string line;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        WordCase wc{line.substr(0, tab), {}};
        if (tab != std::string::npos) {
            std::istringstream ps(line.substr(tab + 1));
            std::string p;
            while (ps >> p) wc.expected_phonemes.push_back(p);
        }
        out.push_back(wc);
    }
    return out;
}

class G2PTest : public ::testing::Test {
protected:
    cactus::kokoro::G2P g2p;
    void SetUp() override {
        ASSERT_TRUE(g2p.load("assets/kokoro/kokoro-g2p.dict"));
    }
};

TEST_F(G2PTest, DictionaryHits) {
    auto cases = load_fixture();
    int hits = 0, oov = 0;
    for (auto& wc : cases) {
        auto result = g2p.lookup(wc.word);
        if (!wc.expected_phonemes.empty()) {
            EXPECT_EQ(result, wc.expected_phonemes) << "word: " << wc.word;
            hits++;
        } else {
            EXPECT_FALSE(result.empty()) << "OOV fallback returned empty for: " << wc.word;
            oov++;
        }
    }
    std::cout << "Tested " << hits << " dictionary hits, " << oov << " OOV (L2S fallback)\n";
}
```

- [ ] **Step 3: Wire into test CMake**

Append to `tests/CMakeLists.txt`:

```cmake
add_executable(test_voice_g2p test_voice_g2p.cpp)
target_link_libraries(test_voice_g2p cactus gtest gtest_main)
add_test(NAME VoiceG2P COMMAND test_voice_g2p WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

- [ ] **Step 4: Build — expect failure**

```bash
cmake --build build --target test_voice_g2p 2>&1 | tail -5
```

Expected: missing `kokoro/g2p.h`.

- [ ] **Step 5: Write the header**

Create `cactus/models/kokoro/g2p.h`:

```cpp
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace cactus { namespace kokoro {

class G2P {
public:
    // Load binary dictionary (format defined in assets/kokoro/build_dict.py).
    bool load(const std::string& dict_path);

    // Look up phonemes for a single word.
    //   Returns dictionary phonemes if found.
    //   Falls back to letter-to-sound rules for OOV.
    //   Word is normalized to lowercase + ASCII before lookup.
    std::vector<std::string> lookup(const std::string& word) const;

    // Tokenize a sentence into phonemes (whitespace-split, then per-word lookup,
    // with sentence-end punctuation preserved as special tokens "." "!" "?").
    std::vector<std::string> phonemize(const std::string& text) const;

private:
    std::unordered_map<std::string, std::vector<uint8_t>> table_;
    std::vector<std::string> alphabet_;

    // L2S fallback for OOV. Implemented as a small set of letter-pattern rules
    // (see g2p.cpp). Best-effort; documented as imperfect for proper nouns.
    std::vector<std::string> l2s_fallback(const std::string& word) const;
};

}}  // namespace
```

- [ ] **Step 6: Write the implementation**

Create `cactus/models/kokoro/g2p.cpp`:

```cpp
#include "g2p.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace cactus { namespace kokoro {

bool G2P::load(const std::string& dict_path) {
    std::ifstream f(dict_path, std::ios::binary);
    if (!f) return false;

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0xCACDCACDu || version != 1) return false;

    uint32_t n_words;
    f.read(reinterpret_cast<char*>(&n_words), 4);
    for (uint32_t i = 0; i < n_words; ++i) {
        uint16_t wlen; f.read(reinterpret_cast<char*>(&wlen), 2);
        std::string word(wlen, '\0');
        f.read(word.data(), wlen);
        uint16_t plen; f.read(reinterpret_cast<char*>(&plen), 2);
        std::vector<uint8_t> phs(plen);
        f.read(reinterpret_cast<char*>(phs.data()), plen);
        table_.emplace(std::move(word), std::move(phs));
    }
    uint16_t alpha_n;
    f.read(reinterpret_cast<char*>(&alpha_n), 2);
    alphabet_.reserve(alpha_n);
    for (uint16_t i = 0; i < alpha_n; ++i) {
        uint8_t l; f.read(reinterpret_cast<char*>(&l), 1);
        std::string ph(l, '\0');
        f.read(ph.data(), l);
        alphabet_.push_back(std::move(ph));
    }
    return f.good() || f.eof();
}

static std::string normalize(const std::string& w) {
    std::string out;
    out.reserve(w.size());
    for (char c : w) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
    }
    return out;
}

std::vector<std::string> G2P::lookup(const std::string& word) const {
    auto norm = normalize(word);
    if (norm.empty()) return {};
    auto it = table_.find(norm);
    if (it == table_.end()) return l2s_fallback(norm);
    std::vector<std::string> out;
    out.reserve(it->second.size());
    for (uint8_t idx : it->second) out.push_back(alphabet_[idx]);
    return out;
}

// Minimal English letter-to-sound rules. Not winning any pronunciation contests
// but covers enough of the common cases (consonants, common vowel patterns) that
// OOV proper nouns are at least intelligible. Documented limitation in the spec.
std::vector<std::string> G2P::l2s_fallback(const std::string& word) const {
    static const std::unordered_map<char, const char*> simple = {
        {'b',"B"},{'c',"K"},{'d',"D"},{'f',"F"},{'g',"G"},{'h',"HH"},{'j',"JH"},
        {'k',"K"},{'l',"L"},{'m',"M"},{'n',"N"},{'p',"P"},{'q',"K"},{'r',"R"},
        {'s',"S"},{'t',"T"},{'v',"V"},{'w',"W"},{'x',"K S"},{'y',"Y"},{'z',"Z"},
        {'a',"AE"},{'e',"EH"},{'i',"IH"},{'o',"AA"},{'u',"AH"},
    };
    std::vector<std::string> out;
    for (size_t i = 0; i < word.size(); ++i) {
        char c = word[i];
        // Tiny digraph table (priority over single-letter)
        if (i + 1 < word.size()) {
            std::string two{c, word[i+1]};
            if (two == "th") { out.push_back("TH"); ++i; continue; }
            if (two == "sh") { out.push_back("SH"); ++i; continue; }
            if (two == "ch") { out.push_back("CH"); ++i; continue; }
            if (two == "ng") { out.push_back("NG"); ++i; continue; }
            if (two == "ph") { out.push_back("F");  ++i; continue; }
            if (two == "qu") { out.push_back("K"); out.push_back("W"); ++i; continue; }
        }
        auto it = simple.find(c);
        if (it != simple.end()) {
            // Some entries contain two phonemes separated by space (e.g. 'x' → "K S")
            std::string s = it->second;
            size_t sp = s.find(' ');
            if (sp == std::string::npos) out.push_back(s);
            else { out.push_back(s.substr(0, sp)); out.push_back(s.substr(sp + 1)); }
        }
        // unknown chars silently dropped (numbers, punctuation handled by phonemize)
    }
    return out;
}

std::vector<std::string> G2P::phonemize(const std::string& text) const {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            auto ph = lookup(cur);
            for (auto& p : ph) out.push_back(p);
            cur.clear();
        }
    };
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'') {
            cur.push_back(c);
        } else {
            flush();
            if (c == '.' || c == '!' || c == '?') {
                out.push_back(std::string(1, c));
            }
        }
    }
    flush();
    return out;
}

}}  // namespace
```

- [ ] **Step 7: Wire into cactus library build**

Append to `cactus/CMakeLists.txt` source list:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/models/kokoro/g2p.cpp
```

- [ ] **Step 8: Build + run**

```bash
cmake --build build --target test_voice_g2p && ./build/tests/test_voice_g2p
```

Expected: `[  PASSED  ] 1 test.` Output line shows roughly "Tested ~150 dictionary hits, ~50 OOV (L2S fallback)".

- [ ] **Step 9: Commit**

```bash
git add cactus/models/kokoro/g2p.h cactus/models/kokoro/g2p.cpp \
        tests/test_voice_g2p.cpp tests/fixtures/g2p_words.txt \
        cactus/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(kokoro): G2P dictionary lookup + L2S fallback"
```

---

## Task 5: KokoroConverter (Python — .pth → cactus binary, INT8 quantize)

**Files:**
- Create: `python/src/converter_kokoro.py`
- Modify: `python/src/converter.py`

This task ports Kokoro's checkpoint into cactus's binary format. The exact tensor names depend on the upstream checkpoint — this requires reading the checkpoint to discover the structure.

- [ ] **Step 1: Inspect the upstream checkpoint structure**

```bash
python <<'EOF'
import torch
ckpt = torch.load("kokoro-v0_19.pth", map_location="cpu", weights_only=True)
# Print top-level keys + tensor shapes
def walk(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, prefix + "/" + k if prefix else k)
    elif hasattr(d, "shape"):
        print(f"{prefix}\t{tuple(d.shape)}\t{d.dtype}")
walk(ckpt)
EOF
```

Expected: prints a tree of tensor names + shapes. **Save this output** — you'll reference it in tasks 6, 7, 8 (text encoder, predictor, decoder).

- [ ] **Step 2: Write the converter skeleton**

Create `python/src/converter_kokoro.py`. The pattern follows existing converters in `python/src/converter.py` (study the existing `LFM2Converter` or `GemmaConverter` for the pattern). Key responsibilities:

```python
"""Kokoro-82M model converter.

Loads kokoro-v0_19.pth from upstream HF (hexgrad/Kokoro-82M),
maps tensor names to cactus's expected layout, applies per-tensor
groupwise INT8 quantization, writes to cactus binary format.

Compatible with the cactus Model loading path in cactus/models/model.h.
"""
from .converter import BaseConverter, write_int8_groupwise

class KokoroConverter(BaseConverter):
    MODEL_TYPE = "kokoro"
    EXPECTED_HIDDEN_DIM = 512  # confirm from inspection in Step 1

    # Map from upstream tensor name → cactus tensor name.
    # Fill in based on Step 1 inspection output. Pattern:
    NAME_MAP = {
        # "bert.embeddings.word_embeddings.weight": "text_encoder.embed.weight",
        # "predictor.lstms.0.weight_ih_l0": "predictor.lstm0.W_ih",
        # ... (port systematically; one mapping per tensor)
    }

    def convert(self, ckpt_path: str, out_path: str):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        flat = self._flatten(ckpt)
        renamed = {self.NAME_MAP.get(k, k): v for k, v in flat.items()}
        quantized = {k: self._quantize_tensor(k, v) for k, v in renamed.items()}
        self._write_cactus_binary(quantized, out_path)

    def _quantize_tensor(self, name, tensor):
        # Per-tensor groupwise INT8 quant, group_size=128
        # See LFM2Converter for the proven pattern; copy that and adapt.
        # Skip quantization for: layer norm scales/biases, position embeddings,
        # any tensor with < 1024 elements (overhead exceeds savings).
        ...

    @staticmethod
    def _flatten(d, prefix=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(KokoroConverter._flatten(v, prefix + "/" + k if prefix else k))
        elif hasattr(d, "shape"):
            out[prefix] = d
        return out
```

**This is genuinely incremental work.** Estimated time: 1.5–2 days. You will:
- Run Step 1 inspection
- Compare to `kokoro-onnx`'s model graph (clone the repo, read `kokoro_onnx/model.py`)
- Map every tensor name 1:1
- Test by quantizing a single weight tensor and running `cactus_dequant_int8` to verify round-trip max-abs-error < 5%

- [ ] **Step 3: Register in `converter.py`**

Add to `python/src/converter.py` registration table:

```python
from .converter_kokoro import KokoroConverter

CONVERTERS["kokoro"] = KokoroConverter
```

- [ ] **Step 4: Acceptance test for the converter**

Add to the converter's existing test file:

```python
def test_kokoro_converter_round_trip():
    """Convert + load + dequant should preserve original tensors within INT8 noise."""
    import torch, numpy as np
    from cactus.python.src.converter_kokoro import KokoroConverter
    converter = KokoroConverter()
    converter.convert("kokoro-v0_19.pth", "/tmp/kokoro-q8.bin")

    # Load the binary back, dequantize, compare to original
    original = torch.load("kokoro-v0_19.pth", weights_only=True)
    loaded = load_cactus_binary("/tmp/kokoro-q8.bin")  # helper in cactus python

    for name, orig in walk_tensors(original):
        cactus_name = KokoroConverter.NAME_MAP.get(name, name)
        deq = loaded[cactus_name]  # auto-dequantized on load
        rel_err = (orig - deq).abs().max() / orig.abs().max()
        assert rel_err < 0.05, f"{name}: rel_err={rel_err}"
```

- [ ] **Step 5: Run the converter**

```bash
python -m cactus.python.src.converter_kokoro \
    --input kokoro-v0_19.pth \
    --output kokoro-82m-q8.bin
```

Expected: produces ~85 MB binary file. If size is wrong: check whether all tensors are quantized (some small tensors should be left fp16 — see step 2 quantization criteria).

- [ ] **Step 6: Commit**

```bash
git add python/src/converter_kokoro.py python/src/converter.py
git commit -m "feat(converter): KokoroConverter — pth → cactus INT8 binary"
```

---

## Task 6: Kokoro text encoder

**Files:**
- Create: `cactus/models/kokoro/text_encoder.h`, `text_encoder.cpp`
- Test: extend Kokoro reference fixture to dump intermediate tensors

This task is **port-from-reference**. The text encoder is a small transformer with specific dim configs determined by the checkpoint. Procedure:

- [ ] **Step 1: Study the reference**

Clone kokoro-onnx and read its model graph:
```bash
git clone https://github.com/thewh1teagle/kokoro-onnx /tmp/kokoro-onnx-ref
ls /tmp/kokoro-onnx-ref/src/kokoro_onnx/
# Open model.py — find the text encoder forward pass.
```

Identify: number of layers, hidden dim, num heads, ffn dim, vocab size (phoneme alphabet size + special tokens).

- [ ] **Step 2: Extend reference fixture to dump text encoder output**

Edit `tests/fixtures/kokoro_reference.py` to also dump the post-text-encoder hidden states for the 5 phrases. The kokoro-onnx Kokoro class doesn't expose intermediates by default; run inference manually using the underlying `onnxruntime.InferenceSession` and inspect output names. Example pattern:

```python
import onnxruntime as ort
sess = ort.InferenceSession("kokoro-v0_19.onnx")
for inp in sess.get_inputs(): print("IN", inp.name, inp.shape)
for out in sess.get_outputs(): print("OUT", out.name, out.shape)
# Then re-run with a custom output spec to capture intermediate tensors.
```

Save text encoder hidden states as `kokoro_p01_text_hidden.bin` etc.

- [ ] **Step 3: Write the failing test**

Pattern (in `tests/test_voice_kokoro.cpp` — created in Task 10):

```cpp
TEST(KokoroTextEncoder, MatchesReference_p01) {
    auto phonemes = load_phoneme_ids("p01");  // helper that runs G2P then maps to ids
    auto expected = load_f32("tests/fixtures/data/kokoro_p01_text_hidden.bin");

    cactus::kokoro::TextEncoder enc;
    ASSERT_TRUE(enc.load_weights("kokoro-82m-q8.bin"));

    std::vector<float> output(expected.size());
    enc.forward(phonemes.data(), phonemes.size(), output.data());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-2f) << "at " << i;
    }
}
```

- [ ] **Step 4: Implement**

Pattern (`cactus/models/kokoro/text_encoder.cpp`):
- Read the existing transformer-style models in `cactus/models/` (e.g. `model_lfm2.cpp` or similar) for the cactus pattern of attention + FFN.
- Compose: phoneme embedding → N transformer blocks → final layer norm → output.
- Use cactus's existing matmul/softmax/layer_norm kernels.

**Time estimate:** 2–3 days, mostly reading the reference and matching shapes.

- [ ] **Step 5: Build + run + iterate until passing**

```bash
cmake --build build --target test_voice_kokoro && ./build/tests/test_voice_kokoro --gtest_filter="*TextEncoder*"
```

Iterate: when the output diverges, find the first layer that diverges by adding intermediate output dumps to both the cactus impl and the reference Python.

- [ ] **Step 6: Commit**

```bash
git add cactus/models/kokoro/text_encoder.h cactus/models/kokoro/text_encoder.cpp \
        tests/fixtures/kokoro_reference.py tests/test_voice_kokoro.cpp \
        cactus/CMakeLists.txt
git commit -m "feat(kokoro): text encoder forward pass"
```

---

## Task 7: Kokoro predictors (style + duration + F0 + N)

**Files:**
- Create: `cactus/models/kokoro/predictor.h`, `predictor.cpp`

Kokoro's predictor block predicts style vector, per-phoneme durations, F0 contour, and noise (N) tensor — fed into the decoder. This is the LSTM-heavy stage (uses Plan 1's `cactus_lstm_cell`).

- [ ] **Step 1: Study reference**

In kokoro-onnx, find the predictor module. Note: shapes of style vector, duration logits, F0/N output sequences.

- [ ] **Step 2: Dump reference intermediates**

Extend `kokoro_reference.py` to save predictor outputs for the 5 phrases.

- [ ] **Step 3: Write failing test**

Add to `test_voice_kokoro.cpp`:

```cpp
TEST(KokoroPredictor, MatchesReference_p01) {
    auto text_hidden = load_f32("tests/fixtures/data/kokoro_p01_text_hidden.bin");
    auto voice_emb = load_f32("tests/fixtures/data/kokoro_voice_af_bella.bin");
    auto exp_dur = load_f32("tests/fixtures/data/kokoro_p01_duration.bin");
    auto exp_f0  = load_f32("tests/fixtures/data/kokoro_p01_f0.bin");
    auto exp_n   = load_f32("tests/fixtures/data/kokoro_p01_n.bin");

    cactus::kokoro::Predictor pred;
    ASSERT_TRUE(pred.load_weights("kokoro-82m-q8.bin"));

    std::vector<float> dur(exp_dur.size()), f0(exp_f0.size()), n_out(exp_n.size());
    pred.forward(text_hidden.data(), voice_emb.data(),
                 dur.data(), f0.data(), n_out.data());

    for (size_t i = 0; i < exp_dur.size(); ++i)
        EXPECT_NEAR(dur[i], exp_dur[i], 1e-2f) << "dur at " << i;
    // ... same for f0 and n
}
```

- [ ] **Step 4: Implement**

Pattern: the predictor calls `cactus_lstm_cell` per timestep (the LSTM op from Plan 1 Task 10). Compose with linear layers and activations.

**Time estimate:** 2 days.

- [ ] **Step 5: Build + iterate to passing**

- [ ] **Step 6: Commit**

```bash
git add cactus/models/kokoro/predictor.h cactus/models/kokoro/predictor.cpp \
        tests/fixtures/kokoro_reference.py tests/test_voice_kokoro.cpp \
        cactus/CMakeLists.txt
git commit -m "feat(kokoro): style + duration + F0 + N predictor"
```

---

## Task 8: ISTFTNet decoder (highest-risk task)

**Files:**
- Create: `cactus/models/kokoro/decoder.h`, `decoder.cpp`

The ISTFTNet decoder takes (text encoder output, style, duration, F0, N) and produces audio. It uses upsampling transposed convs (Plan 1's `cactus_conv1d_transpose`), then ends with magnitude+phase prediction → `cactus_istft` → PCM.

**This is the riskiest task in the project.** Reserve a full day just for debugging, on top of implementation time.

- [ ] **Step 1: Study reference exhaustively**

Read the entire decoder forward in `kokoro-onnx`. Map every op to a cactus equivalent. Make a table on paper before writing any code:

| Layer | Op | cactus op | Notes |
|---|---|---|---|
| upsample 1 | ConvTranspose1d | cactus_conv1d_transpose | stride=N, K=N*2, etc. |
| ... | ... | ... | ... |
| final | iSTFT | cactus_istft | n_fft=??, hop=?? |

- [ ] **Step 2: Dump reference intermediate at every layer boundary**

Extend `kokoro_reference.py` to save *every* decoder layer output. Allows you to find the first layer that diverges instead of debugging the final audio output.

- [ ] **Step 3: Write the FINAL test (decoder produces audio matching reference)**

Add to `test_voice_kokoro.cpp`:

```cpp
TEST(KokoroDecoder, AudioMatchesReference_p01) {
    auto text_hidden = load_f32("tests/fixtures/data/kokoro_p01_text_hidden.bin");
    auto style = load_f32("tests/fixtures/data/kokoro_p01_style.bin");
    auto dur = load_f32("tests/fixtures/data/kokoro_p01_duration.bin");
    auto f0  = load_f32("tests/fixtures/data/kokoro_p01_f0.bin");
    auto n   = load_f32("tests/fixtures/data/kokoro_p01_n.bin");
    auto expected_audio = load_f32("tests/fixtures/data/kokoro_p01_audio.bin");

    cactus::kokoro::Decoder dec;
    ASSERT_TRUE(dec.load_weights("kokoro-82m-q8.bin"));

    std::vector<float> audio(expected_audio.size());
    dec.forward(text_hidden.data(), style.data(),
                dur.data(), f0.data(), n.data(),
                audio.data(), audio.size());

    // Exact match is too strict (FP nondeterminism in ISTFT). Use spectral diff.
    float spectral_diff_db = compute_spectral_diff_db(audio, expected_audio, /*n_fft=*/512);
    EXPECT_LT(spectral_diff_db, 2.0f) << "spectral diff too large";
}
```

Where `compute_spectral_diff_db` is a test helper that computes the max-bin dB diff between magnitude spectrograms of the two signals.

- [ ] **Step 4: Add per-layer tests for each upsample stage**

Pattern: `KokoroDecoder.Upsample1MatchesReference`, `KokoroDecoder.Upsample2MatchesReference`, etc. These are your debugging tools.

- [ ] **Step 5: Implement layer by layer, one upsample at a time, each with its test passing before moving on**

**Time estimate:** 4–5 days, of which 2 days is debugging the ISTFT phase reconstruction.

- [ ] **Step 6: Commit each layer as it passes**

```bash
git add cactus/models/kokoro/decoder.cpp tests/test_voice_kokoro.cpp
git commit -m "feat(kokoro): decoder upsample layer N matches reference"
```

After all layers + final ISTFT pass:

```bash
git commit -m "feat(kokoro): ISTFTNet decoder complete — audio matches kokoro-onnx <2dB"
```

---

## Task 9: Top-level Kokoro Model wiring

**Files:**
- Create: `cactus/models/model_kokoro.h`, `model_kokoro.cpp`

- [ ] **Step 1: Write the Model subclass**

Pattern: read existing `cactus/models/model_*.cpp` (e.g. `model_lfm2.cpp`) for the cactus `Model` interface (forward/prefill/decode signatures, weight loading, etc.). Kokoro doesn't fit the LLM Model API perfectly (it's text-in audio-out, not token-in token-out) — so it gets its own simpler interface:

```cpp
// cactus/models/model_kokoro.h
#pragma once
#include "kokoro/g2p.h"
#include "kokoro/text_encoder.h"
#include "kokoro/predictor.h"
#include "kokoro/decoder.h"
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

class KokoroModel {
public:
    bool load(const std::string& model_path,
              const std::string& dict_path,
              const std::string& voice_path);

    // Synthesize one phrase end-to-end.
    //   text:   input text (English)
    //   speed:  speech rate multiplier (1.0 = normal)
    //   out:    output buffer (24 kHz mono fp32). Caller may pre-size; if too small,
    //           returns false. Pass empty + 0 to get required size.
    //   out_n:  required size on input; actual size written on output.
    bool synthesize(const std::string& text, float speed,
                    float* out, size_t& out_n);

private:
    G2P g2p_;
    TextEncoder text_encoder_;
    Predictor predictor_;
    Decoder decoder_;
    std::vector<float> voice_embedding_;
    // ... phoneme→id table
};

}}
```

- [ ] **Step 2: Implement synthesize() as composition**

```cpp
bool KokoroModel::synthesize(const std::string& text, float speed,
                              float* out, size_t& out_n) {
    auto phonemes = g2p_.phonemize(text);
    auto ids = phoneme_strings_to_ids(phonemes);

    std::vector<float> text_hidden(/*size from text_encoder dims*/);
    text_encoder_.forward(ids.data(), ids.size(), text_hidden.data());

    std::vector<float> dur(...), f0(...), n(...);
    predictor_.forward(text_hidden.data(), voice_embedding_.data(),
                       dur.data(), f0.data(), n.data());

    if (speed != 1.0f) {
        for (auto& d : dur) d /= speed;
    }

    size_t needed = compute_audio_length(dur);
    if (out_n < needed) {
        out_n = needed;
        return false;
    }
    decoder_.forward(text_hidden.data(), voice_embedding_.data(),
                     dur.data(), f0.data(), n.data(),
                     out, needed);
    out_n = needed;
    return true;
}
```

- [ ] **Step 3: Add end-to-end test**

```cpp
TEST(KokoroModel, EndToEnd_p01) {
    cactus::kokoro::KokoroModel m;
    ASSERT_TRUE(m.load("kokoro-82m-q8.bin",
                       "assets/kokoro/kokoro-g2p.dict",
                       "assets/kokoro/voices/af_bella.bin"));

    size_t needed = 0;
    m.synthesize("Hello, my name is Cactus.", 1.0f, nullptr, needed);
    std::vector<float> audio(needed);
    ASSERT_TRUE(m.synthesize("Hello, my name is Cactus.", 1.0f, audio.data(), needed));

    auto expected = load_f32("tests/fixtures/data/kokoro_p01_audio.bin");
    float diff_db = compute_spectral_diff_db(audio, expected, 512);
    EXPECT_LT(diff_db, 2.0f);
}
```

- [ ] **Step 4: Run + commit**

```bash
cmake --build build --target test_voice_kokoro && ./build/tests/test_voice_kokoro
git add cactus/models/model_kokoro.h cactus/models/model_kokoro.cpp tests/test_voice_kokoro.cpp
git commit -m "feat(kokoro): top-level KokoroModel synthesize() end-to-end"
```

---

## Task 10: Run all 5 reference phrases as the final correctness gate

**Files:**
- Modify: `tests/test_voice_kokoro.cpp`

- [ ] **Step 1: Parameterize the end-to-end test over all 5 phrases**

```cpp
class KokoroEndToEnd : public ::testing::TestWithParam<std::string> {};

TEST_P(KokoroEndToEnd, MatchesReference) {
    const std::string tag = GetParam();
    static const std::map<std::string, std::string> phrases = {
        {"p01", "Hello, my name is Cactus."},
        {"p02", "The quick brown fox jumps over the lazy dog."},
        {"p03", "What time is it in Tokyo right now?"},
        {"p04", "I can run entirely on your device, no cloud needed."},
        {"p05", "Speech synthesis is the inverse of speech recognition."},
    };

    cactus::kokoro::KokoroModel m;
    ASSERT_TRUE(m.load("kokoro-82m-q8.bin",
                       "assets/kokoro/kokoro-g2p.dict",
                       "assets/kokoro/voices/af_bella.bin"));

    size_t needed = 0;
    m.synthesize(phrases.at(tag), 1.0f, nullptr, needed);
    std::vector<float> audio(needed);
    ASSERT_TRUE(m.synthesize(phrases.at(tag), 1.0f, audio.data(), needed));

    auto expected = load_f32("tests/fixtures/data/kokoro_" + tag + "_audio.bin");
    float diff_db = compute_spectral_diff_db(audio, expected, 512);
    EXPECT_LT(diff_db, 2.0f) << tag;
}

INSTANTIATE_TEST_SUITE_P(All, KokoroEndToEnd,
    ::testing::Values("p01","p02","p03","p04","p05"));
```

- [ ] **Step 2: Build + run**

```bash
cmake --build build --target test_voice_kokoro && ./build/tests/test_voice_kokoro
```

Expected: all 5 phrases pass. If one phrase fails, that's a clue about which kind of input breaks something (long sentences? questions? specific phonemes?).

- [ ] **Step 3: Commit**

```bash
git add tests/test_voice_kokoro.cpp
git commit -m "test(kokoro): end-to-end correctness gate — 5 reference phrases"
```

---

## Task 11: cactus_tts FFI surface

**Files:**
- Create: `cactus/ffi/cactus_tts.h`, `cactus_tts.cpp`
- Modify: `cactus/CMakeLists.txt`

- [ ] **Step 1: Write the C FFI header**

Create `cactus/ffi/cactus_tts.h`:

```c
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cactus_tts cactus_tts;

// Load a Kokoro TTS model. Returns NULL on failure.
//   model_path: path to .bin produced by KokoroConverter
//   dict_path:  path to G2P dictionary (assets/kokoro/kokoro-g2p.dict)
//   voice_path: path to voice embedding (.bin); NULL → default voice
cactus_tts* cactus_tts_create(const char* model_path,
                              const char* dict_path,
                              const char* voice_path);

void cactus_tts_destroy(cactus_tts*);

// Synthesize one phrase. Returns 0 on success, nonzero on failure.
//   text:       UTF-8 input text (English; non-English chars dropped)
//   speed:      1.0 = normal, 0.5 = half speed, 2.0 = double
//   out:        Caller-allocated buffer for fp32 PCM at 24 kHz mono.
//               Pass NULL with *out_n == 0 to query required size.
//   out_n:      In: caller capacity (frames). Out: actual frames written
//               (or required size if input *out_n was too small, in which
//               case return value is nonzero).
int cactus_tts_synthesize(cactus_tts*, const char* text, float speed,
                          float* out, size_t* out_n);

// Sample rate of all output (always 24000 for Kokoro).
int cactus_tts_sample_rate(cactus_tts*);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the implementation**

Create `cactus/ffi/cactus_tts.cpp`:

```cpp
#include "cactus_tts.h"
#include "../models/model_kokoro.h"
#include <new>

struct cactus_tts {
    cactus::kokoro::KokoroModel model;
};

extern "C" cactus_tts* cactus_tts_create(const char* model_path,
                                          const char* dict_path,
                                          const char* voice_path) {
    auto* h = new (std::nothrow) cactus_tts;
    if (!h) return nullptr;
    if (!h->model.load(model_path, dict_path, voice_path ? voice_path : "")) {
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" void cactus_tts_destroy(cactus_tts* h) { delete h; }

extern "C" int cactus_tts_synthesize(cactus_tts* h, const char* text, float speed,
                                       float* out, size_t* out_n) {
    if (!h || !text || !out_n) return -1;
    size_t cap = *out_n;
    if (!h->model.synthesize(text, speed, out, cap)) {
        *out_n = cap;  // required size for caller to retry
        return -2;
    }
    *out_n = cap;
    return 0;
}

extern "C" int cactus_tts_sample_rate(cactus_tts*) { return 24000; }
```

- [ ] **Step 3: Wire into build**

Append to `cactus/CMakeLists.txt`:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/ffi/cactus_tts.cpp
```

- [ ] **Step 4: Add a smoke test for the FFI**

Create `tests/test_cactus_tts_ffi.cpp`:

```cpp
#include <gtest/gtest.h>
#include "cactus_tts.h"
#include <vector>

TEST(CactusTTSFFI, SynthesizeProducesAudio) {
    auto* tts = cactus_tts_create("kokoro-82m-q8.bin",
                                   "assets/kokoro/kokoro-g2p.dict",
                                   "assets/kokoro/voices/af_bella.bin");
    ASSERT_NE(tts, nullptr);
    EXPECT_EQ(cactus_tts_sample_rate(tts), 24000);

    size_t n = 0;
    int rc = cactus_tts_synthesize(tts, "Hello.", 1.0f, nullptr, &n);
    EXPECT_NE(rc, 0);  // size query
    EXPECT_GT(n, 0);

    std::vector<float> audio(n);
    rc = cactus_tts_synthesize(tts, "Hello.", 1.0f, audio.data(), &n);
    EXPECT_EQ(rc, 0);
    EXPECT_GT(n, 0);

    // Sanity: nontrivial energy
    float sum_sq = 0;
    for (float s : audio) sum_sq += s * s;
    EXPECT_GT(sum_sq / audio.size(), 1e-6f) << "audio is silence";

    cactus_tts_destroy(tts);
}
```

Wire into tests CMake (same pattern as before).

- [ ] **Step 5: Build + run**

```bash
cmake --build build --target test_cactus_tts_ffi && ./build/tests/test_cactus_tts_ffi
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cactus/ffi/cactus_tts.h cactus/ffi/cactus_tts.cpp \
        tests/test_cactus_tts_ffi.cpp \
        cactus/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(ffi): cactus_tts public C API"
```

---

## Task 12: cactus run-tts CLI

**Files:**
- Create: `python/src/cli_run_tts.py`
- Modify: `tools/src/cli.py` (or wherever subcommands are registered)

- [ ] **Step 1: Write the subcommand**

Create `python/src/cli_run_tts.py`:

```python
"""cactus run-tts — synthesize text to a WAV file using Kokoro."""
import argparse
import ctypes
import sys
import wave
import numpy as np

from .ffi import _lib  # whatever the existing FFI loader is in cactus python

def main(argv=None):
    p = argparse.ArgumentParser(prog="cactus run-tts")
    p.add_argument("text", help="Text to synthesize")
    p.add_argument("--out", default="out.wav", help="Output WAV path")
    p.add_argument("--model", default="kokoro-82m-q8.bin")
    p.add_argument("--dict", default="assets/kokoro/kokoro-g2p.dict")
    p.add_argument("--voice", default="assets/kokoro/voices/af_bella.bin")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args(argv)

    # Bind ctypes to the cactus library
    _lib.cactus_tts_create.restype = ctypes.c_void_p
    _lib.cactus_tts_create.argtypes = [ctypes.c_char_p] * 3
    _lib.cactus_tts_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_tts_synthesize.restype = ctypes.c_int
    _lib.cactus_tts_synthesize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t),
    ]
    _lib.cactus_tts_sample_rate.restype = ctypes.c_int
    _lib.cactus_tts_sample_rate.argtypes = [ctypes.c_void_p]

    handle = _lib.cactus_tts_create(args.model.encode(), args.dict.encode(),
                                     args.voice.encode())
    if not handle:
        print("Failed to load model", file=sys.stderr)
        return 1

    try:
        n = ctypes.c_size_t(0)
        rc = _lib.cactus_tts_synthesize(handle, args.text.encode(), args.speed,
                                         None, ctypes.byref(n))
        # rc is nonzero on size query; n is now required capacity
        buf = (ctypes.c_float * n.value)()
        rc = _lib.cactus_tts_synthesize(handle, args.text.encode(), args.speed,
                                         buf, ctypes.byref(n))
        if rc != 0:
            print(f"synthesize failed (rc={rc})", file=sys.stderr)
            return 1

        sr = _lib.cactus_tts_sample_rate(handle)
        audio = np.frombuffer(buf, dtype=np.float32, count=n.value)
        # Convert to int16 for WAV
        clipped = np.clip(audio, -1.0, 1.0)
        i16 = (clipped * 32767).astype(np.int16)
        with wave.open(args.out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(i16.tobytes())

        print(f"Wrote {args.out} ({n.value} samples, {n.value/sr:.2f}s)")
        return 0
    finally:
        _lib.cactus_tts_destroy(handle)

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Register the subcommand**

Edit `tools/src/cli.py` to add `run-tts` to the subcommand dispatch. Pattern depends on existing CLI structure (argparse subparsers, click, typer, etc. — match existing code).

- [ ] **Step 3: Manual test**

```bash
cactus run-tts "Hello, this is cactus speaking from on-device TTS." --out hello.wav
afplay hello.wav    # macOS
```

Expected: audible "Hello, this is cactus speaking from on-device TTS." in af_bella's voice.

- [ ] **Step 4: Commit**

```bash
git add python/src/cli_run_tts.py tools/src/cli.py
git commit -m "feat(cli): cactus run-tts subcommand"
```

---

## Task 13: Plan 2 wrap-up + PR

**Files:**
- Create: `docs/superpowers/plans/2026-05-12-voice-loop-plan-2-kokoro-integration-COMPLETE.md`

- [ ] **Step 1: Run the full test suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: all tests pass (Plan 1 tests + Plan 2's `KokoroEndToEnd/p01..p05`, `CactusTTSFFI`, etc.).

- [ ] **Step 2: Manual verification of the demo flow**

```bash
cactus run-tts "The voice loop is closing." --out demo.wav && afplay demo.wav
```

Listen and confirm intelligibility + naturalness.

- [ ] **Step 3: Write completion note**

Create `docs/superpowers/plans/2026-05-12-voice-loop-plan-2-kokoro-integration-COMPLETE.md`:

```markdown
# Plan 2 — Kokoro Model Integration: COMPLETE

**Date completed:** (fill in)
**Branch:** voice-loop
**Final commit:** (fill in)

## Shipped
- KokoroConverter (`python/src/converter_kokoro.py`) — .pth → cactus INT8 ~85 MB binary
- Bundled CMUdict-derived G2P dictionary at `assets/kokoro/kokoro-g2p.dict` (~5 MB)
- G2P with L2S fallback at `cactus/models/kokoro/g2p.{h,cpp}`
- Kokoro forward pass: text encoder, predictor, ISTFTNet decoder
- KokoroModel top-level (`cactus/models/model_kokoro.{h,cpp}`)
- cactus_tts C FFI (`cactus/ffi/cactus_tts.{h,cpp}`)
- `cactus run-tts` CLI subcommand
- 5 reference phrases pass end-to-end vs kokoro-onnx within 2 dB spectral diff

## Known limitations (per spec, accepted)
- L2S fallback mispronounces some proper nouns (documented)
- Single voice bundled; additional voices are separate asset files
- INT8 only; INT4 deferred to Stage 1.5
- English only; multilingual deferred

## Next: Plan 3 — Voice Session Orchestrator
```

- [ ] **Step 4: Commit + push + PR**

```bash
git add docs/superpowers/plans/2026-05-12-voice-loop-plan-2-kokoro-integration-COMPLETE.md
git commit -m "docs: Plan 2 (Kokoro integration) complete"
git push origin voice-loop

gh pr create --repo cactus-compute/cactus --base main --head sb1992:voice-loop \
  --title "feat(tts): Kokoro-82M integration with cactus_tts FFI" \
  --body "$(cat <<'EOF'
End-to-end Kokoro TTS in cactus:

- KokoroConverter (.pth → INT8 cactus binary, ~85 MB)
- Bundled CMUdict-derived G2P dictionary + L2S fallback for OOV
- Forward pass: text encoder → predictors → ISTFTNet decoder → 24 kHz PCM
- Top-level KokoroModel + cactus_tts C FFI
- `cactus run-tts "<text>"` CLI subcommand
- 5 reference phrases pass within ≤2 dB spectral diff vs kokoro-onnx

Builds on the kernel foundation from the prior voice-loop PR (FFT/STFT/LSTM/conv1d_transpose).

Part of voice-loop stage 1: docs/superpowers/specs/2026-05-12-voice-loop-stage-1-design.md
EOF
)"
```

---

## Self-Review Notes

Spec coverage check (against the spec's "TTS model integration" section):

| Spec requirement | Plan task |
|---|---|
| Kokoro forward pass (text encoder, predictors, ISTFTNet) | Tasks 6, 7, 8, 9 |
| KokoroConverter (.pth → cactus binary) + INT8 | Task 5 |
| Dictionary G2P + L2S fallback | Tasks 3, 4 |
| Streaming output (sentence aggregator) | **NOT in this plan** — that's Plan 3 (Voice Session). The TTS API here is one-shot synthesize-the-whole-string. Streaming is built on top by buffering sentences in the orchestrator. |
| Validate vs kokoro-onnx, 5 phrases, ≤2 dB spectral diff | Task 10 |
| INT4 quantization | Deferred to Stage 1.5 per spec |

**Gaps acknowledged in plan:**
- Tasks 5, 6, 7, 8 are honestly marked as "study reference + port". Cannot fully specify code without the live kokoro-onnx checkpoint inspection.
- Plan 3 (Voice Session Orchestrator) and Plan 4 (macOS Demo App) are separate plans — to be written after Plan 2 completes and surfaces any API surprises.
