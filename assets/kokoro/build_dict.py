"""Build kokoro-g2p.dict from CMUdict.

Downloads CMU Pronouncing Dictionary (public domain), normalizes to ARPAbet
(no stress markers — Kokoro doesn't use them), and emits a binary format
optimized for fast lookup at inference time.

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

Run: python3.11 assets/kokoro/build_dict.py
Output: assets/kokoro/kokoro-g2p.dict (~5 MB)
"""
import os
import re
import struct
import urllib.request
from collections import OrderedDict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro-g2p.dict")
SRC = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

print(f"Downloading CMUdict from {SRC} ...")
raw = urllib.request.urlopen(SRC).read().decode("utf-8")
print(f"  fetched {len(raw)} bytes")

# Parse: each line is "WORD(N)? ph1 ph2 ph3 ..."
# Strip stress digits (AA1 -> AA), drop alternate prons (keep only first).
seen = OrderedDict()
phoneme_alphabet = OrderedDict()
for line in raw.splitlines():
    line = line.strip()
    if not line or line.startswith(";;;"):
        continue
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    word, prons = parts
    # cmudict variant suffix is "(N)" e.g. "READ(1)"
    word = re.sub(r"\(\d+\)$", "", word).lower()
    if word in seen:
        continue  # keep only the primary pronunciation
    # Strip trailing # comment if present (cmudict.dict format)
    if "#" in prons:
        prons = prons.split("#", 1)[0].strip()
    phs = [re.sub(r"\d$", "", p) for p in prons.split() if p]
    if not phs:
        continue
    seen[word] = phs
    for p in phs:
        if p not in phoneme_alphabet:
            phoneme_alphabet[p] = len(phoneme_alphabet)

print(f"  parsed {len(seen)} words, {len(phoneme_alphabet)} unique phonemes")
assert len(phoneme_alphabet) <= 255, "phoneme alphabet must fit in uint8"

# Filter: only allow ASCII alphanumeric + apostrophe in words (defensive — drops
# CMUdict's punctuation entries which a C++ ASCII loader wouldn't parse cleanly)
clean = OrderedDict()
import string
allowed = set(string.ascii_lowercase + string.digits + "'")
for w, phs in seen.items():
    if all(c in allowed for c in w) and 0 < len(w) < 65535:
        clean[w] = phs
print(f"  after ASCII filter: {len(clean)} words")
seen = clean

# Write binary
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

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Wrote {OUT} ({size_mb:.2f} MB)")
print(f"Phoneme alphabet: {list(phoneme_alphabet.keys())}")
