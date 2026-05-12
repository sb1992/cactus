#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cactus { namespace kokoro {

class G2P {
public:
    // Load binary dictionary (format: see assets/kokoro/build_dict.py).
    // Returns true on success.
    bool load(const std::string& dict_path);

    // Look up phonemes for a single word.
    //   - Returns dictionary phonemes if word is in CMUdict
    //   - Falls back to letter-to-sound rules for OOV
    //   - Word is normalized to lowercase + ASCII-letters-only before lookup
    //   - Returns empty vector iff the input has zero usable letters
    std::vector<std::string> lookup(const std::string& word) const;

    // Tokenize a sentence into phonemes (whitespace-split, then per-word lookup).
    // Sentence-final punctuation '.' '!' '?' is preserved as a single-character
    // pseudo-phoneme token.
    std::vector<std::string> phonemize(const std::string& text) const;

private:
    std::unordered_map<std::string, std::vector<uint8_t>> table_;
    std::vector<std::string> alphabet_;

    // L2S fallback for OOV. Tiny rule set, NOT a TTS-grade phonemizer.
    // Documented as imperfect for proper nouns / foreign loanwords.
    std::vector<std::string> l2s_fallback(const std::string& word) const;
};

}}  // namespace cactus::kokoro
