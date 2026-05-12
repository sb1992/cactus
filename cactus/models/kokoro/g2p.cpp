#include "g2p.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

namespace cactus { namespace kokoro {

bool G2P::load(const std::string& dict_path) {
    std::ifstream f(dict_path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0xCACDCACDu || version != 1u) return false;

    uint32_t n_words = 0;
    f.read(reinterpret_cast<char*>(&n_words), 4);
    table_.reserve(n_words);

    for (uint32_t i = 0; i < n_words; ++i) {
        uint16_t wlen = 0;
        f.read(reinterpret_cast<char*>(&wlen), 2);
        std::string word(wlen, '\0');
        f.read(word.data(), wlen);
        uint16_t plen = 0;
        f.read(reinterpret_cast<char*>(&plen), 2);
        std::vector<uint8_t> phs(plen);
        f.read(reinterpret_cast<char*>(phs.data()), plen);
        table_.emplace(std::move(word), std::move(phs));
    }

    uint16_t alpha_n = 0;
    f.read(reinterpret_cast<char*>(&alpha_n), 2);
    alphabet_.reserve(alpha_n);
    for (uint16_t i = 0; i < alpha_n; ++i) {
        uint8_t l = 0;
        f.read(reinterpret_cast<char*>(&l), 1);
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
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalpha(uc) || c == '\'') {
            out.push_back(static_cast<char>(std::tolower(uc)));
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
    for (uint8_t idx : it->second) {
        if (idx < alphabet_.size()) out.push_back(alphabet_[idx]);
    }
    return out;
}

// Minimal English letter-to-sound rules. Not winning any pronunciation contests,
// but covers enough of the common cases (consonants + simple vowel mappings)
// that OOV proper nouns are at least intelligible. Documented as imperfect.
std::vector<std::string> G2P::l2s_fallback(const std::string& word) const {
    static const std::unordered_map<char, const char*> simple = {
        {'b',"B"},{'c',"K"},{'d',"D"},{'f',"F"},{'g',"G"},{'h',"HH"},{'j',"JH"},
        {'k',"K"},{'l',"L"},{'m',"M"},{'n',"N"},{'p',"P"},{'q',"K"},{'r',"R"},
        {'s',"S"},{'t',"T"},{'v',"V"},{'w',"W"},{'x',"K S"},{'y',"Y"},{'z',"Z"},
        {'a',"AE"},{'e',"EH"},{'i',"IH"},{'o',"AA"},{'u',"AH"},
    };
    std::vector<std::string> out;
    for (size_t i = 0; i < word.size(); ++i) {
        const char c = word[i];

        // Tiny digraph table (priority over single-letter)
        if (i + 1 < word.size()) {
            const char d = word[i+1];
            if (c == 't' && d == 'h') { out.push_back("TH"); ++i; continue; }
            if (c == 's' && d == 'h') { out.push_back("SH"); ++i; continue; }
            if (c == 'c' && d == 'h') { out.push_back("CH"); ++i; continue; }
            if (c == 'n' && d == 'g') { out.push_back("NG"); ++i; continue; }
            if (c == 'p' && d == 'h') { out.push_back("F");  ++i; continue; }
            if (c == 'q' && d == 'u') { out.push_back("K"); out.push_back("W"); ++i; continue; }
        }

        const auto it = simple.find(c);
        if (it != simple.end()) {
            std::string s = it->second;
            const auto sp = s.find(' ');
            if (sp == std::string::npos) {
                out.push_back(s);
            } else {
                out.push_back(s.substr(0, sp));
                out.push_back(s.substr(sp + 1));
            }
        }
        // unknown chars (numbers, apostrophes after normalize) silently dropped
    }
    return out;
}

std::vector<std::string> G2P::phonemize(const std::string& text) const {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            const auto ph = lookup(cur);
            for (const auto& p : ph) out.push_back(p);
            cur.clear();
        }
    };
    for (char c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalpha(uc) || c == '\'') {
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

}}  // namespace cactus::kokoro
