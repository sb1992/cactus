#include "test_utils.h"
#include "../cactus/models/kokoro/g2p.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct WordCase {
    std::string word;
    std::vector<std::string> expected_phonemes;  // empty = OOV (just check L2S returned something)
};

static std::vector<WordCase> load_fixture() {
    std::vector<WordCase> out;
    const char* candidates[] = {
        "tests/fixtures/g2p_words.txt",
        "../tests/fixtures/g2p_words.txt",
        "../../tests/fixtures/g2p_words.txt",
        "fixtures/g2p_words.txt",
        "../fixtures/g2p_words.txt",
    };
    std::ifstream f;
    for (const char* p : candidates) {
        f.open(p);
        if (f) break;
        f.clear();
    }
    if (!f) {
        std::cerr << "fixture file g2p_words.txt not found\n";
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        WordCase wc;
        const auto tab = line.find('\t');
        wc.word = line.substr(0, tab);
        if (tab != std::string::npos) {
            std::istringstream ps(line.substr(tab + 1));
            std::string p;
            while (ps >> p) wc.expected_phonemes.push_back(p);
        }
        out.push_back(wc);
    }
    return out;
}

static std::string find_dict_path() {
    const char* candidates[] = {
        "assets/kokoro/kokoro-g2p.dict",
        "../assets/kokoro/kokoro-g2p.dict",
        "../../assets/kokoro/kokoro-g2p.dict",
    };
    for (const char* p : candidates) {
        std::ifstream f(p);
        if (f) return p;
    }
    return "assets/kokoro/kokoro-g2p.dict";
}

static bool test_g2p_dictionary_and_oov() {
    cactus::kokoro::G2P g2p;
    const auto dict_path = find_dict_path();
    if (!g2p.load(dict_path)) {
        std::fprintf(stderr, "G2P load failed: %s\n", dict_path.c_str());
        return false;
    }

    auto cases = load_fixture();
    if (cases.empty()) {
        std::fprintf(stderr, "no fixture cases loaded\n");
        return false;
    }

    int hits = 0, oov = 0, errors = 0;
    for (const auto& wc : cases) {
        const auto result = g2p.lookup(wc.word);
        if (!wc.expected_phonemes.empty()) {
            if (result != wc.expected_phonemes) {
                if (errors < 5) {
                    std::fprintf(stderr, "DICT MISMATCH '%s': got [", wc.word.c_str());
                    for (const auto& p : result) std::fprintf(stderr, "%s ", p.c_str());
                    std::fprintf(stderr, "] expected [");
                    for (const auto& p : wc.expected_phonemes) std::fprintf(stderr, "%s ", p.c_str());
                    std::fprintf(stderr, "]\n");
                }
                ++errors;
            }
            ++hits;
        } else {
            if (result.empty()) {
                if (errors < 5) std::fprintf(stderr, "OOV FALLBACK EMPTY for '%s'\n", wc.word.c_str());
                ++errors;
            }
            ++oov;
        }
    }
    std::printf("Tested %d dictionary hits + %d OOV (L2S fallback), %d errors\n",
                hits, oov, errors);
    return errors == 0;
}

int main() {
    TestUtils::TestRunner runner("cactus::kokoro::G2P");
    runner.run_test("Dictionary lookup + L2S fallback fixture",
                    test_g2p_dictionary_and_oov());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
