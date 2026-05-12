#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cactus_tts cactus_tts;

// Create a TTS instance from a Kokoro model directory.
//   weights_dir:   path to the Kokoro flat weights directory (.weights files)
//   g2p_dict_path: path to the G2P CMU-style dictionary
//   voice_path:    path to a single-voice raw float32 file
//                  (511 * 256 = 130816 floats, 523264 bytes)
// Returns NULL on failure.
cactus_tts* cactus_tts_create(const char* weights_dir,
                              const char* g2p_dict_path,
                              const char* voice_path);

void cactus_tts_destroy(cactus_tts* h);

// Synthesize text. Returns 0 on success, nonzero on error.
//   text:   UTF-8 input, English only for v1
//   speed:  1.0 = normal speech rate
//   out:    caller-allocated f32 buffer for 24 kHz mono PCM.
//           Pass NULL with *out_n == 0 to query required size.
//   out_n:  in:  caller capacity (frames).
//           out: actual frames written (or required size if input *out_n
//                was too small, in which case return value is nonzero —
//                caller should resize and retry).
//
// Return codes:
//    0  success
//   -1  invalid arguments (null handle / null text / null out_n)
//   -2  synthesis failed inside KokoroModel
//   -3  out buffer too small (or out == NULL); *out_n holds required size
int cactus_tts_synthesize(cactus_tts* h,
                          const char* text,
                          float speed,
                          float* out,
                          size_t* out_n);

// Synthesize from pre-computed phoneme IDs (no G2P).
// Use this when the caller has already phonemized text (e.g. via misaki on
// the Python side). Bypasses the imperfect built-in G2P->IPA mapping.
//   ids:    pointer to n_ids int64 phoneme IDs (Kokoro vocab). Caller must
//           include the leading and trailing 0 pad tokens that KModel uses
//           (i.e. ids = [0, ...misaki_ids, 0]).
//   n_ids:  count of phoneme IDs.
//   speed:  must be 1.0 (other values rejected, mirroring text path).
//   out:    caller-allocated f32 buffer for 24 kHz mono PCM. Pass NULL with
//           *out_n == 0 to query required size.
//   out_n:  in:  capacity (frames).
//           out: actual frames written (or required size if too small).
//
// Return codes:
//    0  success
//   -1  invalid arguments (null handle / null ids / null out_n)
//   -2  synthesis failed inside KokoroModel
//   -3  out buffer too small (or out == NULL); *out_n holds required size
//   -4  speed != 1.0 (unsupported)
int cactus_tts_synthesize_phonemes(cactus_tts* h,
                                   const int64_t* ids, size_t n_ids,
                                   float speed,
                                   float* out, size_t* out_n);

// Sample rate of all output (always 24000 for Kokoro).
int cactus_tts_sample_rate(cactus_tts* h);

#ifdef __cplusplus
}
#endif
