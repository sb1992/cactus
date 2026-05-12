#pragma once

// Cactus .weights file loader for the Kokoro models.
//
// Mirrors the converter (python/src/converter_kokoro.py) on-disk format:
//
//   header (84 B) -> pad to 32 -> [scales (FP16) -> pad to 32] -> data
//   precision = 0 (INT8 group-quantized) | 1 (FP16)
//   ndim     in {1, 2}
//   flag bit 3 = INTERLEAVED (only seen on INT8 2D)
//
// The loader dequantizes/decodes everything to fp32 in-memory; downstream
// kokoro modules reuse this for every weight file.
//
// Kept private to cactus/models/kokoro/ because the production graph runtime
// (cactus/graph/graph_io.cpp MappedFile) is graph-coupled and not exported
// as a standalone API.

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

struct WeightsFile {
    std::vector<float> data;
    std::vector<int>   shape;       // logical shape (NOT padded)
    bool               is_int8 = false;
};

// Load and decode a single .weights file. Throws std::runtime_error on
// any I/O or format error.
WeightsFile load_weights_file(const std::string& path);

}}  // namespace cactus::kokoro
