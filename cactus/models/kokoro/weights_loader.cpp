#include "weights_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

namespace {

constexpr uint32_t CACT_MAGIC       = 0x54434143;  // 'CACT' little-endian
constexpr uint32_t FLAG_INTERLEAVED = 1u << 3;
constexpr int      INTERLEAVE_BLOCK = 4;
constexpr int      HEADER_SIZE      = 84;
constexpr int      ALIGNMENT        = 32;

inline int align_offset(int off, int align) {
    int rem = off % align;
    return rem ? off + (align - rem) : off;
}

// IEEE 754 fp16 -> fp32. Standard bit-twiddling, no platform deps.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3FFu;
            exp += 1;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) throw std::runtime_error("short read for " + path);
    return buf;
}

template <typename T>
inline T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// Inverse of tensor_io.interleave_weights: input flattened layout is
//   reshape(N_padded/B, K/4, B, 4) (C-order, B=4)
// We deinterleave to (N_padded, K) row-major int8.
static void deinterleave_weights(const int8_t* in,
                                 int N_padded, int K,
                                 std::vector<int8_t>& out) {
    constexpr int B = INTERLEAVE_BLOCK;
    out.assign(static_cast<size_t>(N_padded) * K, 0);
    const int blocks_n = N_padded / B;
    const int blocks_k = K / 4;
    for (int bn = 0; bn < blocks_n; ++bn) {
        for (int bk = 0; bk < blocks_k; ++bk) {
            for (int r = 0; r < B; ++r) {
                for (int c = 0; c < 4; ++c) {
                    const int src_idx = ((bn * blocks_k + bk) * B + r) * 4 + c;
                    const int dst_row = bn * B + r;
                    const int dst_col = bk * 4 + c;
                    out[static_cast<size_t>(dst_row) * K + dst_col] = in[src_idx];
                }
            }
        }
    }
}

// Inverse of tensor_io.interleave_scales: layout
//   reshape(N_padded/B, num_groups, B) -> (N_padded, num_groups)
static void deinterleave_scales(const float* in,
                                int N_padded, int num_groups,
                                std::vector<float>& out) {
    constexpr int B = INTERLEAVE_BLOCK;
    out.assign(static_cast<size_t>(N_padded) * num_groups, 0.0f);
    const int blocks_n = N_padded / B;
    for (int bn = 0; bn < blocks_n; ++bn) {
        for (int g = 0; g < num_groups; ++g) {
            for (int r = 0; r < B; ++r) {
                const int src = ((bn * num_groups) + g) * B + r;
                const int dst_row = bn * B + r;
                out[static_cast<size_t>(dst_row) * num_groups + g] = in[src];
            }
        }
    }
}

}  // namespace

WeightsFile load_weights_file(const std::string& path) {
    auto raw = read_all(path);
    if (raw.size() < static_cast<size_t>(HEADER_SIZE)) {
        throw std::runtime_error("file too small: " + path);
    }
    const uint8_t* p = raw.data();
    const uint32_t magic = read_le<uint32_t>(p + 0);
    if (magic != CACT_MAGIC) throw std::runtime_error("bad magic in " + path);

    const uint32_t flags = read_le<uint32_t>(p + 4);
    // alignment at +8 (always 32 here, ignore)
    const uint32_t ndim = read_le<uint32_t>(p + 12);
    int dims[4];
    for (int i = 0; i < 4; ++i) {
        const uint64_t d = read_le<uint64_t>(p + 16 + 8 * i);
        dims[i] = static_cast<int>(d);
    }
    const uint32_t precision = read_le<uint32_t>(p + 48);
    const uint64_t data_bytes = read_le<uint64_t>(p + 52);
    const uint64_t scales_bytes = read_le<uint64_t>(p + 60);
    const uint32_t group_size = read_le<uint32_t>(p + 68);
    const uint32_t num_groups = read_le<uint32_t>(p + 72);
    const uint64_t original_N = read_le<uint64_t>(p + 76);

    const int header_aligned = align_offset(HEADER_SIZE, ALIGNMENT);

    WeightsFile out;

    if (precision == 1) {
        // FP16, no scales. Body starts at aligned header.
        const int body_off = header_aligned;
        const size_t n_elems = static_cast<size_t>(data_bytes) / 2;
        if (body_off + data_bytes > raw.size()) {
            throw std::runtime_error("FP16 body OOB in " + path);
        }
        out.data.resize(n_elems);
        const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(p + body_off);
        for (size_t i = 0; i < n_elems; ++i) out.data[i] = fp16_to_fp32(fp16[i]);
        for (uint32_t i = 0; i < ndim; ++i) out.shape.push_back(dims[i]);
        return out;
    }

    if (precision != 0) {
        throw std::runtime_error("unsupported precision in " + path);
    }

    // INT8 path. Layout:
    //   header (84) -> pad to 32 -> scales (FP16) -> pad to 32 -> data (INT8)
    out.is_int8 = true;
    const int scales_off = header_aligned;
    const size_t n_scales = static_cast<size_t>(scales_bytes) / 2;
    if (scales_off + scales_bytes > raw.size()) {
        throw std::runtime_error("INT8 scales OOB in " + path);
    }
    std::vector<float> scales_fp32(n_scales);
    {
        const uint16_t* sfp16 = reinterpret_cast<const uint16_t*>(p + scales_off);
        for (size_t i = 0; i < n_scales; ++i) scales_fp32[i] = fp16_to_fp32(sfp16[i]);
    }
    const int data_off = align_offset(scales_off + static_cast<int>(scales_bytes), ALIGNMENT);
    if (data_off + data_bytes > raw.size()) {
        throw std::runtime_error("INT8 data OOB in " + path);
    }
    const int8_t* qd = reinterpret_cast<const int8_t*>(p + data_off);

    if (ndim == 2) {
        const int N_padded = dims[0];
        const int K = dims[1];
        const int orig_N = static_cast<int>(original_N);

        std::vector<int8_t> q2d;
        std::vector<float> s2d;

        if (flags & FLAG_INTERLEAVED) {
            deinterleave_weights(qd, N_padded, K, q2d);
            deinterleave_scales(scales_fp32.data(), N_padded, static_cast<int>(num_groups), s2d);
        } else {
            // Plain row-major INT8 + per-row-group scales (flat (N_padded, num_groups))
            q2d.assign(qd, qd + static_cast<size_t>(N_padded) * K);
            s2d = scales_fp32;
        }

        // Dequant: out[r, g*GS + j] = q2d[r, g*GS + j] * s2d[r, g]
        out.data.assign(static_cast<size_t>(orig_N) * K, 0.0f);
        const int gs = static_cast<int>(group_size);
        for (int r = 0; r < orig_N; ++r) {
            for (int g = 0; g < static_cast<int>(num_groups); ++g) {
                const float scale = s2d[static_cast<size_t>(r) * num_groups + g];
                for (int j = 0; j < gs; ++j) {
                    const int col = g * gs + j;
                    if (col >= K) break;
                    const int8_t qv = q2d[static_cast<size_t>(r) * K + col];
                    out.data[static_cast<size_t>(r) * K + col] = qv * scale;
                }
            }
        }
        out.shape = {orig_N, K};
        return out;
    }

    if (ndim == 1) {
        const int K = dims[0];
        out.data.assign(static_cast<size_t>(K), 0.0f);
        const int gs = static_cast<int>(group_size);
        for (int g = 0; g < static_cast<int>(num_groups); ++g) {
            const float scale = scales_fp32[g];
            for (int j = 0; j < gs; ++j) {
                const int col = g * gs + j;
                if (col >= K) break;
                out.data[col] = qd[col] * scale;
            }
        }
        out.shape = {K};
        return out;
    }

    throw std::runtime_error("unsupported INT8 ndim in " + path);
}

}}  // namespace cactus::kokoro
