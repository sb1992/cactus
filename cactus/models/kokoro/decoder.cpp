#include "decoder.h"

#include "kokoro_internal.h"
#include "weights_loader.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

namespace {

// Conv1d with arbitrary kernel + stride + padding, no dilation, no groups
// (implicitly groups=1). Input cf is (C_in, T_in) channel-first; output
// cf_out is (C_out, T_out) channel-first.
//
//   T_out = (T_in + 2*padding - K) / stride + 1
//
// Used for F0_conv / N_conv (k=3, s=2, p=1).
void conv1d_strided(const float* cf, int C_in, int T_in,
                    const float* weight, const float* bias /*nullable*/,
                    int C_out, int K, int stride, int padding,
                    std::vector<float>& cf_out, int& T_out) {
    T_out = (T_in + 2 * padding - K) / stride + 1;
    if (T_out < 0) T_out = 0;
    cf_out.assign(static_cast<size_t>(C_out) * T_out, 0.0f);
    for (int oc = 0; oc < C_out; ++oc) {
        const float bv = bias ? bias[oc] : 0.0f;
        const float* wrow = &weight[static_cast<size_t>(oc) * C_in * K];
        float* outrow = &cf_out[static_cast<size_t>(oc) * T_out];
        for (int t = 0; t < T_out; ++t) outrow[t] = bv;
        for (int t = 0; t < T_out; ++t) {
            const int t_in_base = t * stride - padding;
            for (int k = 0; k < K; ++k) {
                const int it = t_in_base + k;
                if (it < 0 || it >= T_in) continue;
                for (int ic = 0; ic < C_in; ++ic) {
                    const float w = wrow[ic * K + k];
                    outrow[t] += w * cf[static_cast<size_t>(ic) * T_in + it];
                }
            }
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------
// load_weights
// ----------------------------------------------------------------------
bool DecoderEncodeStage::load_weights(const std::string& dir) {
    auto load_named = [&](const std::string& base) -> std::vector<float> {
        return load_weights_file(dir + "/decoder_" + base + ".weights").data;
    };

    auto load_aln_fc = [&](AdaLNFc& fc, const std::string& base, int C) {
        fc.w = load_named(base + "_weight");
        fc.b = load_named(base + "_bias");
        if (static_cast<int>(fc.w.size()) != 2 * C * STYLE_DIM ||
            static_cast<int>(fc.b.size()) != 2 * C) {
            throw std::runtime_error("AdaLN fc size mismatch in " + base);
        }
    };

    auto load_conv = [&](Conv1d& c, const std::string& base, int Cout, int Cin,
                         int K, bool with_bias) {
        c.w = load_named(base + "_weight");
        if (with_bias) {
            c.b = load_named(base + "_bias");
            if (static_cast<int>(c.b.size()) != Cout) {
                throw std::runtime_error("conv bias size mismatch in " + base);
            }
        } else {
            c.b.clear();
        }
        if (static_cast<int>(c.w.size()) != Cout * Cin * K) {
            throw std::runtime_error("conv weight size mismatch in " + base
                                     + " got " + std::to_string(c.w.size())
                                     + " expected " + std::to_string(Cout*Cin*K));
        }
        c.C_in = Cin; c.C_out = Cout; c.K = K;
    };

    auto load_resblk = [&](AdainResBlk1d& blk, const std::string& base,
                           int dim_in, int dim_out, bool upsample) {
        blk.dim_in = dim_in;
        blk.dim_out = dim_out;
        blk.upsample = upsample;
        blk.learned_sc = (dim_in != dim_out);
        load_aln_fc(blk.norm1_fc, base + "_norm1_fc", dim_in);
        load_aln_fc(blk.norm2_fc, base + "_norm2_fc", dim_out);
        load_conv(blk.conv1, base + "_conv1", dim_out, dim_in, 3, true);
        load_conv(blk.conv2, base + "_conv2", dim_out, dim_out, 3, true);
        if (upsample) {
            // Depthwise ConvT1d(dim_in, dim_in, K=3, s=2, groups=dim_in,
            //                   padding=1, output_padding=1). PyTorch stores
            // the weight as (in_channels, out_channels/groups=1, K) so
            // expect dim_in * 1 * 3 floats.
            blk.pool.w = load_named(base + "_pool_weight");
            blk.pool.b = load_named(base + "_pool_bias");
            if (static_cast<int>(blk.pool.w.size()) != dim_in * 1 * 3) {
                throw std::runtime_error(base + " pool weight size mismatch");
            }
            if (static_cast<int>(blk.pool.b.size()) != dim_in) {
                throw std::runtime_error(base + " pool bias size mismatch");
            }
            blk.pool.C_in = dim_in; blk.pool.C_out = dim_in; blk.pool.K = 3;
        }
        if (blk.learned_sc) {
            // 1x1 conv, no bias
            load_conv(blk.conv1x1, base + "_conv1x1", dim_out, dim_in, 1, false);
        }
    };

    try {
        // F0_conv / N_conv: Conv1d(1, 1, k=3, s=2, p=1) + bias (1,)
        load_conv(F0_conv_, "F0_conv", 1, 1, 3, true);
        load_conv(N_conv_,  "N_conv",  1, 1, 3, true);

        // encode: AdainResBlk1d(514, 1024)
        load_resblk(encode_, "encode", ASR_CH + 2, ENCODE_OUT_CH, false);

        // asr_res: Sequential(weight_norm(Conv1d(512, 64, k=1))). Weight base
        // includes the Sequential index suffix `_0`.
        load_conv(asr_res_, "asr_res_0", ASR_RES_CH, ASR_CH, 1, true);

        // decode.0..2: AdainResBlk1d(1090, 1024), no upsample
        const int decode_in_ch = ENCODE_OUT_CH + ASR_RES_CH + 2;  // 1090
        load_resblk(decode_[0], "decode_0", decode_in_ch, DECODE_OUT_CH, false);
        load_resblk(decode_[1], "decode_1", decode_in_ch, DECODE_OUT_CH, false);
        load_resblk(decode_[2], "decode_2", decode_in_ch, DECODE_OUT_CH, false);
        // decode.3: AdainResBlk1d(1090, 512, upsample=True)
        load_resblk(decode_[3], "decode_3", decode_in_ch, DECODE3_OUT_CH, true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DecoderEncodeStage load_weights failed: %s\n", e.what());
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------
// run_until_stage — main pipeline.
// ----------------------------------------------------------------------
void DecoderEncodeStage::run_until_stage(const float* asr, int T_mel,
                                         const float* F0, const float* N,
                                         const float* style128,
                                         Stage last,
                                         std::vector<float>& out) const {
    // ---- 1. F0_conv: input (1, 1, 2*T_mel), output (1, 1, T_mel) ---------
    std::vector<float> F0_after, N_after;
    int T_F0_out = 0, T_N_out = 0;
    conv1d_strided(F0, /*C_in=*/1, /*T_in=*/2 * T_mel,
                   F0_conv_.w.data(), F0_conv_.b.data(),
                   /*C_out=*/1, /*K=*/3, /*stride=*/2, /*padding=*/1,
                   F0_after, T_F0_out);
    conv1d_strided(N,  /*C_in=*/1, /*T_in=*/2 * T_mel,
                   N_conv_.w.data(), N_conv_.b.data(),
                   /*C_out=*/1, /*K=*/3, /*stride=*/2, /*padding=*/1,
                   N_after, T_N_out);
    // PyTorch arithmetic: T_out = (2T - 3 + 2)/2 + 1 = T_mel exactly (since
    // 2T-1 is odd and integer-divides to T_mel-1, +1 -> T_mel). Sanity:
    if (T_F0_out != T_mel || T_N_out != T_mel) {
        std::fprintf(stderr, "F0/N conv produced unexpected length %d/%d (expected %d)\n",
                     T_F0_out, T_N_out, T_mel);
    }

    if (last == Stage::F0AfterConv) { out = F0_after; return; }
    if (last == Stage::NAfterConv)  { out = N_after;  return; }

    // ---- 2. asr_res: 1x1 Conv1d(512 -> 64) on asr (channel-first) --------
    // Convert input (T_mel, 512) row-major to channel-first (512, T_mel).
    std::vector<float> asr_cf(static_cast<size_t>(ASR_CH) * T_mel, 0.0f);
    for (int t = 0; t < T_mel; ++t) {
        for (int c = 0; c < ASR_CH; ++c) {
            asr_cf[static_cast<size_t>(c) * T_mel + t] =
                asr[static_cast<size_t>(t) * ASR_CH + c];
        }
    }

    auto cf_to_T_major = [&](const std::vector<float>& cf, int C, int Ti,
                             std::vector<float>& dst) {
        dst.assign(static_cast<size_t>(Ti) * C, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int tt = 0; tt < Ti; ++tt) {
                dst[static_cast<size_t>(tt) * C + c] =
                    cf[static_cast<size_t>(c) * Ti + tt];
            }
        }
    };

    // ---- 3. Build x = cat([asr, F0, N], dim=1) — (514, T_mel) ------------
    std::vector<float> x_cf(static_cast<size_t>(ASR_CH + 2) * T_mel, 0.0f);
    for (int c = 0; c < ASR_CH; ++c) {
        std::memcpy(&x_cf[static_cast<size_t>(c) * T_mel],
                    &asr_cf[static_cast<size_t>(c) * T_mel],
                    T_mel * sizeof(float));
    }
    std::memcpy(&x_cf[static_cast<size_t>(ASR_CH) * T_mel],
                F0_after.data(), T_mel * sizeof(float));
    std::memcpy(&x_cf[static_cast<size_t>(ASR_CH + 1) * T_mel],
                N_after.data(),  T_mel * sizeof(float));

    // ---- 4. encode block: 514 -> 1024 ------------------------------------
    std::vector<float> enc_out;
    int T_enc = 0;
    internal::run_adain_res_blk(encode_, style128, x_cf, T_mel,
                                enc_out, T_enc, LEAKY_SLOPE, INSTNORM_EPS);

    if (last == Stage::AfterEncode) {
        cf_to_T_major(enc_out, ENCODE_OUT_CH, T_enc, out);
        return;
    }

    // ---- 5. asr_res: 1x1 conv (channel-first) ---------------------------
    std::vector<float> asr_res_cf;
    internal::conv1d_padded(asr_cf.data(), ASR_CH, T_mel,
                            asr_res_.w.data(), asr_res_.b.data(),
                            ASR_RES_CH, /*K=*/1, /*pad=*/0, asr_res_cf);

    if (last == Stage::AsrRes) {
        cf_to_T_major(asr_res_cf, ASR_RES_CH, T_mel, out);
        return;
    }

    // ---- 6. Decode loop. Per the reference, `res` is True until the FIRST
    //         block with upsample_type != "none" fires. Only decode.3 has
    //         upsample, so all 4 blocks receive the cat at their input.
    // ---------------------------------------------------------------------
    std::vector<float> cur = enc_out;
    int Tcur = T_enc;
    int Ccur = ENCODE_OUT_CH;

    const int cat_in_ch = ENCODE_OUT_CH + ASR_RES_CH + 2;  // 1090
    const Stage stage_for[4] = {
        Stage::AfterDecode0, Stage::AfterDecode1,
        Stage::AfterDecode2, Stage::AfterDecode3,
    };

    for (int b = 0; b < 4; ++b) {
        // Build cat'd input: (cat_in_ch, Tcur). Channel ordering:
        //   [0,            1024) <- cur
        //   [1024,         1024+64) <- asr_res
        //   [1088, 1089)  <- F0_after
        //   [1089, 1090)  <- N_after
        std::vector<float> cat_in(static_cast<size_t>(cat_in_ch) * Tcur, 0.0f);
        // cur (Ccur x Tcur)
        std::memcpy(cat_in.data(), cur.data(),
                    static_cast<size_t>(Ccur) * Tcur * sizeof(float));
        // asr_res
        std::memcpy(&cat_in[static_cast<size_t>(Ccur) * Tcur],
                    asr_res_cf.data(),
                    static_cast<size_t>(ASR_RES_CH) * Tcur * sizeof(float));
        // F0_after
        std::memcpy(&cat_in[static_cast<size_t>(Ccur + ASR_RES_CH) * Tcur],
                    F0_after.data(), Tcur * sizeof(float));
        // N_after
        std::memcpy(&cat_in[static_cast<size_t>(Ccur + ASR_RES_CH + 1) * Tcur],
                    N_after.data(),  Tcur * sizeof(float));

        std::vector<float> nxt;
        int Tnxt = 0;
        internal::run_adain_res_blk(decode_[b], style128, cat_in, Tcur,
                                    nxt, Tnxt, LEAKY_SLOPE, INSTNORM_EPS);
        cur.swap(nxt);
        Tcur = Tnxt;
        Ccur = decode_[b].dim_out;

        if (last == stage_for[b]) {
            cf_to_T_major(cur, Ccur, Tcur, out);
            return;
        }
    }
    // Shouldn't reach here unless `last` is invalid; leave out empty.
}

void DecoderEncodeStage::forward(const float* asr, int T_mel,
                                  const float* F0, const float* N,
                                  const float* style128,
                                  std::vector<float>& out) const {
    run_until_stage(asr, T_mel, F0, N, style128, Stage::AfterDecode3, out);
}

}}  // namespace cactus::kokoro
