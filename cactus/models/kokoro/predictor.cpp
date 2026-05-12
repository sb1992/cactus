#include "predictor.h"

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

inline float sigmoidf(float x) {
    // Stable sigmoid; predictor logits are bounded (sum-of-50 < ~50).
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = std::exp(x);
        return z / (1.0f + z);
    }
}

}  // namespace

// ----------------------------------------------------------------------
// load_weights
// ----------------------------------------------------------------------
bool Predictor::load_weights(const std::string& dir) {
    auto load_named = [&](const std::string& base) -> std::vector<float> {
        return load_weights_file(dir + "/predictor_" + base + ".weights").data;
    };

    auto load_lstm_dir = [&](LstmDir& d, const std::string& base, int H, int I) {
        d.w_ih = load_named(base + "_weight_ih");
        d.w_hh = load_named(base + "_weight_hh");
        d.b_ih = load_named(base + "_bias_ih");
        d.b_hh = load_named(base + "_bias_hh");
        const int n_4H = 4 * H;
        if (static_cast<int>(d.w_ih.size()) != n_4H * I ||
            static_cast<int>(d.w_hh.size()) != n_4H * H ||
            static_cast<int>(d.b_ih.size()) != n_4H ||
            static_cast<int>(d.b_hh.size()) != n_4H) {
            throw std::runtime_error("LSTM weight size mismatch in " + base);
        }
    };

    auto load_bilstm = [&](BiLSTM& bi, const std::string& base, int H, int I) {
        load_lstm_dir(bi.fwd, base + "_lstm_fwd_0", H, I);
        load_lstm_dir(bi.bwd, base + "_lstm_bwd_0", H, I);
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

    try {
        // ----- DurationEncoder -----
        // Layers 0/2/4 are BiLSTMs, 1/3/5 are AdaLayerNorms.
        for (int i = 0; i < 3; ++i) {
            const std::string lstm_base = "text_encoder_lstms_" + std::to_string(2*i);
            load_bilstm(dur_enc_lstm_[i], lstm_base, LSTM_HIDDEN, D_CAT);
            const std::string aln_base = "text_encoder_lstms_" + std::to_string(2*i + 1) + "_fc";
            load_aln_fc(dur_enc_aln_[i], aln_base, D_HID);
        }

        // ----- Top BiLSTM + duration_proj -----
        load_bilstm(top_lstm_, "lstm", LSTM_HIDDEN, D_CAT);
        dur_proj_w_ = load_named("duration_proj_linear_layer_weight");
        dur_proj_b_ = load_named("duration_proj_linear_layer_bias");
        if (static_cast<int>(dur_proj_w_.size()) != MAX_DUR * D_HID ||
            static_cast<int>(dur_proj_b_.size()) != MAX_DUR) {
            throw std::runtime_error("duration_proj size mismatch");
        }

        // ----- Shared BiLSTM -----
        load_bilstm(shared_lstm_, "shared", LSTM_HIDDEN, D_CAT);

        // ----- F0 / N branches -----
        auto load_resblk = [&](AdainResBlk1d& blk, const std::string& base,
                               int dim_in, int dim_out, bool upsample) {
            blk.dim_in = dim_in;
            blk.dim_out = dim_out;
            blk.upsample = upsample;
            blk.learned_sc = (dim_in != dim_out);

            load_aln_fc(blk.norm1_fc, base + "_norm1_fc", dim_in);
            load_aln_fc(blk.norm2_fc, base + "_norm2_fc", dim_out);
            // conv1: (dim_out, dim_in, 3, p=1, stride=1)
            load_conv(blk.conv1, base + "_conv1", dim_out, dim_in, 3, true);
            // conv2: (dim_out, dim_out, 3, p=1, stride=1)
            load_conv(blk.conv2, base + "_conv2", dim_out, dim_out, 3, true);
            if (upsample) {
                // pool: ConvTranspose1d(dim_in, dim_in, K=3, s=2, groups=dim_in,
                //                       padding=1, output_padding=1)
                // Stored weight shape: (dim_in, 1, K=3) — PyTorch ConvTranspose1d
                // with groups=in stores as (in_channels, out_channels/groups, K).
                // Bias shape: (dim_in,)
                blk.pool.w = load_named(base + "_pool_weight");
                blk.pool.b = load_named(base + "_pool_bias");
                if (static_cast<int>(blk.pool.w.size()) != dim_in * 1 * 3) {
                    throw std::runtime_error(base + " pool weight size mismatch");
                }
                if (static_cast<int>(blk.pool.b.size()) != dim_in) {
                    throw std::runtime_error(base + " pool bias size mismatch");
                }
                blk.pool.C_in = dim_in;
                blk.pool.C_out = dim_in;
                blk.pool.K = 3;
            }
            if (blk.learned_sc) {
                // 1x1 conv, no bias: (dim_out, dim_in, 1)
                load_conv(blk.conv1x1, base + "_conv1x1", dim_out, dim_in, 1, false);
            }
        };

        load_resblk(F0_blocks_[0], "F0_0", 512, 512, false);
        load_resblk(F0_blocks_[1], "F0_1", 512, 256, true);
        load_resblk(F0_blocks_[2], "F0_2", 256, 256, false);
        load_resblk(N_blocks_[0],  "N_0",  512, 512, false);
        load_resblk(N_blocks_[1],  "N_1",  512, 256, true);
        load_resblk(N_blocks_[2],  "N_2",  256, 256, false);

        load_conv(F0_proj_, "F0_proj", 1, 256, 1, true);
        load_conv(N_proj_,  "N_proj",  1, 256, 1, true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Predictor load_weights failed: %s\n", e.what());
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------
// run_until_stage — main pipeline
// ----------------------------------------------------------------------
void Predictor::run_until_stage(const int64_t* /*phonemes*/, int T,
                                 const float* style128,
                                 const float* d_en,
                                 Stage last,
                                 std::vector<float>& out_buf,
                                 std::vector<int64_t>* out_durations) const {
    // ---- DurationEncoder -----------------------------------------------
    // Input d_en is (T, 512) row-major. Convert to channel-first (640, T)
    // with style appended, matching the reference cat-then-permute path.
    std::vector<float> x(static_cast<size_t>(D_CAT) * T, 0.0f);
    // Channel 0..511 -> d_en[t, c]; Channel 512..639 -> style128[c-512]
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < D_HID; ++c) {
            x[static_cast<size_t>(c) * T + t] =
                d_en[static_cast<size_t>(t) * D_HID + c];
        }
        for (int c = 0; c < STYLE_DIM; ++c) {
            x[static_cast<size_t>(D_HID + c) * T + t] = style128[c];
        }
    }

    auto write_T_HID_view = [&](const std::vector<float>& cf, int Cdim,
                                 std::vector<float>& dst) {
        dst.assign(static_cast<size_t>(T) * Cdim, 0.0f);
        for (int c = 0; c < Cdim; ++c) {
            for (int tt = 0; tt < T; ++tt) {
                dst[static_cast<size_t>(tt) * Cdim + c] =
                    cf[static_cast<size_t>(c) * T + tt];
            }
        }
    };

    // Run alternating LSTM / AdaLN stages
    for (int i = 0; i < 3; ++i) {
        // ---- BiLSTM ----
        // Input: cf (640, T) -> seq (T, 640) -> bilstm -> (T, 512) -> cf (512, T)
        std::vector<float> seq(static_cast<size_t>(T) * D_CAT, 0.0f);
        for (int c = 0; c < D_CAT; ++c) {
            for (int tt = 0; tt < T; ++tt) {
                seq[static_cast<size_t>(tt) * D_CAT + c] =
                    x[static_cast<size_t>(c) * T + tt];
            }
        }
        std::vector<float> lstm_out;
        internal::bidirectional_lstm(seq.data(), T, D_CAT, LSTM_HIDDEN,
            dur_enc_lstm_[i].fwd.w_ih.data(), dur_enc_lstm_[i].fwd.w_hh.data(),
            dur_enc_lstm_[i].fwd.b_ih.data(), dur_enc_lstm_[i].fwd.b_hh.data(),
            dur_enc_lstm_[i].bwd.w_ih.data(), dur_enc_lstm_[i].bwd.w_hh.data(),
            dur_enc_lstm_[i].bwd.b_ih.data(), dur_enc_lstm_[i].bwd.b_hh.data(),
            lstm_out);
        // lstm_out: (T, 512) row-major -> back to channel-first (512, T)
        x.assign(static_cast<size_t>(D_HID) * T, 0.0f);
        for (int tt = 0; tt < T; ++tt) {
            for (int c = 0; c < D_HID; ++c) {
                x[static_cast<size_t>(c) * T + tt] =
                    lstm_out[static_cast<size_t>(tt) * D_HID + c];
            }
        }

        // Capture stage if requested (T-major: (T, 512))
        if ((i == 0 && last == Stage::DurEncLstm0) ||
            (i == 1 && last == Stage::DurEncLstm2) ||
            (i == 2 && last == Stage::DurEncLstm4)) {
            write_T_HID_view(x, D_HID, out_buf);
            return;
        }

        // ---- AdaLayerNorm (in place on channel-first (512, T)) ----
        internal::ada_layer_norm(x.data(), D_HID, T, style128,
                                 dur_enc_aln_[i], LAYERNORM_EPS);

        // Capture post-AdaLN BEFORE re-cat (matches fixture convention)
        if ((i == 0 && last == Stage::DurEncAln1) ||
            (i == 1 && last == Stage::DurEncAln3) ||
            (i == 2 && last == Stage::DurEncAln5)) {
            write_T_HID_view(x, D_HID, out_buf);
            return;
        }

        // Re-cat style: (512, T) -> (640, T)
        std::vector<float> x_new(static_cast<size_t>(D_CAT) * T, 0.0f);
        for (int c = 0; c < D_HID; ++c) {
            std::memcpy(&x_new[static_cast<size_t>(c) * T],
                        &x[static_cast<size_t>(c) * T],
                        T * sizeof(float));
        }
        for (int c = 0; c < STYLE_DIM; ++c) {
            for (int tt = 0; tt < T; ++tt) {
                x_new[static_cast<size_t>(D_HID + c) * T + tt] = style128[c];
            }
        }
        x.swap(x_new);
    }

    // After loop, x is (640, T). Final transpose to (T, 640) gives `d`.
    std::vector<float> d(static_cast<size_t>(T) * D_CAT, 0.0f);
    for (int c = 0; c < D_CAT; ++c) {
        for (int tt = 0; tt < T; ++tt) {
            d[static_cast<size_t>(tt) * D_CAT + c] =
                x[static_cast<size_t>(c) * T + tt];
        }
    }
    if (last == Stage::D) { out_buf = d; return; }

    // ---- Top BiLSTM: (T, 640) -> (T, 512) ------------------------------
    std::vector<float> top_out;
    internal::bidirectional_lstm(d.data(), T, D_CAT, LSTM_HIDDEN,
        top_lstm_.fwd.w_ih.data(), top_lstm_.fwd.w_hh.data(),
        top_lstm_.fwd.b_ih.data(), top_lstm_.fwd.b_hh.data(),
        top_lstm_.bwd.w_ih.data(), top_lstm_.bwd.w_hh.data(),
        top_lstm_.bwd.b_ih.data(), top_lstm_.bwd.b_hh.data(),
        top_out);
    if (last == Stage::TopLstm) { out_buf = top_out; return; }

    // ---- duration_proj: Linear(512 -> 50) ------------------------------
    std::vector<float> dur_logits(static_cast<size_t>(T) * MAX_DUR, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < MAX_DUR; ++o) {
            float acc = dur_proj_b_[o];
            const float* w_row = &dur_proj_w_[static_cast<size_t>(o) * D_HID];
            const float* in_row = &top_out[static_cast<size_t>(t) * D_HID];
            for (int k = 0; k < D_HID; ++k) acc += w_row[k] * in_row[k];
            dur_logits[static_cast<size_t>(t) * MAX_DUR + o] = acc;
        }
    }
    if (last == Stage::DurationProj) { out_buf = dur_logits; return; }

    // ---- sigmoid + sum + clamp + round -> pred_dur ---------------------
    std::vector<int64_t> pred_dur(T, 0);
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        for (int o = 0; o < MAX_DUR; ++o) {
            sum += sigmoidf(dur_logits[static_cast<size_t>(t) * MAX_DUR + o]);
        }
        // speed = 1.0 (kokoro default; not configurable here)
        double rounded = std::round(sum);
        if (rounded < 1.0) rounded = 1.0;
        pred_dur[t] = static_cast<int64_t>(rounded);
    }
    if (last == Stage::Durations) {
        if (out_durations) *out_durations = pred_dur;
        out_buf.clear();
        return;
    }

    // ---- alignment matrix -----------------------------------------------
    int T_mel = 0;
    for (int t = 0; t < T; ++t) T_mel += static_cast<int>(pred_dur[t]);
    std::vector<float> aln(static_cast<size_t>(T) * T_mel, 0.0f);
    {
        int frame = 0;
        for (int t = 0; t < T; ++t) {
            for (int64_t k = 0; k < pred_dur[t]; ++k) {
                aln[static_cast<size_t>(t) * T_mel + frame] = 1.0f;
                ++frame;
            }
        }
    }
    if (last == Stage::Alignment) { out_buf = aln; return; }

    // ---- en = d.T @ aln  -- result shape (640, T_mel), stored T-major (T_mel, 640)
    // d is (T, 640) row-major. d.T is (640, T). aln is (T, T_mel).
    // out[c, m] = sum_t d[t, c] * aln[t, m]
    std::vector<float> en_cf(static_cast<size_t>(D_CAT) * T_mel, 0.0f);  // (640, T_mel)
    for (int c = 0; c < D_CAT; ++c) {
        for (int m = 0; m < T_mel; ++m) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) {
                acc += d[static_cast<size_t>(t) * D_CAT + c] *
                       aln[static_cast<size_t>(t) * T_mel + m];
            }
            en_cf[static_cast<size_t>(c) * T_mel + m] = static_cast<float>(acc);
        }
    }
    if (last == Stage::En) {
        // Store as (T_mel, 640) row-major to match fixture
        out_buf.assign(static_cast<size_t>(T_mel) * D_CAT, 0.0f);
        for (int c = 0; c < D_CAT; ++c) {
            for (int m = 0; m < T_mel; ++m) {
                out_buf[static_cast<size_t>(m) * D_CAT + c] =
                    en_cf[static_cast<size_t>(c) * T_mel + m];
            }
        }
        return;
    }

    // ---- shared BiLSTM on en.transpose(-1,-2) -> (T_mel, 640) -> (T_mel, 512)
    std::vector<float> en_seq(static_cast<size_t>(T_mel) * D_CAT, 0.0f);
    for (int c = 0; c < D_CAT; ++c) {
        for (int m = 0; m < T_mel; ++m) {
            en_seq[static_cast<size_t>(m) * D_CAT + c] =
                en_cf[static_cast<size_t>(c) * T_mel + m];
        }
    }
    std::vector<float> shared_out;
    internal::bidirectional_lstm(en_seq.data(), T_mel, D_CAT, LSTM_HIDDEN,
        shared_lstm_.fwd.w_ih.data(), shared_lstm_.fwd.w_hh.data(),
        shared_lstm_.fwd.b_ih.data(), shared_lstm_.fwd.b_hh.data(),
        shared_lstm_.bwd.w_ih.data(), shared_lstm_.bwd.w_hh.data(),
        shared_lstm_.bwd.b_ih.data(), shared_lstm_.bwd.b_hh.data(),
        shared_out);
    if (last == Stage::SharedLstm) { out_buf = shared_out; return; }

    // ---- Branch point: x_branch = shared.transpose(-1,-2) -> (512, T_mel)
    std::vector<float> x_branch(static_cast<size_t>(D_HID) * T_mel, 0.0f);
    for (int m = 0; m < T_mel; ++m) {
        for (int c = 0; c < D_HID; ++c) {
            x_branch[static_cast<size_t>(c) * T_mel + m] =
                shared_out[static_cast<size_t>(m) * D_HID + c];
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

    auto run_branch = [&](const AdainResBlk1d* blocks, const Conv1d& proj,
                          Stage block0_stage, Stage block1_stage,
                          Stage block2_stage, Stage pred_stage,
                          std::vector<float>& branch_out_pred) -> bool {
        // returns true if we returned early (last matched a branch stage)
        std::vector<float> cur = x_branch;
        int Tcur = T_mel;
        for (int b = 0; b < 3; ++b) {
            std::vector<float> nxt;
            int Tnxt = 0;
            internal::run_adain_res_blk(blocks[b], style128, cur, Tcur,
                                        nxt, Tnxt, LEAKY_SLOPE, INSTNORM_EPS);
            cur.swap(nxt);
            Tcur = Tnxt;
            const Stage stage_for_b =
                (b == 0) ? block0_stage : (b == 1) ? block1_stage : block2_stage;
            if (last == stage_for_b) {
                cf_to_T_major(cur, blocks[b].dim_out, Tcur, out_buf);
                return true;
            }
        }
        // Final 1x1 projection (256 -> 1)
        std::vector<float> pred_cf;
        internal::conv1d_padded(cur.data(), proj.C_in, Tcur,
                                proj.w.data(), proj.b.data(),
                                proj.C_out, proj.K, 0, pred_cf);
        // squeeze(1): (1, 1, Tcur) -> (Tcur,)
        branch_out_pred.assign(pred_cf.begin(), pred_cf.end());
        if (last == pred_stage) {
            out_buf = branch_out_pred;
            return true;
        }
        return false;
    };

    // F0 branch
    std::vector<float> F0_pred;
    if (run_branch(F0_blocks_, F0_proj_,
                   Stage::F0Block0, Stage::F0Block1,
                   Stage::F0Block2, Stage::F0Pred, F0_pred)) {
        return;
    }
    // N branch
    std::vector<float> N_pred;
    if (run_branch(N_blocks_, N_proj_,
                   Stage::NBlock0, Stage::NBlock1,
                   Stage::NBlock2, Stage::NPred, N_pred)) {
        return;
    }
    // Shouldn't reach here unless `last` is invalid; leave out_buf empty.
}

void Predictor::forward(const int64_t* phonemes, int T,
                         const float* style128,
                         const float* d_en,
                         Output& out) const {
    // We need F0_pred, N_pred, and durations. Run by hand.
    out.durations.clear();
    out.F0.clear();
    out.N.clear();

    // Get durations and (T, 640) d via partial.
    std::vector<float> d_buf;
    run_until_stage(phonemes, T, style128, d_en, Stage::D, d_buf);

    std::vector<int64_t> dur;
    std::vector<float> ignore;
    run_until_stage(phonemes, T, style128, d_en, Stage::Durations, ignore, &dur);
    out.durations = dur;
    out.T_mel = 0;
    for (int64_t v : dur) out.T_mel += static_cast<int>(v);

    // Run F0 + N branches separately.
    std::vector<float> f0_buf, n_buf;
    run_until_stage(phonemes, T, style128, d_en, Stage::F0Pred, f0_buf);
    run_until_stage(phonemes, T, style128, d_en, Stage::NPred,  n_buf);
    out.F0 = f0_buf;
    out.N  = n_buf;
}

}}  // namespace cactus::kokoro
