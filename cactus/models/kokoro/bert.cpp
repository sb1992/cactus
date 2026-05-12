#include "bert.h"

#include "weights_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

namespace {

// y[r, j] = sum_k x[r, k] * w[j, k] + b[j]   (PyTorch nn.Linear convention).
// x: (rows, in_dim), w: (out_dim, in_dim), b: (out_dim,) or nullptr.
inline void linear(const float* x, int rows, int in_dim,
                   const float* w, const float* b,
                   int out_dim, float* y) {
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < out_dim; ++j) {
            float acc = b ? b[j] : 0.0f;
            const float* xrow = x + static_cast<size_t>(r) * in_dim;
            const float* wrow = w + static_cast<size_t>(j) * in_dim;
            // Hand-unroll by 4 for a touch of throughput.
            int k = 0;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (; k + 4 <= in_dim; k += 4) {
                a0 += xrow[k    ] * wrow[k    ];
                a1 += xrow[k + 1] * wrow[k + 1];
                a2 += xrow[k + 2] * wrow[k + 2];
                a3 += xrow[k + 3] * wrow[k + 3];
            }
            acc += (a0 + a1) + (a2 + a3);
            for (; k < in_dim; ++k) acc += xrow[k] * wrow[k];
            y[static_cast<size_t>(r) * out_dim + j] = acc;
        }
    }
}

// Row-major LayerNorm over the last axis of a (rows, dim) buffer in place.
inline void layer_norm_last_axis(float* x, int rows, int dim,
                                 const float* gamma, const float* beta,
                                 float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<size_t>(r) * dim;
        double mean = 0.0;
        for (int d = 0; d < dim; ++d) mean += row[d];
        mean /= dim;
        double var = 0.0;
        for (int d = 0; d < dim; ++d) {
            const double t = row[d] - mean;
            var += t * t;
        }
        var /= dim;
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int d = 0; d < dim; ++d) {
            const float n = (row[d] - static_cast<float>(mean)) * inv;
            row[d] = n * gamma[d] + beta[d];
        }
    }
}

// gelu_new (HuggingFace activations.py "gelu_new"):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_new(float x) {
    constexpr float k = 0.7978845608028654f;   // sqrt(2/pi)
    const float x3 = x * x * x;
    const float t = std::tanh(k * (x + 0.044715f * x3));
    return 0.5f * x * (1.0f + t);
}

}  // namespace

// ----------------------------------------------------------------------
// load_weights
// ----------------------------------------------------------------------
bool Bert::load_weights(const std::string& dir) {
    auto load_named = [&](const std::string& base) -> std::vector<float> {
        return load_weights_file(dir + "/" + base + ".weights").data;
    };
    try {
        word_emb_     = load_named("bert_word_embeddings");
        pos_emb_      = load_named("bert_position_embeddings");
        tok_type_emb_ = load_named("bert_token_type_embeddings");
        emb_ln_gamma_ = load_named("bert_embeddings_norm_weight");
        emb_ln_beta_  = load_named("bert_embeddings_norm_bias");

        emb_hidden_w_ = load_named("bert_emb_hidden_mapping_weight");
        emb_hidden_b_ = load_named("bert_emb_hidden_mapping_bias");

        attn_q_w_     = load_named("bert_layer_0_attn_q_weight");
        attn_q_b_     = load_named("bert_layer_0_attn_q_bias");
        attn_k_w_     = load_named("bert_layer_0_attn_k_weight");
        attn_k_b_     = load_named("bert_layer_0_attn_k_bias");
        attn_v_w_     = load_named("bert_layer_0_attn_v_weight");
        attn_v_b_     = load_named("bert_layer_0_attn_v_bias");
        attn_dense_w_ = load_named("bert_layer_0_attn_dense_weight");
        attn_dense_b_ = load_named("bert_layer_0_attn_dense_bias");
        attn_ln_gamma_= load_named("bert_layer_0_attn_norm_weight");
        attn_ln_beta_ = load_named("bert_layer_0_attn_norm_bias");
        ffn_w_        = load_named("bert_layer_0_ffn_weight");
        ffn_b_        = load_named("bert_layer_0_ffn_bias");
        ffn_out_w_    = load_named("bert_layer_0_ffn_output_weight");
        ffn_out_b_    = load_named("bert_layer_0_ffn_output_bias");
        full_ln_gamma_= load_named("bert_layer_0_full_norm_weight");
        full_ln_beta_ = load_named("bert_layer_0_full_norm_bias");

        pooler_w_     = load_named("bert_pooler_weight");
        pooler_b_     = load_named("bert_pooler_bias");

        bert_encoder_w_ = load_named("bert_encoder_weight");
        bert_encoder_b_ = load_named("bert_encoder_bias");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Bert load_weights failed: %s\n", e.what());
        return false;
    }

    // Sanity checks.
    auto check_size = [&](const char* name, size_t got, size_t want) -> bool {
        if (got != want) {
            std::fprintf(stderr, "Bert: %s size mismatch got=%zu want=%zu\n",
                         name, got, want);
            return false;
        }
        return true;
    };
    if (!check_size("word_emb", word_emb_.size(),
                    static_cast<size_t>(VOCAB) * EMBEDDING_SIZE)) return false;
    if (!check_size("pos_emb", pos_emb_.size(),
                    static_cast<size_t>(MAX_POSITION) * EMBEDDING_SIZE)) return false;
    if (!check_size("tok_type_emb", tok_type_emb_.size(),
                    static_cast<size_t>(2) * EMBEDDING_SIZE)) return false;
    if (!check_size("emb_hidden_w", emb_hidden_w_.size(),
                    static_cast<size_t>(HIDDEN) * EMBEDDING_SIZE)) return false;
    if (!check_size("attn_q_w", attn_q_w_.size(),
                    static_cast<size_t>(HIDDEN) * HIDDEN)) return false;
    if (!check_size("ffn_w", ffn_w_.size(),
                    static_cast<size_t>(INTERMEDIATE) * HIDDEN)) return false;
    if (!check_size("ffn_out_w", ffn_out_w_.size(),
                    static_cast<size_t>(HIDDEN) * INTERMEDIATE)) return false;
    if (!check_size("bert_encoder_w", bert_encoder_w_.size(),
                    static_cast<size_t>(OUT_HIDDEN) * HIDDEN)) return false;

    return true;
}

// ----------------------------------------------------------------------
// embed_and_norm: (T,) i64 -> (T, 128) row-major (in `out128`)
//
// In KModel.forward_with_tokens, no token_type_ids are supplied, so HF
// AlbertEmbeddings creates an all-zero token_type tensor — i.e. every token
// adds row 0 of the token_type_embeddings matrix.
// position_ids = [0, 1, ..., T-1].
// Then LayerNorm with eps = 1e-12.
// ----------------------------------------------------------------------
void Bert::embed_and_norm(const int64_t* input_ids, int T,
                          std::vector<float>& out128) const {
    out128.assign(static_cast<size_t>(T) * EMBEDDING_SIZE, 0.0f);
    const float* tt0 = tok_type_emb_.data();   // row 0 (token_type=0)
    for (int t = 0; t < T; ++t) {
        int64_t id = input_ids[t];
        if (id < 0 || id >= VOCAB) id = 0;  // defensive
        const float* w = word_emb_.data()  + static_cast<size_t>(id) * EMBEDDING_SIZE;
        const float* p = pos_emb_.data()   + static_cast<size_t>(t)  * EMBEDDING_SIZE;
        float* dst     = out128.data()     + static_cast<size_t>(t)  * EMBEDDING_SIZE;
        for (int d = 0; d < EMBEDDING_SIZE; ++d) {
            dst[d] = w[d] + p[d] + tt0[d];
        }
    }
    layer_norm_last_axis(out128.data(), T, EMBEDDING_SIZE,
                         emb_ln_gamma_.data(), emb_ln_beta_.data(),
                         LAYER_NORM_EPS);
}

// ----------------------------------------------------------------------
// project_to_hidden: (T, 128) -> (T, 768)
// ----------------------------------------------------------------------
void Bert::project_to_hidden(const std::vector<float>& in128, int T,
                             std::vector<float>& out768) const {
    out768.assign(static_cast<size_t>(T) * HIDDEN, 0.0f);
    linear(in128.data(), T, EMBEDDING_SIZE,
           emb_hidden_w_.data(), emb_hidden_b_.data(),
           HIDDEN, out768.data());
}

// ----------------------------------------------------------------------
// run_albert_layer
//
//   Q = x @ Wq^T + bq
//   K = x @ Wk^T + bk
//   V = x @ Wv^T + bv
//   reshape to (H=12, T, D=64);  attn = softmax(Q @ K^T / sqrt(D)) @ V
//   reshape back to (T, 768)
//   proj = attn @ W_dense^T + b_dense
//   x = LayerNorm(proj + x)              # attention.LayerNorm
//   h = gelu_new(x @ W_ffn^T + b_ffn)    # (T, 2048)
//   h = h @ W_ffn_out^T + b_ffn_out      # (T, 768)
//   x = LayerNorm(h + x)                 # full_layer_layer_norm
// ----------------------------------------------------------------------
void Bert::run_albert_layer(std::vector<float>& x, int T) const {
    const size_t TH = static_cast<size_t>(T) * HIDDEN;

    // Q, K, V projections — each (T, HIDDEN).
    std::vector<float> Q(TH), K(TH), V(TH);
    linear(x.data(), T, HIDDEN, attn_q_w_.data(), attn_q_b_.data(), HIDDEN, Q.data());
    linear(x.data(), T, HIDDEN, attn_k_w_.data(), attn_k_b_.data(), HIDDEN, K.data());
    linear(x.data(), T, HIDDEN, attn_v_w_.data(), attn_v_b_.data(), HIDDEN, V.data());

    // Per-head scaled dot-product attention, accumulating into `attn`
    // (laid out as (T, HIDDEN) in head-major order matching HF reshape).
    std::vector<float> attn(TH, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // Scratch for per-head logits & softmax.
    std::vector<float> scores(static_cast<size_t>(T) * T);

    for (int h = 0; h < NUM_HEADS; ++h) {
        const int head_off = h * HEAD_DIM;

        // scores[i, j] = (Q[i, head_off:head_off+D] . K[j, head_off:head_off+D]) * scale
        for (int i = 0; i < T; ++i) {
            const float* qi = Q.data() + static_cast<size_t>(i) * HIDDEN + head_off;
            float* row = scores.data() + static_cast<size_t>(i) * T;
            for (int j = 0; j < T; ++j) {
                const float* kj = K.data() + static_cast<size_t>(j) * HIDDEN + head_off;
                float s = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) s += qi[d] * kj[d];
                row[j] = s * scale;
            }
            // softmax over j.
            float m = row[0];
            for (int j = 1; j < T; ++j) if (row[j] > m) m = row[j];
            float sum = 0.0f;
            for (int j = 0; j < T; ++j) {
                row[j] = std::exp(row[j] - m);
                sum += row[j];
            }
            const float inv = 1.0f / sum;
            for (int j = 0; j < T; ++j) row[j] *= inv;
        }

        // attn[i, head_off:head_off+D] = sum_j scores[i, j] * V[j, head_off:head_off+D]
        for (int i = 0; i < T; ++i) {
            const float* row = scores.data() + static_cast<size_t>(i) * T;
            float* out = attn.data() + static_cast<size_t>(i) * HIDDEN + head_off;
            for (int d = 0; d < HEAD_DIM; ++d) out[d] = 0.0f;
            for (int j = 0; j < T; ++j) {
                const float a = row[j];
                if (a == 0.0f) continue;
                const float* v = V.data() + static_cast<size_t>(j) * HIDDEN + head_off;
                for (int d = 0; d < HEAD_DIM; ++d) out[d] += a * v[d];
            }
        }
    }

    // proj = attn @ W_dense^T + b_dense
    std::vector<float> proj(TH);
    linear(attn.data(), T, HIDDEN, attn_dense_w_.data(), attn_dense_b_.data(),
           HIDDEN, proj.data());

    // x = LayerNorm(proj + x)  (attention.LayerNorm)
    for (size_t i = 0; i < TH; ++i) x[i] += proj[i];
    layer_norm_last_axis(x.data(), T, HIDDEN,
                         attn_ln_gamma_.data(), attn_ln_beta_.data(),
                         LAYER_NORM_EPS);

    // FFN: gelu_new(x @ W_ffn^T + b_ffn)  -> (T, 2048)
    std::vector<float> h(static_cast<size_t>(T) * INTERMEDIATE);
    linear(x.data(), T, HIDDEN, ffn_w_.data(), ffn_b_.data(), INTERMEDIATE, h.data());
    for (size_t i = 0; i < h.size(); ++i) h[i] = gelu_new(h[i]);

    // ffn_output: h @ W_out^T + b_out  -> (T, 768)
    std::vector<float> h_out(TH);
    linear(h.data(), T, INTERMEDIATE, ffn_out_w_.data(), ffn_out_b_.data(),
           HIDDEN, h_out.data());

    // x = LayerNorm(h_out + x)  (full_layer_layer_norm)
    for (size_t i = 0; i < TH; ++i) x[i] += h_out[i];
    layer_norm_last_axis(x.data(), T, HIDDEN,
                         full_ln_gamma_.data(), full_ln_beta_.data(),
                         LAYER_NORM_EPS);
}

// ----------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------
void Bert::forward(const int64_t* input_ids, int T,
                   std::vector<float>& out) const {
    std::vector<float> e128;
    embed_and_norm(input_ids, T, e128);

    project_to_hidden(e128, T, out);  // (T, HIDDEN) row-major

    for (int l = 0; l < NUM_LAYERS; ++l) {
        run_albert_layer(out, T);
    }
}

void Bert::apply_bert_encoder(const float* bert_dur, int T,
                              std::vector<float>& d_en) const {
    d_en.assign(static_cast<size_t>(T) * OUT_HIDDEN, 0.0f);
    linear(bert_dur, T, HIDDEN,
           bert_encoder_w_.data(), bert_encoder_b_.data(),
           OUT_HIDDEN, d_en.data());
}

void Bert::apply_pooler(const float* bert_dur, int T,
                        std::vector<float>& pooled) const {
    (void)T;  // pooler only sees the [CLS] (first) token.
    pooled.assign(HIDDEN, 0.0f);
    linear(bert_dur, 1, HIDDEN, pooler_w_.data(), pooler_b_.data(),
           HIDDEN, pooled.data());
    for (auto& v : pooled) v = std::tanh(v);
}

}}  // namespace cactus::kokoro
