#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cactus { namespace kokoro {

// Kokoro `bert` (PLBERT — a HuggingFace ALBERT with a single shared layer
// applied 12 times). See assets/kokoro/kokoromodel_pipeline.md for the full
// architectural breakdown and assets/kokoro/weights/bert_*.weights for the
// 25 tensor files this loads.
//
// Shapes (Kokoro v0_19 plbert config):
//   embedding_size    = 128
//   hidden_size       = 768
//   intermediate_size = 2048
//   num_attention_heads = 12 (head_dim = 64)
//   num_hidden_layers   = 12 (single shared layer applied 12 times)
//   max_position_embeddings = 512
//   layer_norm_eps    = 1e-12
//   hidden_act        = "gelu_new"   (tanh-approx GELU)
//
// Inference-only, batch=1, no padding mask (KModel only ever runs B=1 with
// unpadded input — see KModel.forward_with_tokens, model.py:86-119).
class Bert {
public:
    bool load_weights(const std::string& weights_dir);

    // Full forward.
    //   input_ids: (T,) i64, KModel-padded (leading + trailing 0).
    //   out:       resized to T*HIDDEN row-major (T, 768).
    //              Equivalent to PyTorch CustomAlbert.forward(input_ids).last_hidden_state.
    void forward(const int64_t* input_ids, int T,
                 std::vector<float>& out) const;

    // bert_encoder linear projection (768 -> 512), applied to (T, 768) row-major.
    //   bert_dur: (T, 768) row-major
    //   d_en:     resized to (T, 512) row-major
    void apply_bert_encoder(const float* bert_dur, int T,
                            std::vector<float>& d_en) const;

    // Pooler — tanh(W * x[:, 0, :] + b). Returned only for per-stage testing
    // (KModel does NOT use the pooler output downstream).
    //   bert_dur: (T, 768) row-major
    //   pooled:   resized to (768,)
    void apply_pooler(const float* bert_dur, int T,
                      std::vector<float>& pooled) const;

    static constexpr int VOCAB             = 178;
    static constexpr int EMBEDDING_SIZE    = 128;
    static constexpr int HIDDEN            = 768;
    static constexpr int INTERMEDIATE      = 2048;
    static constexpr int NUM_HEADS         = 12;
    static constexpr int HEAD_DIM          = HIDDEN / NUM_HEADS;   // 64
    static constexpr int NUM_LAYERS        = 12;
    static constexpr int MAX_POSITION      = 512;
    static constexpr int OUT_HIDDEN        = 512;                  // bert_encoder output dim
    static constexpr float LAYER_NORM_EPS  = 1e-12f;

private:
    // ---- embeddings -----------------------------------------------------
    std::vector<float> word_emb_;       // (VOCAB,            EMBEDDING_SIZE)
    std::vector<float> pos_emb_;        // (MAX_POSITION,     EMBEDDING_SIZE)
    std::vector<float> tok_type_emb_;   // (2,                EMBEDDING_SIZE)
    std::vector<float> emb_ln_gamma_;   // (EMBEDDING_SIZE,)
    std::vector<float> emb_ln_beta_;    // (EMBEDDING_SIZE,)

    // ---- 128 -> 768 projection -----------------------------------------
    std::vector<float> emb_hidden_w_;   // (HIDDEN, EMBEDDING_SIZE) PyTorch Linear.weight
    std::vector<float> emb_hidden_b_;   // (HIDDEN,)

    // ---- single shared albert layer (applied NUM_LAYERS times) ---------
    std::vector<float> attn_q_w_;       // (HIDDEN, HIDDEN)
    std::vector<float> attn_q_b_;       // (HIDDEN,)
    std::vector<float> attn_k_w_;
    std::vector<float> attn_k_b_;
    std::vector<float> attn_v_w_;
    std::vector<float> attn_v_b_;
    std::vector<float> attn_dense_w_;   // (HIDDEN, HIDDEN)
    std::vector<float> attn_dense_b_;   // (HIDDEN,)
    std::vector<float> attn_ln_gamma_;  // (HIDDEN,)  - attention.LayerNorm
    std::vector<float> attn_ln_beta_;   // (HIDDEN,)
    std::vector<float> ffn_w_;          // (INTERMEDIATE, HIDDEN)
    std::vector<float> ffn_b_;          // (INTERMEDIATE,)
    std::vector<float> ffn_out_w_;      // (HIDDEN, INTERMEDIATE)
    std::vector<float> ffn_out_b_;      // (HIDDEN,)
    std::vector<float> full_ln_gamma_;  // (HIDDEN,)  - full_layer_layer_norm
    std::vector<float> full_ln_beta_;   // (HIDDEN,)

    // ---- pooler ---------------------------------------------------------
    std::vector<float> pooler_w_;       // (HIDDEN, HIDDEN)
    std::vector<float> pooler_b_;       // (HIDDEN,)

    // ---- bert_encoder (Kokoro KModel.bert_encoder Linear 768 -> 512) ----
    std::vector<float> bert_encoder_w_; // (OUT_HIDDEN, HIDDEN)
    std::vector<float> bert_encoder_b_; // (OUT_HIDDEN,)

    // ---- forward sub-steps --------------------------------------------------
    // Compute the (T, EMBEDDING_SIZE=128) embedding sum + LayerNorm in place.
    void embed_and_norm(const int64_t* input_ids, int T,
                        std::vector<float>& out128) const;

    // Project (T, 128) -> (T, 768) via emb_hidden_mapping.
    void project_to_hidden(const std::vector<float>& in128, int T,
                           std::vector<float>& out768) const;

    // Run one albert layer on x (T, 768) in place.
    //   scratch_q, scratch_k, scratch_v, scratch_attn — reusable buffers.
    void run_albert_layer(std::vector<float>& x, int T) const;
};

}}  // namespace cactus::kokoro
