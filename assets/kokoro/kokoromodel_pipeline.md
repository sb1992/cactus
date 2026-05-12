# KokoroModel pipeline (cactus C++)

Top-level wiring of the Kokoro TTS pipeline as implemented in
`cactus::kokoro::KokoroModel`. This document is the source of truth for the
exact data flow and tensor shapes traveling between sub-modules. The reference
PyTorch implementation lives in
`/opt/homebrew/lib/python3.11/site-packages/kokoro/model.py`
(`KModel.forward_with_tokens`, lines 86-119).

## High-level flow

```
text (str)
   ├─ G2P.phonemize         → phoneme strings
   └─ vocab[ph] for ph       → raw_ids   (list[int])
                               + KModel pads with leading + trailing 0:
                               input_ids = [0, *raw_ids, 0]   (T = len(raw_ids) + 2)
input_ids (T,)  i64
   │
   ├──────────────────────────────────────────────────────────────┐
   │                                                              │
   ▼                                                              ▼
TextEncoder.forward                                       Bert(input_ids)
  in:  input_ids (T,)                                       in:  input_ids (T,)
  out: t_en (T, 512) row-major                              out: bert_dur (T, 768) row-major
                                                                 │
                                                                 ▼
                                                          bert_encoder (Linear 768→512)
                                                            d_en (T, 512) row-major

Voice embedding (511, 1, 256) loaded from voices.bin (af_bella):
  ref_s = voice_emb[T - 1]                  shape (1, 256)
  s_predictor = ref_s[:, 128:]              shape (128,)   — feeds predictor + decoder.encode
  s_decoder   = ref_s[:, :128]              shape (128,)   — feeds decoder + generator (style128)

Predictor.forward(d_en, s_predictor)
  in:  d_en (T, 512), s (128,)
  out: durations (T,)         i64,  pred_dur per phoneme
       T_mel = sum(durations)
       F0    (2*T_mel,)       f32  (note: predictor's internal upsample = 2x)
       N     (2*T_mel,)       f32

Alignment (built inside Predictor.forward and re-applied here):
  pred_aln_trg (T, T_mel) — one-hot expansion of phonemes to mel frames

asr (T_mel, 512):
  asr = t_en (T, 512)^T  @  pred_aln_trg (T, T_mel)        # row-major
                                                            # i.e. asr[m, c] = sum_p t_en[p, c] * pred_aln[p, m]

DecoderEncodeStage.forward(asr, F0, N, s_decoder)
  in:  asr (T_mel, 512), F0 (2*T_mel,), N (2*T_mel,), style128 (128,)
  out: x_decode3 (2*T_mel, 512) row-major

Generator.forward(x_decode3, F0, style128, [seed])
  in:  x_decode3 (2*T_mel, 512), F0 (2*T_mel,), style128 (128,)
  out: audio (600 * T_mel,)     f32   (24 kHz mono PCM)
```

## Stage-by-stage tensor shapes (concrete for p01)

| Stage                            | Shape (rank-major)         | Comment                              |
|----------------------------------|----------------------------|--------------------------------------|
| input_ids                        | `(T=28,)`                  | KModel pads with leading + trailing 0|
| TextEncoder embedding lookup     | `(T, 512)`                 | row-major                            |
| TextEncoder CNN x3               | `(512, T)` channel-first   | ops in CF; transposed back at end    |
| TextEncoder LSTM (`t_en`)        | `(T, 512)`                 | bidirectional concat                  |
| Bert embeddings.{word,pos,tok_type} | `(T, 128)` (sum)        | embedding_size = 128                 |
| Bert embeddings.LayerNorm        | `(T, 128)`                 | eps = 1e-12                          |
| Bert embedding_hidden_mapping_in | `(T, 768)`                 | Linear 128→768                       |
| Bert albert layer (×12)          | `(T, 768)`                 | shared weights, applied 12 times     |
| Bert pooler                      | `(768,)`                   | `tanh(W·x[:,0,:])` — UNUSED downstream|
| `bert_dur` (KModel `last_hidden_state`)| `(T, 768)`           | post 12-layer stack                  |
| `bert_encoder` Linear 768→512    | `(T, 512)`                 | row-major                            |
| `d_en` (`d_en.transpose(-1,-2)`) | `(T, 512)`                 | predictor expects channel-LAST       |
| Predictor durations              | `(T,)`  i64                | `round(sigmoid(50d).sum(-1)/speed)`  |
| Predictor F0_pred / N_pred       | `(2*T_mel,)`               | predictor branches upsample 2x       |
| Alignment matrix                 | `(T, T_mel)` f32 0/1       | `repeat_interleave(arange(T), durations)` indices|
| `asr` (text_encoder × alignment) | `(T_mel, 512)`             | `(t_en^T @ aln)^T` row-major         |
| Decoder encode→decode4           | `(2*T_mel, 512)`           | matches generator input              |
| Generator audio                  | `(600 * T_mel,)`           | 24 kHz                               |

For p01, T = 28, T_mel = 112, audio length = 67200 samples (= 2.8 s @ 24 kHz).

## Subtle details (from reading model.py)

1. **Padding**: `KModel.forward` pads phoneme IDs with one leading and one
   trailing `0` token (vocab id 0 = `$` in the Kokoro vocab). Both BERT and
   text_encoder receive the *same* padded input_ids; they are NOT processed
   separately.

2. **`text_mask`**: For B=1 inference (which is all KModel ever does), there
   is no padding within the sequence — `input_lengths == T == input_ids.shape[-1]`,
   so `text_mask = (arange(T) + 1) > T` is all-False. The cactus port does not
   need to thread `text_mask` between modules.

3. **bert receives `attention_mask=(~text_mask).int()`** which is all-ones in
   the B=1 path. This means standard self-attention with no masking.

4. **bert_encoder** is a single `nn.Linear(768, 512)`, NOT a Conv1d. From
   `model.py:53`: `self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])`.
   The cactus port stores the weight as `(512, 768)` row-major (PyTorch's
   `Linear.weight` convention) and bias as `(512,)`.

5. **The `transpose(-1, -2)` after `bert_encoder(bert_dur)`** flips
   `(1, T, 512) → (1, 512, T)` so `d_en` is channel-LAST in time. Predictor's
   input expects this exact layout — see `Predictor::forward` in
   `cactus/models/kokoro/predictor.h` which documents `d_en (T, 512)` row-major
   (which is the same memory as `(512, T)` once you read it the other way).

6. **`asr` input to decoder**: `t_en @ pred_aln_trg` (model.py:115). Note:
   - `t_en` from text_encoder is shape `(1, T, 512)` *channel-LAST in time* but
     internally text_encoder.forward returns it transposed — see the source:
     ```
     x = x.transpose(-1, -2)
     x_pad[:, :, :x.shape[-1]] = x
     ```
     The TextEncoder fixture stores `(T, 512)` row-major. To produce `asr`,
     we treat that as `(C=512, T)` and right-multiply by `(T, T_mel)`
     alignment, yielding `(C=512, T_mel)`. The cactus DecoderEncodeStage takes
     `asr (T_mel, 512)` row-major, so we transpose back at the boundary.

7. **`en` for predictor F0Ntrain**: `d.transpose(-1, -2) @ aln` is built
   *inside* `Predictor::forward` already — the cactus Predictor handles the
   alignment internally and exposes `F0`, `N`, `durations` directly. We do
   NOT need to recompute `en` at the KokoroModel level.

8. **Style split**:
   - `s_predictor = ref_s[:, 128:]` — second half (128..255) feeds Predictor +
     decode/encode AdaIN/AdaLayerNorm
   - `s_decoder   = ref_s[:, :128]` — first half (0..127) feeds DecoderEncodeStage
     and Generator AdaINResBlock1
   This is the convention from `model.py:103` and `model.py:117`.

9. **`ref_s` slice index**: `voice_emb[T - 1]` where T includes the 2 padding
   tokens. So for raw phoneme count P, T = P + 2, and slice index is P + 1.
   This is how kokoro/pipeline.py does it (`pack[len(input_ids)-1]`).

10. **PRNG for SineGen**: Generator's `m_source` adds Gaussian noise to the
    sinusoidal source. This is non-deterministic in PyTorch unless seeded.
    `KokoroModel::synthesize_with_phonemes` accepts a pre-computed `har_source`
    to bypass this for golden tests. The production `synthesize()` API uses
    `Generator::forward(x, F0, style, seed)` which calls `std::mt19937(seed)`
    internally.

## BERT layer details (ALBERT-tiny, 1 shared block applied 12 times)

```
embeddings: word(178,128) + position(512,128) + token_type(2,128)
LayerNorm  on (T, 128) with gamma/beta (128,), eps=1e-12

embedding_hidden_mapping_in: Linear(128 → 768) with bias

# Apply same albert_layer 12 times:
for _ in range(12):
    # Pre-attention LayerNorm is on the residual side per HuggingFace ALBERT
    # source — but their implementation order is:
    #   attn_out = attention(x)        # internal LayerNorm here
    #   ffn_out  = full_layer_layer_norm(attn_out + ffn(attn_out))

    # Attention sub-block:
    Q = Linear(768→768)(x);  K = Linear(768→768)(x);  V = Linear(768→768)(x)
    # Reshape to (B=1, H=12, T, D=64), scaled dot-product:
    attn = softmax(Q · K^T / sqrt(64)) · V
    # Reshape back to (T, 768):
    proj = Linear(768→768)(attn)        # attention.dense
    attn_out = LayerNorm(proj + x, eps=1e-12, gamma/beta from attention.LayerNorm)

    # FFN sub-block:
    h = Linear(768→2048)(attn_out)      # ffn
    h = gelu_new(h)                     # tanh-approx GELU (Hendrycks-Gimpel)
    h = Linear(2048→768)(h)             # ffn_output
    x = LayerNorm(attn_out + h, eps=1e-12, gamma/beta from full_layer_layer_norm)

# Pooler — UNUSED by KModel (it consumes last_hidden_state directly):
pooled = tanh(Linear(768→768)(x[:, 0, :]))
```

### `gelu_new` formula

ALBERT's `hidden_act = "gelu_new"` is the tanh approximation:

```
gelu_new(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

NOT the exact erf-based GELU. This matters at 2-3 decimal places of accuracy.

### LayerNorm placement

HuggingFace ALBERT (which CustomAlbert subclasses) puts LayerNorm POST-residual
inside `AlbertAttention.forward` and `AlbertLayer.forward`:

```
# AlbertAttention (transformers/models/albert/modeling_albert.py)
def forward(self, hidden_states):
    # ... compute attention, then:
    attention_output = self.LayerNorm(self.dense(context_layer) + hidden_states)
    return attention_output

# AlbertLayer
def forward(self, hidden_states):
    attention_output = self.attention(hidden_states)
    ffn_output = self.ffn_output(self.activation(self.ffn(attention_output)))
    hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
    return hidden_states
```

So both LayerNorms are **post-residual** ("PostLN"), matching the original
BERT/Transformer architecture. The cactus port mirrors this exactly.

### Number of transformer layers

`config['plbert']['num_hidden_layers'] = 12`, BUT ALBERT shares parameters
across layers (`num_hidden_groups = 1`, `inner_group_num = 1`). So the
checkpoint contains weights for **one** layer, applied **12** times in series.

Our weights directory has exactly one set:
```
bert_layer_0_attn_q_weight, bert_layer_0_attn_k_weight, ...
bert_layer_0_attn_dense_weight, bert_layer_0_attn_norm_weight, ...
bert_layer_0_ffn_weight, bert_layer_0_ffn_output_weight,
bert_layer_0_full_norm_weight, ...
```

The cactus `Bert` class loads these once and runs the layer 12 times in a loop.
