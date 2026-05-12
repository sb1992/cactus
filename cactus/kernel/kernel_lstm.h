#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-compatible LSTM cell forward (single timestep, batch=1).
//
// Inputs (all row-major fp32):
//   x:    [I]      input
//   h_in: [H]      previous hidden state
//   c_in: [H]      previous cell state
//   W_ih: [4H, I]  input-hidden weights (gates ordered i, f, g, o per nn.LSTMCell)
//   W_hh: [4H, H]  hidden-hidden weights
//   b_ih: [4H]     input-hidden bias
//   b_hh: [4H]     hidden-hidden bias
//
// Outputs:
//   h_out: [H]
//   c_out: [H]
//
// Math (per nn.LSTMCell):
//   gates = W_ih @ x + b_ih + W_hh @ h_in + b_hh           // [4H]
//   i, f, g, o = split(gates, H)                            // each [H]
//   c_out = sigmoid(f) * c_in + sigmoid(i) * tanh(g)
//   h_out = sigmoid(o) * tanh(c_out)
//
// Scalar fp32 reference. NEON/SIMD optimization deferred — the LSTM is called
// at most a few hundred times per second in Kokoro's predictor stage, not in
// the hot path. If profiling reveals it matters, replace the inner accumulation
// with cactus_gemv_* kernels.
void cactus_lstm_cell(
    const float* x, const float* h_in, const float* c_in,
    const float* W_ih, const float* W_hh,
    const float* b_ih, const float* b_hh,
    float* h_out, float* c_out,
    int I, int H);

#ifdef __cplusplus
}
#endif
