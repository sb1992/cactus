#include "kernel_conv1d_transpose.h"

#include <cstring>

void cactus_conv1d_transpose(
    const float* x, const float* W, const float* b, float* y,
    int c_in, int c_out, int t_in, int k, int stride, int padding)
{
    const int t_out = (t_in - 1) * stride - 2 * padding + k;
    std::memset(y, 0, static_cast<size_t>(c_out) * t_out * sizeof(float));

    // Scatter: for each (ic, t_in, kk) input position+kernel-tap, add into output
    // at t_pos = t_in * stride + kk - padding (if in range).
    for (int ic = 0; ic < c_in; ++ic) {
        for (int ti = 0; ti < t_in; ++ti) {
            const float xv = x[ic * t_in + ti];
            for (int oc = 0; oc < c_out; ++oc) {
                const float* w_ic_oc = W + (static_cast<size_t>(ic) * c_out + oc) * k;
                float* y_oc = y + static_cast<size_t>(oc) * t_out;
                for (int kk = 0; kk < k; ++kk) {
                    const int t_pos = ti * stride + kk - padding;
                    if (t_pos >= 0 && t_pos < t_out) {
                        y_oc[t_pos] += xv * w_ic_oc[kk];
                    }
                }
            }
        }
    }

    if (b) {
        for (int oc = 0; oc < c_out; ++oc) {
            float* y_oc = y + static_cast<size_t>(oc) * t_out;
            const float bv = b[oc];
            for (int t = 0; t < t_out; ++t) y_oc[t] += bv;
        }
    }
}
