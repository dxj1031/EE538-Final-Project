#pragma once

#include "int8/quantization.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace int8 {

// Conservative bound: every partial sum fits INT32 for any valid operand codes.
inline constexpr std::size_t max_safe_k =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / (127 * 127);

// X[M,K], W[N,K] -> S[M,N], S[m,n] = sum_k X[m,k] * W[n,k].
// No bias or scaling here; exposed for exact comparison with other kernels.
Matrix<std::int32_t> scalar_accumulate(const Matrix<std::int8_t>& x,
                                       const Matrix<std::int8_t>& weights);

// Input must be PerTensor; weights must be PerRow. Empty bias means no bias.
// Y[m,n] = float(double(S[m,n]) * sx * sw[n] + bias[n]).
// Returns a fresh output; retains no input pointers or mutable shared state.
Matrix<float> scalar_linear(const QuantizedMatrix& x,
                            const QuantizedMatrix& weights,
                            const std::vector<float>& bias = {});

} // namespace int8
