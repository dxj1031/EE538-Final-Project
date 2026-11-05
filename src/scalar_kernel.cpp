#include "int8/scalar_kernel.hpp"

#include <cmath>
#include <stdexcept>

namespace int8 {
namespace {
void validate_codes(const Matrix<std::int8_t>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input.data()[i] < -127) {
            throw std::invalid_argument("-128 is outside the symmetric code range");
        }
    }
}
} // namespace

Matrix<std::int32_t> scalar_accumulate(const Matrix<std::int8_t>& x,
                                       const Matrix<std::int8_t>& weights) {
    if (x.size() == 0 || weights.size() == 0 || x.cols() != weights.cols()) {
        throw std::invalid_argument("expected nonempty X[M,K] and W[N,K]");
    }
    if (x.cols() > max_safe_k) {
        throw std::overflow_error("K exceeds the conservative INT32 accumulation limit");
    }
    validate_codes(x);
    validate_codes(weights);
    Matrix<std::int32_t> sums(x.rows(), weights.rows());
    for (std::size_t m = 0; m < x.rows(); ++m) {
        const std::int8_t* xr = x.row_data(m);
        std::int32_t* out = sums.row_data(m);
        for (std::size_t n = 0; n < weights.rows(); ++n) {
            const std::int8_t* wr = weights.row_data(n);
            std::int32_t sum = 0;
            for (std::size_t k = 0; k < x.cols(); ++k) {
                sum += static_cast<std::int32_t>(xr[k]) * static_cast<std::int32_t>(wr[k]);
            }
            out[n] = sum;
        }
    }
    return sums;
}

Matrix<float> scalar_linear(const QuantizedMatrix& x,
                            const QuantizedMatrix& weights,
                            const std::vector<float>& bias) {
    validate_quantized(x);
    validate_quantized(weights);
    if (x.mode != ScaleMode::PerTensor || weights.mode != ScaleMode::PerRow) {
        throw std::invalid_argument("scalar_linear requires PerTensor X and PerRow W");
    }
    if (!bias.empty() && bias.size() != weights.values.rows()) {
        throw std::invalid_argument("bias length must equal the number of output channels");
    }
    for (float value : bias) {
        if (!std::isfinite(value)) throw std::invalid_argument("bias must be finite");
    }

    const auto sums = scalar_accumulate(x.values, weights.values);
    Matrix<float> output(sums.rows(), sums.cols());
    for (std::size_t m = 0; m < sums.rows(); ++m) {
        for (std::size_t n = 0; n < sums.cols(); ++n) {
            const double value = static_cast<double>(sums(m, n)) * x.scales[0]
                               * weights.scales[n] + (bias.empty() ? 0.0 : bias[n]);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
                throw std::overflow_error("linear output exceeds FP32 range");
            }
            output(m, n) = static_cast<float>(value);
        }
    }
    return output;
}

} // namespace int8
