#include "int8/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace int8 {
namespace {
std::size_t group_count(ScaleMode mode, std::size_t rows) {
    switch (mode) {
    case ScaleMode::PerTensor: return 1;
    case ScaleMode::PerRow: return rows;
    }
    throw std::invalid_argument("unknown scale mode");
}
} // namespace

QuantizedMatrix quantize(const Matrix<float>& input, ScaleMode mode) {
    if (input.size() == 0) throw std::invalid_argument("cannot quantize an empty matrix");
    const std::size_t groups = group_count(mode, input.rows());
    std::vector<double> scales(groups, 0.0);
    for (std::size_t r = 0; r < input.rows(); ++r) {
        double& maximum = scales[mode == ScaleMode::PerTensor ? 0 : r];
        const float* row = input.row_data(r);
        for (std::size_t c = 0; c < input.cols(); ++c) {
            if (!std::isfinite(row[c])) {
                throw std::invalid_argument("quantization requires finite inputs");
            }
            maximum = std::max(maximum, std::abs(static_cast<double>(row[c])));
        }
    }
    for (double& scale : scales) scale = scale == 0.0 ? 1.0 : scale / 127.0;

    Matrix<std::int8_t> codes(input.rows(), input.cols());
    for (std::size_t r = 0; r < input.rows(); ++r) {
        const double scale = scales[mode == ScaleMode::PerTensor ? 0 : r];
        const float* source = input.row_data(r);
        std::int8_t* target = codes.row_data(r);
        for (std::size_t c = 0; c < input.cols(); ++c) {
            const double rounded = std::round(static_cast<double>(source[c]) / scale);
            target[c] = static_cast<std::int8_t>(std::clamp(rounded, -127.0, 127.0));
        }
    }
    return {std::move(codes), std::move(scales), mode};
}

void validate_quantized(const QuantizedMatrix& input) {
    if (input.values.size() == 0) throw std::invalid_argument("empty quantized matrix");
    if (input.scales.size() != group_count(input.mode, input.values.rows())) {
        throw std::invalid_argument("scale count does not match scale mode");
    }
    for (double scale : input.scales) {
        if (!std::isfinite(scale) || scale <= 0.0) {
            throw std::invalid_argument("scales must be finite and positive");
        }
    }
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        if (input.values.data()[i] < -127) {
            throw std::invalid_argument("-128 is outside the symmetric code range");
        }
    }
}

Matrix<float> dequantize(const QuantizedMatrix& input) {
    validate_quantized(input);
    Matrix<float> output(input.values.rows(), input.values.cols());
    for (std::size_t r = 0; r < output.rows(); ++r) {
        const double scale = input.scales[input.mode == ScaleMode::PerTensor ? 0 : r];
        for (std::size_t c = 0; c < output.cols(); ++c) {
            const double value = static_cast<double>(input.values(r, c)) * scale;
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
                throw std::overflow_error("dequantized value exceeds FP32 range");
            }
            output(r, c) = static_cast<float>(value);
        }
    }
    return output;
}

} // namespace int8
