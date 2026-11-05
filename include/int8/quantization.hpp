#pragma once

#include "int8/matrix.hpp"

#include <cstdint>
#include <vector>

namespace int8 {

enum class ScaleMode { PerTensor, PerRow };

struct QuantizedMatrix {
    Matrix<std::int8_t> values;
    std::vector<double> scales;
    ScaleMode mode;
};

// Symmetric [-127,127], zero point 0, max-abs / 127, ties away from zero.
// Double scales preserve even subnormal FP32 inputs. All-zero groups use scale 1.
QuantizedMatrix quantize(const Matrix<float>& input, ScaleMode mode);

// Reject malformed metadata and codes outside the symmetric representation.
void validate_quantized(const QuantizedMatrix& input);

// A debugging/quality helper. The scalar kernel does NOT dequantize operands.
Matrix<float> dequantize(const QuantizedMatrix& input);

} // namespace int8
