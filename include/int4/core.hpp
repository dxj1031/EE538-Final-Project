#pragma once

// Reuse the existing owning matrix rather than duplicate its storage implementation.
#include "int8/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace int4 {

template <typename T>
using Matrix = int8::Matrix<T>;

enum class ScaleAxis { Row, Column };

inline constexpr std::size_t max_safe_k =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 49;

inline void check_inner_dimension(std::size_t k) {
    if (k == 0) throw std::invalid_argument("K must be positive");
    if (k > max_safe_k) throw std::overflow_error("49*K exceeds the INT32 bound");
}

// Logical row-major matrix. Each row starts at a byte boundary.
// Low nibble = even column; high nibble = odd column; unused high nibble = 0.
// Negative values use four-bit two's complement; nibble 8 (-8) is forbidden.
template <typename Scale = float>
class PackedInt4Matrix {
    static_assert(std::is_same_v<Scale, float> || std::is_same_v<Scale, double>,
                  "INT4 scale type must be float or double");
public:
    using scale_type = Scale;

    PackedInt4Matrix(std::size_t rows, std::size_t cols, ScaleAxis axis,
                     std::vector<std::uint8_t> bytes, std::vector<Scale> scales)
        : rows_(rows), cols_(cols), axis_(axis),
          bytes_(std::move(bytes)), scales_(std::move(scales)) {
        validate();
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t row_bytes() const noexcept { return cols_ / 2 + cols_ % 2; }
    ScaleAxis scale_axis() const noexcept { return axis_; }
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
    const std::vector<Scale>& scales() const noexcept { return scales_; }
    // Logical buffers only, excluding capacity, allocator and object overhead.
    std::size_t storage_bytes() const noexcept {
        return bytes_.size() + scales_.size() * sizeof(Scale);
    }

    std::int8_t code(std::size_t row, std::size_t col) const {
        if (row >= rows_ || col >= cols_) throw std::out_of_range("INT4 index out of range");
        const unsigned byte = bytes_.at(row * row_bytes() + col / 2);
        const unsigned nibble = (byte >> (4 * (col % 2))) & 15U;
        const int value = nibble >= 8 ? static_cast<int>(nibble) - 16
                                      : static_cast<int>(nibble);
        return static_cast<std::int8_t>(value);
    }

    void validate() const {
        if (rows_ == 0 || cols_ == 0) throw std::invalid_argument("INT4 shape must be positive");
        if (rows_ > std::numeric_limits<std::size_t>::max() / cols_ ||
            rows_ > std::numeric_limits<std::size_t>::max() / row_bytes()) {
            throw std::length_error("INT4 shape overflows size_t");
        }
        if (bytes_.size() != rows_ * row_bytes()) {
            throw std::invalid_argument("INT4 packed byte count does not match shape");
        }
        if (axis_ != ScaleAxis::Row && axis_ != ScaleAxis::Column) {
            throw std::invalid_argument("unknown scale axis");
        }
        const std::size_t count = axis_ == ScaleAxis::Row ? rows_ : cols_;
        if (scales_.size() != count) throw std::invalid_argument("INT4 scale count mismatch");
        for (Scale scale : scales_) {
            if (!std::isfinite(scale) || scale <= 0) {
                throw std::invalid_argument("INT4 scales must be finite and positive");
            }
        }
        for (std::uint8_t byte : bytes_) {
            if ((byte & 15U) == 8 || (byte >> 4) == 8) {
                throw std::invalid_argument("nibble 8 (-8) is outside [-7,7]");
            }
        }
        if (cols_ % 2 != 0) {
            for (std::size_t row = 0; row < rows_; ++row) {
                if ((bytes_[(row + 1) * row_bytes() - 1] >> 4) != 0) {
                    throw std::invalid_argument("unused high nibble must be zero");
                }
            }
        }
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    ScaleAxis axis_;
    std::vector<std::uint8_t> bytes_;
    std::vector<Scale> scales_;
};

// Scale computation: cast max_abs/7 to Scale FIRST, then quantize using that
// stored scale. Ratio and std::round use double for both precision variants.
template <typename Scale = float>
PackedInt4Matrix<Scale> quantize_int4(const Matrix<float>& input, ScaleAxis axis) {
    if (input.size() == 0) throw std::invalid_argument("cannot quantize an empty matrix");
    if (axis != ScaleAxis::Row && axis != ScaleAxis::Column) {
        throw std::invalid_argument("unknown scale axis");
    }
    const std::size_t count = axis == ScaleAxis::Row ? input.rows() : input.cols();
    std::vector<double> maxima(count, 0.0);
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < input.cols(); ++col) {
            const double value = input(row, col);
            if (!std::isfinite(value)) throw std::invalid_argument("INT4 input must be finite");
            double& maximum = maxima[axis == ScaleAxis::Row ? row : col];
            maximum = std::max(maximum, std::abs(value));
        }
    }
    std::vector<Scale> scales(count);
    for (std::size_t i = 0; i < count; ++i) {
        scales[i] = static_cast<Scale>(maxima[i] == 0 ? 1.0 : maxima[i] / 7.0);
        if (scales[i] == 0) {
            throw std::underflow_error("INT4 scale underflows; use double scales or rescale data");
        }
    }
    const std::size_t stride = input.cols() / 2 + input.cols() % 2;
    std::vector<std::uint8_t> bytes(input.rows() * stride, 0);
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < input.cols(); ++col) {
            const double scale = scales[axis == ScaleAxis::Row ? row : col];
            const double rounded = std::round(static_cast<double>(input(row, col)) / scale);
            const int code = static_cast<int>(std::clamp(rounded, -7.0, 7.0));
            const unsigned nibble = static_cast<unsigned>(code < 0 ? code + 16 : code);
            const std::size_t index = row * stride + col / 2;
            bytes[index] = static_cast<std::uint8_t>(bytes[index] | (nibble << (4 * (col % 2))));
        }
    }
    return {input.rows(), input.cols(), axis, std::move(bytes), std::move(scales)};
}

// Both scalar and future blocked kernels implement this exact interface.
// Returns INTEGER sums only. Scaling is common to all kernels and happens once.
template <typename Scale = float>
class MatmulKernel {
public:
    virtual ~MatmulKernel() = default;
    virtual Matrix<std::int32_t> run(const PackedInt4Matrix<Scale>& x,
                                     const PackedInt4Matrix<Scale>& w) const = 0;
};

template <typename Scale = float>
class ScalarKernel final : public MatmulKernel<Scale> {
public:
    Matrix<std::int32_t> run(const PackedInt4Matrix<Scale>& x,
                             const PackedInt4Matrix<Scale>& w) const override {
        x.validate();
        w.validate();
        if (x.cols() != w.rows() || x.scale_axis() != ScaleAxis::Row ||
            w.scale_axis() != ScaleAxis::Column) {
            throw std::invalid_argument("expected X[M,K] row scales and W[K,N] column scales");
        }
        check_inner_dimension(x.cols());
        Matrix<std::int32_t> sums(x.rows(), w.cols());
        for (std::size_t m = 0; m < x.rows(); ++m) {
            for (std::size_t n = 0; n < w.cols(); ++n) {
                std::int32_t sum = 0;
                for (std::size_t k = 0; k < x.cols(); ++k) {
                    sum += static_cast<std::int32_t>(x.code(m, k))
                         * static_cast<std::int32_t>(w.code(k, n));
                }
                sums(m, n) = sum;
            }
        }
        return sums;
    }
};

template <typename Scale>
Matrix<float> rescale_output(const Matrix<std::int32_t>& sums,
                             const std::vector<Scale>& x_scales,
                             const std::vector<Scale>& w_scales) {
    static_assert(std::is_same_v<Scale, float> || std::is_same_v<Scale, double>);
    if (sums.size() == 0 || sums.rows() != x_scales.size() || sums.cols() != w_scales.size()) {
        throw std::invalid_argument("output shape does not match scales");
    }
    for (const auto* scales : {&x_scales, &w_scales}) {
        for (Scale scale : *scales) {
            if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("invalid output scale");
        }
    }
    Matrix<float> output(sums.rows(), sums.cols());
    for (std::size_t m = 0; m < sums.rows(); ++m) {
        for (std::size_t n = 0; n < sums.cols(); ++n) {
            const double value = static_cast<double>(sums(m, n))
                               * static_cast<double>(x_scales[m]) * static_cast<double>(w_scales[n]);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
                throw std::overflow_error("INT4 output exceeds FP32 range");
            }
            output(m, n) = static_cast<float>(value);
        }
    }
    return output;
}

} // namespace int4
