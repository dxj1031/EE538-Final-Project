#pragma once

// Wooyoung Kim: operator paths driven by the benchmark.
//
//  * Fp32Linear      - scalar FP32 baseline (independent of the INT4 path).
//  * W4A4Linear      - direct or Hadamard-rotated W4A4 with a prepared/retained weight.
//  * HadamardRotation
//
// STAND-IN NOTICE: the rotation and prepared operator are Member 2's deliverables
// (QuantizedLinear::prepare/run). They were not in the repository when the benchmark
// was written, so W4A4Linear/HadamardRotation implement the lifecycle from HANDOFF.md
// Section 4 so that all three modes can be measured. When Member 2's operator is
// merged, replace these two classes in bench/int4_bench.cpp; the driver uses only
// prepare(W), quantize_input(X), packed_weights(), run(X, kernel) and the storage
// accessors below.

#include "int4/core.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench {

using int4::Matrix;

// ---------------------------------------------------------------------------------
// Operator domain from the proposal: 1 <= M <= 32, 1 <= N <= 1024, K = 2^r, r <= 10.
inline constexpr std::size_t max_domain_m = 32;
inline constexpr std::size_t max_domain_n = 1024;
inline constexpr std::size_t max_domain_k = 1024;

inline bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

inline void check_weight_domain(std::size_t k, std::size_t n) {
    if (!is_power_of_two(k) || k > max_domain_k) {
        throw std::invalid_argument("K must be a power of two in [1, 1024]");
    }
    if (n == 0 || n > max_domain_n) throw std::invalid_argument("N must be in [1, 1024]");
}

inline void check_input_domain(std::size_t m, std::size_t k, std::size_t prepared_k) {
    if (m == 0 || m > max_domain_m) throw std::invalid_argument("M must be in [1, 32]");
    if (k != prepared_k) throw std::invalid_argument("X columns do not match prepared K");
}

inline void check_finite(const Matrix<float>& matrix, const char* what) {
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        if (!std::isfinite(matrix.data()[i])) throw std::invalid_argument(std::string(what) + " must be finite");
    }
}

// ---------------------------------------------------------------------------------
// Scalar FP32 baseline: Y[m,n] = sum_k X[m,k] W[k,n], FP32 accumulation in k order.
inline Matrix<float> fp32_matmul(const Matrix<float>& x, const Matrix<float>& w) {
    if (x.cols() != w.rows()) throw std::invalid_argument("fp32 matmul: inner dimensions differ");
    const std::size_t k_count = x.cols();
    const std::size_t n_count = w.cols();
    Matrix<float> y(x.rows(), n_count);
    for (std::size_t m = 0; m < x.rows(); ++m) {
        float* out = y.row_data(m);
        const float* x_row = x.row_data(m);
        for (std::size_t k = 0; k < k_count; ++k) {
            const float a = x_row[k];
            const float* w_row = w.data() + k * n_count;
            for (std::size_t n = 0; n < n_count; ++n) out[n] += a * w_row[n];
        }
    }
    return y;
}

class Fp32Linear {
public:
    void prepare(const Matrix<float>& w) {
        check_weight_domain(w.rows(), w.cols());
        check_finite(w, "W");
        Matrix<float> copy = w; // Retained FP32 copy, built before the noexcept swap-in.
        w_ = std::move(copy);
    }
    Matrix<float> run(const Matrix<float>& x) const {
        const Matrix<float>& w = weights();
        check_input_domain(x.rows(), x.cols(), w.rows());
        check_finite(x, "X");
        return fp32_matmul(x, w);
    }
    const Matrix<float>& weights() const {
        if (!w_) throw std::logic_error("operator used before prepare()");
        return *w_;
    }
    std::size_t resident_bytes() const { return weights().storage_bytes(); }

private:
    std::optional<Matrix<float>> w_;
};

// ---------------------------------------------------------------------------------
// Q = D H_K with seeded signs D_ii in {-1, +1} and the normalized (orthonormal)
// Walsh-Hadamard matrix H_K. Q is applied implicitly with the fast transform; a dense
// Q is never allocated. Because H is symmetric and H H = I, D D = I:
//   X' = X Q   = (X D) H     -> scale columns by d, then FWHT every row;
//   W' = Q^T W = H (D W)     -> scale rows by d, then FWHT along the K axis;
//   X' W' = X D H H D W = X W.
class HadamardRotation {
public:
    HadamardRotation(std::size_t k, std::uint64_t seed) : k_(k), seed_(seed), signs_(k) {
        if (!is_power_of_two(k)) throw std::invalid_argument("Hadamard size K must be a power of two");
        std::mt19937_64 engine(seed); // Fully specified engine: portable signs.
        for (std::int8_t& sign : signs_) sign = (engine() >> 63) != 0 ? std::int8_t{-1} : std::int8_t{1};
        scale_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(k)));
    }

    std::size_t k() const noexcept { return k_; }
    std::uint64_t seed() const noexcept { return seed_; }
    const std::vector<std::int8_t>& signs() const noexcept { return signs_; }
    std::size_t storage_bytes() const noexcept { return signs_.size() * sizeof(std::int8_t); }

    // In place: X <- X Q.
    void apply_right(Matrix<float>& x) const {
        if (x.cols() != k_) throw std::invalid_argument("rotation: X columns must equal K");
        for (std::size_t row = 0; row < x.rows(); ++row) {
            float* v = x.row_data(row);
            for (std::size_t i = 0; i < k_; ++i) {
                if (signs_[i] < 0) v[i] = -v[i];
            }
            for (std::size_t h = 1; h < k_; h <<= 1) {
                for (std::size_t i = 0; i < k_; i += 2 * h) {
                    for (std::size_t j = i; j < i + h; ++j) {
                        const float a = v[j];
                        const float b = v[j + h];
                        v[j] = a + b;
                        v[j + h] = a - b;
                    }
                }
            }
            for (std::size_t i = 0; i < k_; ++i) v[i] *= scale_;
        }
    }

    // In place: W <- Q^T W. The butterfly pairs whole rows, so inner loops are contiguous.
    void apply_transpose_left(Matrix<float>& w) const {
        if (w.rows() != k_) throw std::invalid_argument("rotation: W rows must equal K");
        const std::size_t n_count = w.cols();
        float* data = w.data();
        for (std::size_t i = 0; i < k_; ++i) {
            if (signs_[i] < 0) {
                float* row = data + i * n_count;
                for (std::size_t n = 0; n < n_count; ++n) row[n] = -row[n];
            }
        }
        for (std::size_t h = 1; h < k_; h <<= 1) {
            for (std::size_t i = 0; i < k_; i += 2 * h) {
                for (std::size_t j = i; j < i + h; ++j) {
                    float* top = data + j * n_count;
                    float* bottom = data + (j + h) * n_count;
                    for (std::size_t n = 0; n < n_count; ++n) {
                        const float a = top[n];
                        const float b = bottom[n];
                        top[n] = a + b;
                        bottom[n] = a - b;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < w.size(); ++i) data[i] *= scale_;
    }

private:
    std::size_t k_;
    std::uint64_t seed_;
    std::vector<std::int8_t> signs_;
    float scale_ = 1;
};

// ---------------------------------------------------------------------------------
enum class QuantMode { Direct, Rotated };

inline const char* mode_name(QuantMode mode) { return mode == QuantMode::Direct ? "direct" : "rotated"; }

template <typename Scale = float>
class W4A4Linear {
public:
    W4A4Linear(QuantMode mode, std::uint64_t rotation_seed) : mode_(mode), rotation_seed_(rotation_seed) {}

    QuantMode mode() const noexcept { return mode_; }

    // Validates, optionally rotates, and quantizes W with per-column scales. New state is
    // built completely before it replaces the old state (strong exception guarantee).
    void prepare(const Matrix<float>& w) {
        check_weight_domain(w.rows(), w.cols());
        check_finite(w, "W");
        std::optional<HadamardRotation> rotation;
        std::optional<int4::PackedInt4Matrix<Scale>> packed;
        if (mode_ == QuantMode::Rotated) {
            rotation.emplace(w.rows(), rotation_seed_);
            Matrix<float> rotated = w;
            rotation->apply_transpose_left(rotated);
            packed.emplace(int4::quantize_int4<Scale>(rotated, int4::ScaleAxis::Column));
        } else {
            packed.emplace(int4::quantize_int4<Scale>(w, int4::ScaleAxis::Column));
        }
        rotation_ = std::move(rotation);
        qw_ = std::move(packed);
    }

    // Online input path up to the kernel: rotation (same Q as prepare) and per-row
    // quantization/packing.
    int4::PackedInt4Matrix<Scale> quantize_input(const Matrix<float>& x) const {
        const auto& w = packed_weights();
        check_input_domain(x.rows(), x.cols(), w.rows());
        if (mode_ == QuantMode::Rotated) {
            Matrix<float> rotated = x;
            rotation_->apply_right(rotated);
            return int4::quantize_int4<Scale>(rotated, int4::ScaleAxis::Row);
        }
        return int4::quantize_int4<Scale>(x, int4::ScaleAxis::Row);
    }

    // Full input call: rotation, quantization/packing, integer kernel, shared rescaling.
    Matrix<float> run(const Matrix<float>& x, const int4::MatmulKernel<Scale>& kernel) const {
        const auto qx = quantize_input(x);
        const auto sums = kernel.run(qx, packed_weights());
        return int4::rescale_output(sums, qx.scales(), packed_weights().scales());
    }

    const int4::PackedInt4Matrix<Scale>& packed_weights() const {
        if (!qw_) throw std::logic_error("operator used before prepare()");
        return *qw_;
    }
    const HadamardRotation* rotation() const noexcept { return rotation_ ? &*rotation_ : nullptr; }

    std::size_t weight_bytes() const { return packed_weights().storage_bytes(); }
    std::size_t signs_bytes() const noexcept { return rotation_ ? rotation_->storage_bytes() : 0; }

private:
    QuantMode mode_;
    std::uint64_t rotation_seed_;
    std::optional<HadamardRotation> rotation_;
    std::optional<int4::PackedInt4Matrix<Scale>> qw_;
};

} // namespace bench
