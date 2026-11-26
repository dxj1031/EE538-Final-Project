#pragma once

// Qinyan Zeng: implicit Q = D H_K, with normalized Sylvester H_K.
// The sign stream matches Wooyoung Kim's original benchmark fixture.
#include "int4/domain.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace int4 {

class HadamardRotation {
public:
    HadamardRotation(std::size_t k, std::uint64_t seed)
        : k_(checked_k(k)), seed_(seed), signs_(k_),
          normalization_(1.0 / std::sqrt(static_cast<double>(k_))) {
        std::mt19937_64 engine(seed);
        for (auto& sign : signs_) {
            sign = (engine() >> 63) != 0 ? std::int8_t{-1} : std::int8_t{1};
        }
    }

    std::size_t k() const noexcept { return k_; }
    std::uint64_t seed() const noexcept { return seed_; }
    const std::vector<std::int8_t>& signs() const noexcept { return signs_; }
    std::size_t storage_bytes() const noexcept { return signs_.size() * sizeof(std::int8_t); }
    std::size_t scratch_bytes() const noexcept { return k_ * sizeof(double); }

    // XQ = (XD)H: signs on the K columns, then one FWHT per row.
    void apply_right(Matrix<float>& x) const {
        if (x.cols() != k_ || x.rows() == 0) {
            throw std::invalid_argument("rotation: expected nonempty X with K columns");
        }
        check_finite(x, "rotation X");
        std::vector<double> work(k_);
        for (std::size_t m = 0; m < x.rows(); ++m) {
            for (std::size_t i = 0; i < k_; ++i) {
                work[i] = static_cast<double>(x(m, i)) * signs_[i];
            }
            transform(work);
            for (std::size_t i = 0; i < k_; ++i) {
                x(m, i) = checked_fp32(work[i] * normalization_);
            }
        }
    }

    // Q^T W = H(DW): signs on the K rows, then one FWHT per column.
    void apply_transpose_left(Matrix<float>& w) const {
        if (w.rows() != k_ || w.cols() == 0) {
            throw std::invalid_argument("rotation: expected nonempty W with K rows");
        }
        check_finite(w, "rotation W");
        std::vector<double> work(k_);
        for (std::size_t n = 0; n < w.cols(); ++n) {
            for (std::size_t i = 0; i < k_; ++i) {
                work[i] = static_cast<double>(w(i, n)) * signs_[i];
            }
            transform(work);
            for (std::size_t i = 0; i < k_; ++i) {
                w(i, n) = checked_fp32(work[i] * normalization_);
            }
        }
    }

private:
    static std::size_t checked_k(std::size_t k) {
        check_weight_domain(k, 1); // Validate before allocating the signs buffer.
        return k;
    }

    static void transform(std::vector<double>& values) {
        // O(K log K), O(K) scratch, no dense Q. Double intermediates avoid
        // overflow of an unnormalized butterfly when its normalized result fits.
        for (std::size_t half = 1; half < values.size(); half *= 2) {
            for (std::size_t base = 0; base < values.size(); base += 2 * half) {
                for (std::size_t j = base; j < base + half; ++j) {
                    const double a = values[j];
                    const double b = values[j + half];
                    values[j] = a + b;
                    values[j + half] = a - b;
                }
            }
        }
    }

    std::size_t k_;
    std::uint64_t seed_;
    std::vector<std::int8_t> signs_;
    double normalization_;
};

} // namespace int4
