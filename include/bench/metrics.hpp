#pragma once

// Wooyoung Kim: FP64 reference product and error metrics (proposal Section 6.5, Eq. 3).
// The NumPy script scripts/reference_check.py recomputes the same metrics independently.

#include "int4/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace bench {

using int4::Matrix;

// Y64 = FP64(X) FP64(W), accumulated in double with a k-outer loop.
inline Matrix<double> reference_fp64(const Matrix<float>& x, const Matrix<float>& w) {
    if (x.cols() != w.rows()) throw std::invalid_argument("reference: inner dimensions differ");
    Matrix<double> y(x.rows(), w.cols());
    const std::size_t k_count = x.cols();
    const std::size_t n_count = w.cols();
    for (std::size_t m = 0; m < x.rows(); ++m) {
        double* out = y.row_data(m);
        const float* x_row = x.row_data(m);
        for (std::size_t k = 0; k < k_count; ++k) {
            const double a = x_row[k];
            const float* w_row = w.data() + k * n_count;
            for (std::size_t n = 0; n < n_count; ++n) out[n] += a * static_cast<double>(w_row[n]);
        }
    }
    return y;
}

struct ErrorMetrics {
    double max_abs = 0;
    double rel_frobenius = 0; // ||Y - Y64||_F / max(||Y64||_F, 1e-12)
};

inline ErrorMetrics compute_errors(const Matrix<float>& y, const Matrix<double>& reference) {
    if (y.rows() != reference.rows() || y.cols() != reference.cols()) {
        throw std::invalid_argument("error metrics: shapes differ");
    }
    ErrorMetrics metrics;
    double diff_sq = 0;
    double ref_sq = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double ref = reference.data()[i];
        const double diff = static_cast<double>(y.data()[i]) - ref;
        metrics.max_abs = std::max(metrics.max_abs, std::abs(diff));
        diff_sq += diff * diff;
        ref_sq += ref * ref;
    }
    metrics.rel_frobenius = std::sqrt(diff_sq) / std::max(std::sqrt(ref_sq), 1e-12);
    return metrics;
}

// Consumes an output after timing so the timed computation is observable.
inline double checksum(const Matrix<float>& y) {
    double sum = 0;
    for (std::size_t i = 0; i < y.size(); ++i) sum += static_cast<double>(y.data()[i]);
    return sum;
}

template <typename T>
bool identical(const Matrix<T>& a, const Matrix<T>& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::equal(a.data(), a.data() + a.size(), b.data());
}

} // namespace bench
