#pragma once

// Wooyoung Kim: independent scalar FP32 baseline.
// Shared domain checks live in int4/domain.hpp; the formal prepared
// operator and rotation are in int4/linear.hpp and int4/hadamard.hpp.

#include "int4/domain.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace bench {

using int4::Matrix;

using int4::max_domain_m;
using int4::max_domain_n;
using int4::max_domain_k;
using int4::is_power_of_two;
using int4::check_weight_domain;
using int4::check_input_domain;
using int4::check_finite;

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

} // namespace bench
