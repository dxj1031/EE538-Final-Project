#pragma once

// Wooyoung Kim: controlled, reproducible workload generation (proposal Section 6.3-6.4).
//
// Per scale (K = N) and data seed, a Gaussian X0 [32,K] and W [K,N] are drawn once.
// A case takes the first M rows of X0 and multiplies the same ceil(0.01K) columns by
// the outlier factor a (a = 1 is the control). W is shared by every case of a scale.
//
// std::normal_distribution and std::uniform_int_distribution are implementation-
// defined, so data would differ between compilers. The generator below uses only the
// fully specified std::mt19937_64 and std::seed_seq plus explicit Box-Muller, so the
// same seeds give the same matrices on every platform (up to last-ulp libm
// differences in log/cos/sin, which are rare after rounding to FP32).

#include "int4/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

using int4::Matrix;

inline constexpr std::size_t max_batch_rows = 32;

class PortableRng {
public:
    // std::seed_seq::generate is fully specified, unlike most distributions.
    PortableRng(std::uint64_t seed, std::uint64_t k, std::uint64_t n, std::uint32_t stream) {
        std::seed_seq sequence{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
                               static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(n), stream};
        engine_.seed(sequence);
    }

    // Uniform on [0, 1) with 53 random bits.
    double uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }

    // Unbiased integer in [0, bound) by rejection sampling.
    std::uint64_t below(std::uint64_t bound) {
        if (bound == 0) throw std::invalid_argument("empty integer range");
        const std::uint64_t limit = engine_.max() - engine_.max() % bound;
        std::uint64_t value = engine_();
        while (value >= limit) value = engine_();
        return value % bound;
    }

    // Standard normal via Box-Muller; the second value of each pair is cached.
    double gaussian() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double u1 = 1.0 - uniform(); // (0, 1], so log(u1) is finite.
        const double u2 = uniform();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 6.283185307179586476925286766559 * u2;
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    std::mt19937_64 engine_;
    double spare_ = 0;
    bool has_spare_ = false;
};

inline std::size_t outlier_count(std::size_t k) {
    if (k == 0) throw std::invalid_argument("K must be positive");
    return (k + 99) / 100; // ceil(0.01 K) in exact integer arithmetic.
}

struct ScaleData {
    std::size_t k;
    std::size_t n;
    std::uint64_t seed;
    Matrix<float> x0;                      // [32, K]
    Matrix<float> w;                       // [K, N]
    std::vector<std::size_t> outlier_cols; // Sorted, distinct, size ceil(0.01K).
};

inline ScaleData generate_scale_data(std::size_t k, std::size_t n, std::uint64_t seed) {
    PortableRng values(seed, k, n, 0);
    Matrix<float> x0(max_batch_rows, k);
    Matrix<float> w(k, n);
    for (std::size_t i = 0; i < x0.size(); ++i) x0.data()[i] = static_cast<float>(values.gaussian());
    for (std::size_t i = 0; i < w.size(); ++i) w.data()[i] = static_cast<float>(values.gaussian());

    // Independent stream for column choice; partial Fisher-Yates without replacement.
    PortableRng picker(seed, k, n, 1);
    std::vector<std::size_t> columns(k);
    for (std::size_t i = 0; i < k; ++i) columns[i] = i;
    const std::size_t count = outlier_count(k);
    for (std::size_t i = 0; i < count; ++i) {
        const auto j = i + static_cast<std::size_t>(picker.below(k - i));
        std::swap(columns[i], columns[j]);
    }
    columns.resize(count);
    std::sort(columns.begin(), columns.end());
    return {k, n, seed, std::move(x0), std::move(w), std::move(columns)};
}

// First M rows of X0 with the shared outlier columns multiplied by a (FP32 multiply).
inline Matrix<float> make_case_input(const ScaleData& data, std::size_t m, float a) {
    if (m == 0 || m > data.x0.rows()) throw std::invalid_argument("case M must be in [1, 32]");
    if (!std::isfinite(a) || a <= 0) throw std::invalid_argument("outlier factor must be finite and positive");
    Matrix<float> x(m, data.k);
    std::copy(data.x0.data(), data.x0.data() + x.size(), x.data());
    for (std::size_t row = 0; row < m; ++row) {
        float* values = x.row_data(row);
        for (std::size_t col : data.outlier_cols) values[col] *= a;
    }
    return x;
}

struct CaseSpec {
    std::size_t k;
    std::size_t n;
    std::size_t m;
    float a;
    std::string sweep; // "batch", "outlier", or "both" for the shared M=8, a=10 case.
};

// Controlled sweeps (not the Cartesian product): at each scale, M in {1,8,32} with
// a = 10 and a in {1,10,100} with M = 8. The shared (M=8, a=10) case is listed once,
// giving 5 distinct cases per scale and 15 per seed for three scales.
inline std::vector<CaseSpec> protocol_cases(const std::vector<std::size_t>& scales) {
    std::vector<CaseSpec> cases;
    for (std::size_t scale : scales) {
        cases.push_back({scale, scale, 1, 10.0F, "batch"});
        cases.push_back({scale, scale, 8, 10.0F, "both"});
        cases.push_back({scale, scale, 32, 10.0F, "batch"});
        cases.push_back({scale, scale, 8, 1.0F, "outlier"});
        cases.push_back({scale, scale, 8, 100.0F, "outlier"});
    }
    return cases;
}

} // namespace bench
