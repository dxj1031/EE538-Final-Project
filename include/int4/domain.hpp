#pragma once

#include "int4/core.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace int4 {

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
        if (!std::isfinite(matrix.data()[i])) {
            throw std::invalid_argument(std::string(what) + " must be finite");
        }
    }
}

inline float checked_fp32(double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
        throw std::overflow_error("result exceeds finite FP32 range");
    }
    return static_cast<float>(value);
}

} // namespace int4
