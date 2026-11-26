#pragma once

// Qinyan Zeng: strict, locale-independent text matrices for the file-to-file CLI.
#include "int4/domain.hpp"

#include <charconv>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace int4 {

inline std::uint64_t parse_unsigned(const std::string& text, const char* what) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string(what) + " expects an unsigned decimal integer");
    }
    return value;
}

enum class MatrixFileKind { Input, Weight };

inline Matrix<float> read_matrix_text(std::istream& stream, MatrixFileKind kind) {
    std::string token_rows;
    std::string token_cols;
    if (!(stream >> token_rows >> token_cols)) throw std::invalid_argument("missing matrix dimensions");
    const auto rows = parse_unsigned(token_rows, "matrix rows");
    const auto cols = parse_unsigned(token_cols, "matrix columns");
    if (kind == MatrixFileKind::Input) {
        if (rows == 0 || rows > max_domain_m || cols == 0 || cols > max_domain_k ||
            !is_power_of_two(static_cast<std::size_t>(cols))) {
            throw std::invalid_argument("X requires M in [1,32] and power-of-two K in [1,1024]");
        }
    } else if (kind == MatrixFileKind::Weight) {
        if (rows == 0 || rows > max_domain_k || cols == 0 || cols > max_domain_n) {
            throw std::invalid_argument("W dimensions must be in [1,1024]");
        }
        check_weight_domain(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    } else {
        throw std::invalid_argument("unknown matrix file kind");
    }
    // Domain checks precede allocation, including checks on untrusted huge headers.
    Matrix<float> result(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    for (std::size_t i = 0; i < result.size(); ++i) {
        std::string token;
        if (!(stream >> token)) throw std::invalid_argument("too few matrix values");
        // Parse directly to FP32 so max_digits10 text for FLT_MAX rounds back
        // correctly even when that decimal is slightly greater than exact FLT_MAX.
        const char* begin = token.data();
        const char* end = begin + token.size();
        if (begin != end && *begin == '+') {
            ++begin;
            if (begin == end || *begin == '-' || *begin == '+') {
                throw std::invalid_argument("invalid matrix value '" + token + "'");
            }
        }
        float value = 0;
        const auto parsed = std::from_chars(begin, end, value, std::chars_format::general);
        if (begin == end || parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(value)) {
            throw std::invalid_argument("invalid matrix value '" + token + "'");
        }
        result.data()[i] = value;
    }
    std::string extra;
    if (stream >> extra) throw std::invalid_argument("extra token after matrix values: '" + extra + "'");
    if (stream.bad()) throw std::runtime_error("matrix input read failed");
    return result;
}

inline void write_matrix_text(std::ostream& stream, const Matrix<float>& matrix) {
    check_finite(matrix, "output");
    stream.imbue(std::locale::classic());
    stream << matrix.rows() << ' ' << matrix.cols() << '\n'
           << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            if (col != 0) stream << ' ';
            stream << matrix(row, col);
        }
        stream << '\n';
    }
    if (!stream) throw std::runtime_error("matrix output write failed");
}

} // namespace int4
