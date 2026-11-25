#pragma once

// Member 3: binary matrix dumps consumed by scripts/reference_check.py.
//
// Layout (little-endian): uint64 rows, uint64 cols, then rows*cols FP32 values in
// row-major order (16-byte header). NumPy: np.fromfile(path, "<f4", offset=16).
// A binary format keeps a full sweep's dumps (~25 MB) inside the proposal's 100 MB
// budget for stored data; the CLI's text input format is Member 2's interface.

#include "int4/core.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench {

using int4::Matrix;

inline bool host_is_little_endian() noexcept {
    const std::uint32_t probe = 1;
    unsigned char first = 0;
    std::memcpy(&first, &probe, 1);
    return first == 1;
}

inline void write_matrix_bin(const std::string& path, const Matrix<float>& matrix) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "FP32 must be IEEE-754");
    if (!host_is_little_endian()) throw std::runtime_error("binary dumps require a little-endian host");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open dump file for writing: " + path);
    const std::uint64_t header[2] = {matrix.rows(), matrix.cols()};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(matrix.data()),
              static_cast<std::streamsize>(matrix.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed while writing dump file: " + path);
}

inline Matrix<float> read_matrix_bin(const std::string& path) {
    if (!host_is_little_endian()) throw std::runtime_error("binary dumps require a little-endian host");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dump file: " + path);
    std::uint64_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[0] == 0 || header[1] == 0 ||
        header[0] > (std::uint64_t{1} << 24) || header[1] > (std::uint64_t{1} << 24) ||
        header[0] * header[1] > (std::uint64_t{1} << 28)) {
        throw std::runtime_error("invalid dump header: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(header[0] * header[1]));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("dump size does not match header: " + path);
    }
    return Matrix<float>(static_cast<std::size_t>(header[0]), static_cast<std::size_t>(header[1]),
                         std::move(values));
}

} // namespace bench
