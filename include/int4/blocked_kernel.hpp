#pragma once

// Member 3: blocked W4A4 kernel.
//
// Implements the same MatmulKernel<Scale>::run contract as ScalarKernel: it takes the
// identical packed operands (row-scaled X[M,K], column-scaled W[K,N]) and returns the
// exact INT32 sums. Reconstruction stays in the shared int4::rescale_output.
//
// Difference from the scalar path: instead of decoding two nibbles for every
// multiply-accumulate, each T-by-T tile of the packed operands is decoded once into
// reusable INT8 scratch, and the inner loop multiplies decoded INT8 values.
// Within the proposal domain (M <= 32 = T) every X and W code is decoded exactly once
// per call; in general X is decoded once and W ceil(M/T) times.

#include "int4/core.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace int4 {

inline constexpr std::size_t default_tile_edge = 32;
inline constexpr std::size_t max_tile_edge = 1024; // Caps scratch at 2 MiB per call.

namespace detail {
// Signed (low, high) codes for every byte value, built at compile time. Nibble 8 maps
// to -8 here, but validate() rejects it before any decoding happens.
struct NibblePairTable {
    std::int8_t values[256][2];
    constexpr NibblePairTable() : values{} {
        for (int byte = 0; byte < 256; ++byte) {
            values[byte][0] = static_cast<std::int8_t>(((byte & 15) ^ 8) - 8);
            values[byte][1] = static_cast<std::int8_t>(((byte >> 4) ^ 8) - 8);
        }
    }
};
inline constexpr NibblePairTable nibble_pairs{};
} // namespace detail

template <typename Scale = float>
class BlockedKernel final : public MatmulKernel<Scale> {
public:
    explicit BlockedKernel(std::size_t tile_edge = default_tile_edge) : tile_(tile_edge) {
        if (tile_ == 0 || tile_ > max_tile_edge) {
            throw std::invalid_argument("blocked tile edge must be in [1, 1024]");
        }
    }

    std::size_t tile_edge() const noexcept { return tile_; }

    // Logical INT8 scratch allocated by one run() with m_rows activation rows:
    // one X tile (min(M,T) x T) plus one W tile (T x T). Excludes allocator overhead.
    std::size_t scratch_bytes(std::size_t m_rows) const noexcept {
        return (std::min(m_rows, tile_) + tile_) * tile_;
    }

    Matrix<std::int32_t> run(const PackedInt4Matrix<Scale>& x,
                             const PackedInt4Matrix<Scale>& w) const override {
        // Same acceptance rules as ScalarKernel: representation, axes, shape, INT32 bound.
        x.validate();
        w.validate();
        if (x.cols() != w.rows() || x.scale_axis() != ScaleAxis::Row ||
            w.scale_axis() != ScaleAxis::Column) {
            throw std::invalid_argument("expected X[M,K] row scales and W[K,N] column scales");
        }
        check_inner_dimension(x.cols());

        const std::size_t m_count = x.rows();
        const std::size_t k_count = x.cols();
        const std::size_t n_count = w.cols();
        Matrix<std::int32_t> sums(m_count, n_count); // Zero-initialized.

        // Scratch is allocated once per call and reused for every tile; partial tiles
        // use only the leading rt x ct entries, so real and padded extents never mix.
        std::vector<std::int8_t> x_tile(std::min(m_count, tile_) * tile_);
        std::vector<std::int8_t> w_tile(tile_ * tile_);
        std::int32_t* const out = sums.data();

        for (std::size_t m0 = 0; m0 < m_count; m0 += tile_) {
            const std::size_t mt = std::min(tile_, m_count - m0);
            for (std::size_t k0 = 0; k0 < k_count; k0 += tile_) {
                const std::size_t kt = std::min(tile_, k_count - k0);
                decode_tile(x, m0, mt, k0, kt, x_tile.data());
                for (std::size_t n0 = 0; n0 < n_count; n0 += tile_) {
                    const std::size_t nt = std::min(tile_, n_count - n0);
                    decode_tile(w, k0, kt, n0, nt, w_tile.data());
                    for (std::size_t i = 0; i < mt; ++i) {
                        std::int32_t* const out_row = out + (m0 + i) * n_count + n0;
                        const std::int8_t* const x_row = x_tile.data() + i * tile_;
                        for (std::size_t kk = 0; kk < kt; ++kk) {
                            const std::int32_t a = x_row[kk];
                            const std::int8_t* const w_row = w_tile.data() + kk * tile_;
                            // |partial sum| <= 49*K <= INT32_MAX, checked above.
                            for (std::size_t j = 0; j < nt; ++j) {
                                out_row[j] += a * static_cast<std::int32_t>(w_row[j]);
                            }
                        }
                    }
                }
            }
        }
        return sums;
    }

private:
    // Decodes logical rows [r0, r0+rt) x columns [c0, c0+ct) into dst (row stride T).
    // Tiles starting on an even column (every tile when T is even) decode one byte into
    // two codes through the lookup table; the final unused padding nibble of an odd-length
    // row is never read. Tiles starting on an odd column (odd T) use the per-nibble path,
    // where (nibble ^ 8) - 8 is the four-bit two's-complement sign extension.
    void decode_tile(const PackedInt4Matrix<Scale>& packed, std::size_t r0, std::size_t rt,
                     std::size_t c0, std::size_t ct, std::int8_t* dst) const {
        const std::uint8_t* const bytes = packed.bytes().data();
        const std::size_t stride = packed.row_bytes();
        for (std::size_t r = 0; r < rt; ++r) {
            const std::uint8_t* const src = bytes + (r0 + r) * stride;
            std::int8_t* const row = dst + r * tile_;
            if (c0 % 2 == 0) {
                const std::uint8_t* const pairs = src + c0 / 2;
                const std::size_t full = ct / 2;
                for (std::size_t p = 0; p < full; ++p) {
                    const std::int8_t* const codes = detail::nibble_pairs.values[pairs[p]];
                    row[2 * p] = codes[0];
                    row[2 * p + 1] = codes[1];
                }
                if (ct % 2 != 0) row[ct - 1] = detail::nibble_pairs.values[pairs[full]][0];
            } else {
                for (std::size_t c = 0; c < ct; ++c) {
                    const std::size_t col = c0 + c;
                    const unsigned nibble = (static_cast<unsigned>(src[col / 2]) >> (4 * (col % 2))) & 15U;
                    row[c] = static_cast<std::int8_t>(static_cast<int>(nibble ^ 8U) - 8);
                }
            }
        }
    }

    std::size_t tile_;
};

} // namespace int4
