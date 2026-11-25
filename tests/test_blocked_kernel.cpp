// Member 3: tests for int4::BlockedKernel.
//
// The blocked kernel must return INT32 sums identical to ScalarKernel on the same packed
// operands (exact agreement, proposal Section 5.3) and to an independent INT64 reference
// decoded straight from the bytes, across partial tiles, odd/padded rows, K = 1, N = 33,
// all-zero groups, extreme codes, several tile edges, and both scale types.

#include "int4/blocked_kernel.hpp"
#include "int4/core.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void rejects(Function fn, const std::string& message) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception was not thrown: " + message);
}

std::uint8_t nibble(int code) { return static_cast<std::uint8_t>(static_cast<unsigned>(code) & 15U); }

// Packs explicit codes with the documented layout (independent of quantize_int4).
template <typename Scale>
int4::PackedInt4Matrix<Scale> pack(std::size_t rows, std::size_t cols, int4::ScaleAxis axis,
                                   const std::vector<int>& codes) {
    const std::size_t row_bytes = cols / 2 + cols % 2;
    std::vector<std::uint8_t> bytes(rows * row_bytes, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t index = r * row_bytes + c / 2;
            bytes[index] = static_cast<std::uint8_t>(bytes[index] | (nibble(codes[r * cols + c]) << (4 * (c % 2))));
        }
    }
    const std::size_t count = axis == int4::ScaleAxis::Row ? rows : cols;
    return {rows, cols, axis, std::move(bytes), std::vector<Scale>(count, Scale{1})};
}

// Independent reference: decode with a branch (not the kernel's xor trick), INT64 sums.
template <typename Scale>
int4::Matrix<std::int64_t> reference_sums(const int4::PackedInt4Matrix<Scale>& x,
                                          const int4::PackedInt4Matrix<Scale>& w) {
    const auto decode = [](const int4::PackedInt4Matrix<Scale>& p, std::size_t r, std::size_t c) {
        const unsigned v = (p.bytes()[r * p.row_bytes() + c / 2] >> (4 * (c % 2))) & 15U;
        return v <= 7 ? static_cast<std::int64_t>(v) : static_cast<std::int64_t>(v) - 16;
    };
    int4::Matrix<std::int64_t> out(x.rows(), w.cols());
    for (std::size_t n = 0; n < w.cols(); ++n) {
        for (std::size_t k = 0; k < x.cols(); ++k) {
            const std::int64_t b = decode(w, k, n);
            for (std::size_t m = 0; m < x.rows(); ++m) out(m, n) += decode(x, m, k) * b;
        }
    }
    return out;
}

template <typename Scale>
void require_equal_to_reference(const int4::Matrix<std::int32_t>& actual,
                                const int4::Matrix<std::int64_t>& expected, const std::string& label) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(), label + ": shape");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(static_cast<std::int64_t>(actual.data()[i]) == expected.data()[i], label + ": INT64 reference");
    }
}

template <typename Scale>
void test_all_code_pairs() {
    // 225 one-term signed products (K = 1) through the virtual interface.
    const int4::BlockedKernel<Scale> blocked;
    const int4::MatmulKernel<Scale>& kernel = blocked;
    for (int a = -7; a <= 7; ++a) {
        for (int b = -7; b <= 7; ++b) {
            const auto x = pack<Scale>(1, 1, int4::ScaleAxis::Row, {a});
            const auto w = pack<Scale>(1, 1, int4::ScaleAxis::Column, {b});
            require(kernel.run(x, w)(0, 0) == a * b, "signed one-term product");
        }
    }
    // Every code pair in the low/high nibbles of one X byte and of W bytes:
    // X = [a, b], W = [[1, 0, b, a], [0, 1, a, b]] -> [a, b, 2ab, a^2 + b^2].
    for (int a = -7; a <= 7; ++a) {
        for (int b = -7; b <= 7; ++b) {
            const auto x = pack<Scale>(1, 2, int4::ScaleAxis::Row, {a, b});
            const auto w = pack<Scale>(2, 4, int4::ScaleAxis::Column, {1, 0, b, a, 0, 1, a, b});
            const auto sums = kernel.run(x, w);
            require(sums(0, 0) == a && sums(0, 1) == b && sums(0, 2) == 2 * a * b &&
                        sums(0, 3) == a * a + b * b,
                    "low/high nibble decoding");
        }
    }
}

template <typename Scale>
void test_handoff_fixture() {
    // HANDOFF.md Section 6: X = [[7,-3.5]], W = [[7,0],[0,14]].
    const auto qx = int4::quantize_int4<Scale>(int4::Matrix<float>(1, 2, {7, -3.5F}), int4::ScaleAxis::Row);
    const auto qw = int4::quantize_int4<Scale>(int4::Matrix<float>(2, 2, {7, 0, 0, 14}), int4::ScaleAxis::Column);
    require(qx.bytes() == std::vector<std::uint8_t>({0xC7}), "fixture qx bytes");
    require(qw.bytes() == std::vector<std::uint8_t>({0x07, 0x70}), "fixture qw bytes");
    const int4::BlockedKernel<Scale> kernel;
    const auto sums = kernel.run(qx, qw);
    require(sums(0, 0) == 49 && sums(0, 1) == -28, "fixture integer sums");
    const auto y = int4::rescale_output(sums, qx.scales(), qw.scales());
    require(y(0, 0) == 49.0F && y(0, 1) == -56.0F, "fixture reconstructed Y");
    // Proposal example: X = [1,-1], W = [7,0]^T -> Y = 7.
    const auto px = int4::quantize_int4<Scale>(int4::Matrix<float>(1, 2, {1, -1}), int4::ScaleAxis::Row);
    const auto pw = int4::quantize_int4<Scale>(int4::Matrix<float>(2, 1, {7, 0}), int4::ScaleAxis::Column);
    const auto py = int4::rescale_output(kernel.run(px, pw), px.scales(), pw.scales());
    require(std::abs(py(0, 0) - 7.0F) < 1e-6F, "proposal fixture");
}

template <typename Scale>
void test_random_against_scalar() {
    std::mt19937 rng(538);
    std::uniform_int_distribution<int> code(-7, 7);
    const int4::ScalarKernel<Scale> scalar;
    const std::vector<std::size_t> m_values{1, 2, 7, 8, 31, 32, 33, 65};
    const std::vector<std::size_t> k_values{1, 2, 3, 31, 32, 33, 64, 100};
    const std::vector<std::size_t> n_values{1, 2, 33, 63, 64, 65};
    const std::vector<std::size_t> tiles{1, 2, 3, 7, 16, 32, 64};
    std::size_t checked = 0;
    for (std::size_t tile : tiles) {
        const int4::BlockedKernel<Scale> blocked(tile);
        for (std::size_t m : m_values) {
            for (std::size_t k : k_values) {
                for (std::size_t n : n_values) {
                    if ((m * k * n) % 3 != 0 && tile != 32) continue; // Keep runtime modest.
                    std::vector<int> xc(m * k);
                    std::vector<int> wc(k * n);
                    for (int& v : xc) v = code(rng);
                    for (int& v : wc) v = code(rng);
                    // All-zero activation row and weight column (all-zero groups).
                    for (std::size_t c = 0; c < k; ++c) xc[(m - 1) * k + c] = 0;
                    for (std::size_t r = 0; r < k; ++r) wc[r * n + (n - 1)] = 0;
                    const auto x = pack<Scale>(m, k, int4::ScaleAxis::Row, xc);
                    const auto w = pack<Scale>(k, n, int4::ScaleAxis::Column, wc);
                    const auto x_before = x.bytes();
                    const auto w_before = w.bytes();
                    const auto expected = scalar.run(x, w);
                    const auto actual = blocked.run(x, w);
                    const std::string label = "M=" + std::to_string(m) + " K=" + std::to_string(k) +
                                              " N=" + std::to_string(n) + " T=" + std::to_string(tile);
                    require(actual.rows() == m && actual.cols() == n, label + ": shape");
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        require(actual.data()[i] == expected.data()[i], label + ": scalar mismatch");
                    }
                    for (std::size_t c = 0; c < n; ++c) require(actual(m - 1, c) == 0, label + ": zero row");
                    for (std::size_t r = 0; r < m; ++r) require(actual(r, n - 1) == 0, label + ": zero column");
                    if (tile == 32) require_equal_to_reference<Scale>(actual, reference_sums(x, w), label);
                    require(x.bytes() == x_before && w.bytes() == w_before, label + ": inputs unchanged");
                    ++checked;
                }
            }
        }
    }
    require(checked > 300, "random coverage");
}

template <typename Scale>
void test_quantized_pipeline() {
    // Quantized random FP32 data: identical sums and bitwise-identical reconstruction.
    std::mt19937 rng(2026);
    std::normal_distribution<float> normal(0, 1);
    const int4::ScalarKernel<Scale> scalar;
    const int4::BlockedKernel<Scale> blocked;
    for (std::size_t k : {1U, 2U, 64U, 256U}) {
        for (std::size_t n : {1U, 33U, 100U}) {
            int4::Matrix<float> x(8, k);
            int4::Matrix<float> w(k, n);
            for (std::size_t i = 0; i < x.size(); ++i) x.data()[i] = normal(rng);
            for (std::size_t i = 0; i < w.size(); ++i) w.data()[i] = normal(rng);
            x(0, 0) = 100; // Outlier.
            const auto qx = int4::quantize_int4<Scale>(x, int4::ScaleAxis::Row);
            const auto qw = int4::quantize_int4<Scale>(w, int4::ScaleAxis::Column);
            const auto s = scalar.run(qx, qw);
            const auto b = blocked.run(qx, qw);
            const auto ys = int4::rescale_output(s, qx.scales(), qw.scales());
            const auto yb = int4::rescale_output(b, qx.scales(), qw.scales());
            for (std::size_t i = 0; i < s.size(); ++i) require(s.data()[i] == b.data()[i], "pipeline sums");
            for (std::size_t i = 0; i < ys.size(); ++i) require(ys.data()[i] == yb.data()[i], "pipeline Y");
        }
    }
}

template <typename Scale>
void test_extremes() {
    // Largest per-term magnitude at the proposal's largest K: 49 * 1024 = 50176.
    const std::size_t k = 1024;
    const int4::BlockedKernel<Scale> kernel;
    const auto x = pack<Scale>(2, k, int4::ScaleAxis::Row, [&] {
        std::vector<int> v(2 * k, 7);
        for (std::size_t c = 0; c < k; ++c) v[k + c] = -7;
        return v;
    }());
    const auto w = pack<Scale>(k, 3, int4::ScaleAxis::Column, [&] {
        std::vector<int> v(k * 3);
        for (std::size_t r = 0; r < k; ++r) {
            v[r * 3] = 7;
            v[r * 3 + 1] = -7;
            v[r * 3 + 2] = r % 2 == 0 ? 7 : -7;
        }
        return v;
    }());
    const auto sums = kernel.run(x, w);
    require(sums(0, 0) == 50176 && sums(0, 1) == -50176 && sums(1, 0) == -50176 && sums(1, 1) == 50176,
            "extreme accumulation");
    require(sums(0, 2) == 0 && sums(1, 2) == 0, "alternating signs cancel");
}

template <typename Scale>
void test_errors_and_metadata() {
    rejects<std::invalid_argument>([] { (void)int4::BlockedKernel<Scale>(0); }, "tile 0");
    rejects<std::invalid_argument>([] { (void)int4::BlockedKernel<Scale>(int4::max_tile_edge + 1); }, "tile too big");
    const int4::BlockedKernel<Scale> kernel(16);
    require(kernel.tile_edge() == 16, "tile accessor");
    require(int4::BlockedKernel<Scale>().tile_edge() == 32, "default tile edge is 32");
    require(kernel.scratch_bytes(1) == (1 + 16) * 16, "scratch bytes for M=1");
    require(kernel.scratch_bytes(100) == (16 + 16) * 16, "scratch bytes cap at one tile");
    require(int4::BlockedKernel<Scale>().scratch_bytes(8) == (8 + 32) * 32, "scratch bytes M=8, T=32");

    const auto x = pack<Scale>(1, 2, int4::ScaleAxis::Row, {1, 2});
    const auto w_good = pack<Scale>(2, 1, int4::ScaleAxis::Column, {1, 1});
    const auto w_bad_k = pack<Scale>(3, 1, int4::ScaleAxis::Column, {1, 1, 1});
    const auto w_row_axis = pack<Scale>(2, 1, int4::ScaleAxis::Row, {1, 1});
    const auto x_col_axis = pack<Scale>(1, 2, int4::ScaleAxis::Column, {1, 2});
    require(kernel.run(x, w_good)(0, 0) == 3, "small product");
    rejects<std::invalid_argument>([&] { (void)kernel.run(x, w_bad_k); }, "K mismatch");
    rejects<std::invalid_argument>([&] { (void)kernel.run(x, w_row_axis); }, "W must use column scales");
    rejects<std::invalid_argument>([&] { (void)kernel.run(x_col_axis, w_good); }, "X must use row scales");
    rejects<std::invalid_argument>([&] { (void)int4::ScalarKernel<Scale>().run(x, w_bad_k); },
                                   "scalar applies the same K rule");
}

} // namespace

int main() {
    try {
        test_all_code_pairs<float>();
        test_all_code_pairs<double>();
        test_handoff_fixture<float>();
        test_handoff_fixture<double>();
        test_random_against_scalar<float>();
        test_random_against_scalar<double>();
        test_quantized_pipeline<float>();
        test_quantized_pipeline<double>();
        test_extremes<float>();
        test_extremes<double>();
        test_errors_and_metadata<float>();
        test_errors_and_metadata<double>();
        std::cout << "PASS: blocked kernel (float and double scales): 225 code pairs + nibble positions, "
                     "HANDOFF/proposal fixtures, exact scalar/INT64 agreement over partial tiles and "
                     "tile edges 1-64, zero groups, extremes, rejection rules\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
