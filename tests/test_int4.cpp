#include "int4/core.hpp"

#include <iostream>
#include <random>
#include <string>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void rejects(Function fn) {
    try { fn(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

template <typename Scale>
void test_all_codes() {
    const int4::ScalarKernel<Scale> kernel;
    for (int a = -7; a <= 7; ++a) {
        for (int b = -7; b <= 7; ++b) {
            // 225 known adjacent code pairs. max=7 -> exact scale 1.
            const int4::Matrix<float> values(1, 3, {static_cast<float>(a), static_cast<float>(b), 7});
            const auto q = int4::quantize_int4<Scale>(values, int4::ScaleAxis::Row);
            const unsigned low = static_cast<unsigned>(a) & 15U;
            const unsigned high = static_cast<unsigned>(b) & 15U;
            require(q.bytes().size() == 2 && q.bytes()[0] == (low | (high << 4)),
                    "known nibble pair encoding");
            require(q.bytes()[1] == 7 && q.code(0, 0) == a && q.code(0, 1) == b,
                    "signed decoding and odd-row padding");
            // 225 exact one-term signed dot products (includes K=1).
            const int4::PackedInt4Matrix<Scale> x(1, 1, int4::ScaleAxis::Row,
                {static_cast<std::uint8_t>(low)}, {Scale{1}});
            const int4::PackedInt4Matrix<Scale> w(1, 1, int4::ScaleAxis::Column,
                {static_cast<std::uint8_t>(high)}, {Scale{1}});
            require(kernel.run(x, w)(0, 0) == a * b, "signed dot product");
        }
    }
}

template <typename Scale>
void test_scales() {
    const auto ties = int4::quantize_int4<Scale>(
        int4::Matrix<float>(1, 5, {7, 0.5F, -0.5F, 1.5F, -1.5F}), int4::ScaleAxis::Row);
    const int expected[] = {7, 1, -1, 2, -2};
    for (std::size_t i = 0; i < 5; ++i) require(ties.code(0, i) == expected[i], "tie rounding");
    const auto rows = int4::quantize_int4<Scale>(
        int4::Matrix<float>(3, 2, {0, 0, 7, -7, 14, -14}), int4::ScaleAxis::Row);
    require(rows.scales() == std::vector<Scale>({Scale{1}, Scale{1}, Scale{2}}), "row scales");
    require(rows.bytes()[0] == 0 && rows.bytes()[1] == 0x97 && rows.bytes()[2] == 0x97,
            "zero and per-row codes");
    const auto cols = int4::quantize_int4<Scale>(
        int4::Matrix<float>(2, 3, {7, 14, 0, -7, -14, 0}), int4::ScaleAxis::Column);
    require(cols.scales() == std::vector<Scale>({Scale{1}, Scale{2}, Scale{1}}), "column scales");
    require(cols.bytes().size() == 4 && cols.bytes()[0] == 0x77 && cols.bytes()[2] == 0x99,
            "column scaling must not transpose row-major storage");
    require(cols.storage_bytes() == 4 + 3 * sizeof(Scale), "precision-dependent metadata bytes");
    const auto positive = int4::quantize_int4<Scale>(
        int4::Matrix<float>(1, 3, {1, 1.5F, 2}), int4::ScaleAxis::Row);
    const auto negative = int4::quantize_int4<Scale>(
        int4::Matrix<float>(1, 3, {-1, -1.5F, -2}), int4::ScaleAxis::Row);
    require(positive.code(0, 0) < positive.code(0, 2), "positive range must not collapse");
    require(negative.code(0, 0) > negative.code(0, 2), "negative range must not collapse");
    const auto constant = int4::quantize_int4<Scale>(
        int4::Matrix<float>(1, 2, {-3, -3}), int4::ScaleAxis::Row);
    require(constant.code(0, 0) == -7 && constant.code(0, 1) == -7, "nonzero constant group");
    // Scale storage must be rounded before encoding, identically in both variants.
    const auto rounded_scale = int4::quantize_int4<Scale>(
        int4::Matrix<float>(1, 2, {1, 0.5F}), int4::ScaleAxis::Row);
    require(rounded_scale.scales()[0] == static_cast<Scale>(1.0 / 7.0), "stored scale type");
    const int expected_second = static_cast<int>(std::round(0.5 / rounded_scale.scales()[0]));
    require(rounded_scale.code(0, 1) == expected_second, "encoding must use the stored scale");
}

template <typename Scale>
void test_references() {
    std::mt19937 rng(538);
    std::uniform_real_distribution<float> distribution(-2, 2);
    const int4::ScalarKernel<Scale> scalar;
    const int4::MatmulKernel<Scale>& kernel = scalar; // Public virtual interface.
    for (std::size_t k_count : {1U, 2U, 4U, 32U}) {
        for (std::size_t n_count : {1U, 3U, 33U}) {
            int4::Matrix<float> x(3, k_count);
            int4::Matrix<float> w(k_count, n_count);
            for (std::size_t i = 0; i < x.size(); ++i) x.data()[i] = distribution(rng);
            for (std::size_t i = 0; i < w.size(); ++i) w.data()[i] = distribution(rng);
            x(0, 0) = 100;
            // Explicit zero row tests scaling and zero-dot reconstruction.
            for (std::size_t k = 0; k < k_count; ++k) x(2, k) = 0;
            const auto qx = int4::quantize_int4<Scale>(x, int4::ScaleAxis::Row);
            const auto qw = int4::quantize_int4<Scale>(w, int4::ScaleAxis::Column);
            const auto actual = kernel.run(qx, qw);
            const auto y = int4::rescale_output(actual, qx.scales(), qw.scales());
            int4::Matrix<std::int64_t> reference(x.rows(), n_count);
            // Independent integer loop order, using source bytes (not code()).
            for (std::size_t k = 0; k < k_count; ++k) {
                for (std::size_t n = 0; n < n_count; ++n) {
                    const unsigned wn = (qw.bytes()[k * qw.row_bytes() + n / 2] >> (4 * (n % 2))) & 15U;
                    const int wcode = wn <= 7 ? static_cast<int>(wn) : static_cast<int>(wn) - 16;
                    for (std::size_t m = 0; m < x.rows(); ++m) {
                        const unsigned xn = (qx.bytes()[m * qx.row_bytes() + k / 2] >> (4 * (k % 2))) & 15U;
                        const int xcode = xn <= 7 ? static_cast<int>(xn) : static_cast<int>(xn) - 16;
                        reference(m, n) += static_cast<std::int64_t>(xcode) * wcode;
                    }
                }
            }
            for (std::size_t m = 0; m < x.rows(); ++m) {
                for (std::size_t n = 0; n < n_count; ++n) {
                    require(actual(m, n) == reference(m, n), "INT64 reference mismatch");
                    double decoded_reference = 0;
                    double original_reference = 0;
                    double bound = 0;
                    for (std::size_t k = 0; k < k_count; ++k) {
                        const double xhat = static_cast<double>(qx.code(m, k)) * qx.scales()[m];
                        const double what = static_cast<double>(qw.code(k, n)) * qw.scales()[n];
                        decoded_reference += xhat * what;
                        original_reference += static_cast<double>(x(m, k)) * w(k, n);
                        const double dx = std::abs(xhat - x(m, k));
                        const double dw = std::abs(what - w(k, n));
                        bound += std::abs(static_cast<double>(x(m, k))) * dw
                               + std::abs(static_cast<double>(w(k, n))) * dx + dx * dw;
                    }
                    const double tolerance = 1e-6 + 1e-5 * std::abs(decoded_reference);
                    require(std::abs(y(m, n) - decoded_reference) <= tolerance, "same-code FP64 reference");
                    require(std::abs(y(m, n) - original_reference) <= bound + tolerance,
                            "original FP64 reference error bound");
                    if (m == 2) require(y(m, n) == 0, "zero row output");
                }
            }
        }
    }
    // The proposal's hand-checkable K=2 example.
    const auto x = int4::quantize_int4<Scale>(int4::Matrix<float>(1, 2, {1, -1}), int4::ScaleAxis::Row);
    const auto w = int4::quantize_int4<Scale>(int4::Matrix<float>(2, 1, {7, 0}), int4::ScaleAxis::Column);
    const auto y = int4::rescale_output(kernel.run(x, w), x.scales(), w.scales());
    require(std::abs(y(0, 0) - 7) < 1e-6F, "proposal fixture");
}

template <typename Scale>
void test_errors() {
    using Packed = int4::PackedInt4Matrix<Scale>;
    const auto axis = int4::ScaleAxis::Row;
    rejects<std::invalid_argument>([&] { (void)Packed(1, 1, axis, {8}, {Scale{1}}); });
    rejects<std::invalid_argument>([&] { (void)Packed(1, 1, axis, {0x10}, {Scale{1}}); });
    rejects<std::invalid_argument>([&] { (void)Packed(1, 2, axis, {}, {Scale{1}}); });
    rejects<std::invalid_argument>([&] { (void)Packed(1, 2, axis, {0}, {}); });
    rejects<std::invalid_argument>([&] { (void)Packed(0, 2, axis, {}, {}); });
    rejects<std::invalid_argument>([&] {
        (void)Packed(1, 2, static_cast<int4::ScaleAxis>(99), {0}, {Scale{1}});
    });
    rejects<std::length_error>([&] {
        (void)Packed(std::numeric_limits<std::size_t>::max(), 2, axis, {}, {});
    });
    for (Scale bad : {Scale{0}, Scale{-1}, std::numeric_limits<Scale>::infinity(),
                      std::numeric_limits<Scale>::quiet_NaN()}) {
        rejects<std::invalid_argument>([&] { (void)Packed(1, 1, axis, {0}, {bad}); });
    }
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        rejects<std::invalid_argument>([&] {
            (void)int4::quantize_int4<Scale>(int4::Matrix<float>(1, 1, {bad}), axis);
        });
    }
    const Packed a(1, 2, axis, {0}, {Scale{1}});
    const Packed w(1, 1, int4::ScaleAxis::Column, {0}, {Scale{1}});
    const int4::ScalarKernel<Scale> scalar;
    rejects<std::invalid_argument>([&] { (void)scalar.run(a, w); });
    rejects<std::invalid_argument>([&] { (void)scalar.run(w, w); });
    rejects<std::out_of_range>([&] { (void)a.code(0, 2); });
    rejects<std::invalid_argument>([&] {
        (void)int4::rescale_output(int4::Matrix<std::int32_t>(1, 2), std::vector<Scale>{Scale{1}},
                                  std::vector<Scale>{Scale{1}});
    });
    rejects<std::invalid_argument>([&] {
        (void)int4::rescale_output(int4::Matrix<std::int32_t>(1, 1), std::vector<Scale>{Scale{0}},
                                  std::vector<Scale>{Scale{1}});
    });
    rejects<std::overflow_error>([&] {
        (void)int4::rescale_output(int4::Matrix<std::int32_t>(1, 1, {49}),
            std::vector<Scale>{std::numeric_limits<Scale>::max()}, std::vector<Scale>{Scale{2}});
    });
    int4::check_inner_dimension(int4::max_safe_k);
    rejects<std::overflow_error>([] { int4::check_inner_dimension(int4::max_safe_k + 1); });
    rejects<std::invalid_argument>([] { int4::check_inner_dimension(0); });
    require(static_cast<std::int64_t>(int4::max_safe_k) * 49 <= std::numeric_limits<std::int32_t>::max(),
            "INT32 safe bound");
    require(static_cast<std::int64_t>(int4::max_safe_k + 1) * 49 > std::numeric_limits<std::int32_t>::max(),
            "INT32 rejected bound");
}
} // namespace

int main() {
    try {
        test_all_codes<float>(); test_all_codes<double>();
        test_scales<float>(); test_scales<double>();
        test_references<float>(); test_references<double>();
        test_errors<float>(); test_errors<double>();
        const int4::Matrix<float> boundary(1, 2, {1, 0.5F});
        const auto qf = int4::quantize_int4<float>(boundary, int4::ScaleAxis::Row);
        const auto qd = int4::quantize_int4<double>(boundary, int4::ScaleAxis::Row);
        require(qf.code(0, 1) == 3 && qd.code(0, 1) == 4,
                "float and double may differ at a stored-scale rounding boundary");
        const int4::Matrix<float> tiny(1, 1, {std::numeric_limits<float>::denorm_min()});
        rejects<std::underflow_error>([&] { (void)int4::quantize_int4<float>(tiny, int4::ScaleAxis::Row); });
        const auto q = int4::quantize_int4<double>(tiny, int4::ScaleAxis::Row);
        require(q.code(0, 0) == 7 && q.scales()[0] > 0, "double scale handles tiny FP32 input");
        std::cout << "PASS: float and double INT4 variants; 225 encoding pairs + 225 dot products "
                     "per variant; scales, padding, references, errors, INT32 bounds\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
