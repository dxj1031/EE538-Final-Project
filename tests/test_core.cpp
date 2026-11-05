#include "int8/scalar_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_throw(Function fn) {
    try { fn(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

void test_storage() {
    int8::Matrix<float> a(2, 3, {1, 2, 3, 4, 5, 6});
    require(a.row_data(1) == a.data() + 3, "row-major layout");
    require(a.storage_bytes() == 6 * sizeof(float), "logical storage bytes");
    auto copy = a;
    copy(0, 0) = 99;
    require(a(0, 0) == 1, "copy must own independent storage");
    auto moved = std::move(copy);
    require(moved(0, 0) == 99 && copy.size() == 0 && copy.rows() == 0,
            "move must transfer storage and empty the source");
    a = moved;
    moved(0, 0) = 7;
    require(a(0, 0) == 99, "copy assignment");
    a = std::move(moved);
    require(a(0, 0) == 7 && moved.cols() == 0, "move assignment");
    const auto& const_a = a;
    require(const_a(1, 2) == 6, "const access");
    expect_throw<std::out_of_range>([&] { (void)a(2, 0); });
    expect_throw<std::out_of_range>([&] { (void)const_a(0, 3); });
    expect_throw<std::out_of_range>([&] { (void)moved.row_data(0); });
    expect_throw<std::invalid_argument>([] { int8::Matrix<float> bad(0, 3); });
    expect_throw<std::invalid_argument>([] { int8::Matrix<float> bad(2, 3, {1}); });
    expect_throw<std::length_error>([] {
        int8::Matrix<float> bad(std::numeric_limits<std::size_t>::max(), 2);
    });
}

void test_quantization() {
    // max=254 -> scale=2: explicit half-way ties establish the rounding contract.
    const int8::Matrix<float> source(1, 5, {254, 1, -1, 3, -3});
    const auto q = int8::quantize(source, int8::ScaleMode::PerTensor);
    const int expected[] = {127, 1, -1, 2, -2};
    require(q.scales.size() == 1 && q.scales[0] == 2.0, "tensor scale");
    for (std::size_t i = 0; i < 5; ++i) {
        require(q.values(0, i) == expected[i], "ties must round away from zero");
    }
    const int8::Matrix<float> constants(3, 3, {0, 0, 0, 2, 2, 2, -3, -3, -3});
    const auto qc = int8::quantize(constants, int8::ScaleMode::PerRow);
    require(qc.scales[0] == 1, "zero row scale");
    const auto restored = int8::dequantize(qc);
    for (std::size_t i = 0; i < constants.size(); ++i) {
        require(restored.data()[i] == constants.data()[i], "constant row round trip");
    }
    const int8::Matrix<float> one_sided(2, 3, {1, 1.5F, 2, -1, -1.5F, -2});
    const auto qs = int8::quantize(one_sided, int8::ScaleMode::PerRow);
    require(qs.values(0, 0) < qs.values(0, 2), "positive range must not collapse");
    require(qs.values(1, 0) > qs.values(1, 2), "negative range must not collapse");
    for (float extreme : {std::numeric_limits<float>::denorm_min(),
                          std::numeric_limits<float>::max()}) {
        const int8::Matrix<float> tiny_or_large(1, 2, {extreme, -extreme});
        const auto qt = int8::quantize(tiny_or_large, int8::ScaleMode::PerTensor);
        const auto rt = int8::dequantize(qt);
        require(qt.scales[0] > 0 && rt(0, 0) == extreme && rt(0, 1) == -extreme,
                "finite extreme input must survive round trip");
    }
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        expect_throw<std::invalid_argument>([&] {
            (void)int8::quantize(int8::Matrix<float>(1, 1, {bad}), int8::ScaleMode::PerTensor);
        });
    }
    expect_throw<std::invalid_argument>([&] {
        (void)int8::quantize(source, static_cast<int8::ScaleMode>(99));
    });
}

void test_known_linear() {
    // Scale=1 for both operands; integer sums and bias outputs are hand-checkable.
    const int8::Matrix<float> x(2, 2, {127, -127, 0, 127});
    const int8::Matrix<float> w(2, 2, {127, 0, -127, 127});
    const auto qx = int8::quantize(x, int8::ScaleMode::PerTensor);
    const auto qw = int8::quantize(w, int8::ScaleMode::PerRow);
    const auto s = int8::scalar_accumulate(qx.values, qw.values);
    require(s(0, 0) == 16129 && s(0, 1) == -32258 &&
            s(1, 0) == 0 && s(1, 1) == 16129, "hand-calculated accumulators");
    const auto y = int8::scalar_linear(qx, qw, {0.5F, -0.5F});
    require(y(0, 0) == 16129.5F && y(0, 1) == -32258.5F &&
            y(1, 0) == 0.5F && y(1, 1) == 16128.5F, "bias and reconstruction");
    const auto no_bias = int8::scalar_linear(qx, qw);
    require(no_bias(1, 1) == 16129, "optional bias");
}

void test_random_reference() {
    std::mt19937 rng(538);
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    // Odd K/N and M=1 establish that there is no grouping or tile restriction.
    for (const std::size_t m : {1U, 4U}) {
        for (const std::size_t k_count : {1U, 3U, 17U, 129U}) {
            const std::size_t n_count = 5;
            int8::Matrix<float> x(m, k_count);
            int8::Matrix<float> w(n_count, k_count);
            for (std::size_t i = 0; i < x.size(); ++i) x.data()[i] = distribution(rng);
            for (std::size_t i = 0; i < w.size(); ++i) w.data()[i] = distribution(rng);
            if (k_count > 3) x(0, 0) = 100; // An outlier shares the input scale.
            const auto qx = int8::quantize(x, int8::ScaleMode::PerTensor);
            const auto qw = int8::quantize(w, int8::ScaleMode::PerRow);
            const auto actual = int8::scalar_accumulate(qx.values, qw.values);
            // Independent INT64 reference with a different loop order.
            int8::Matrix<std::int64_t> reference(m, n_count);
            for (std::size_t k = 0; k < k_count; ++k)
                for (std::size_t n = 0; n < n_count; ++n)
                    for (std::size_t r = 0; r < m; ++r)
                        reference(r, n) += static_cast<std::int64_t>(qx.values(r, k))
                                         * static_cast<std::int64_t>(qw.values(n, k));
            const std::vector<float> bias(n_count, 0.25F);
            const auto y = int8::scalar_linear(qx, qw, bias);
            for (std::size_t r = 0; r < m; ++r) {
                for (std::size_t n = 0; n < n_count; ++n) {
                    require(actual(r, n) == reference(r, n), "INT64 reference mismatch");
                    double original = bias[n];
                    double reconstructed = bias[n];
                    double error_bound = 0;
                    const double ex = qx.scales[0] / 2;
                    const double ew = qw.scales[n] / 2;
                    for (std::size_t k = 0; k < k_count; ++k) {
                        original += static_cast<double>(x(r, k)) * w(n, k);
                        const double xhat = static_cast<double>(qx.values(r, k)) * qx.scales[0];
                        const double what = static_cast<double>(qw.values(n, k)) * qw.scales[n];
                        reconstructed += xhat * what;
                        error_bound += std::abs(static_cast<double>(x(r, k))) * ew
                                     + std::abs(static_cast<double>(w(n, k))) * ex + ex * ew;
                    }
                    const double tolerance = 1e-5 * std::max(1.0, std::abs(reconstructed));
                    require(std::abs(y(r, n) - reconstructed) <= tolerance,
                            "output must match FP64 reconstructed-operand reference");
                    require(std::abs(y(r, n) - original) <= error_bound + tolerance,
                            "quantization error exceeds independent rounding bound");
                }
            }
            const auto again = int8::scalar_linear(qx, qw, bias);
            for (std::size_t i = 0; i < y.size(); ++i)
                require(y.data()[i] == again.data()[i], "repeated calls must be stable");
        }
    }
}

void test_rejections() {
    auto x = int8::quantize(int8::Matrix<float>(1, 2, {1, 2}), int8::ScaleMode::PerTensor);
    auto w = int8::quantize(int8::Matrix<float>(1, 2, {3, 4}), int8::ScaleMode::PerRow);
    expect_throw<std::invalid_argument>([&] { (void)int8::scalar_linear(x, w, {1, 2}); });
    expect_throw<std::invalid_argument>([&] {
        (void)int8::scalar_linear(x, w, {std::numeric_limits<float>::infinity()});
    });
    expect_throw<std::invalid_argument>([&] {
        (void)int8::scalar_accumulate(x.values, int8::Matrix<std::int8_t>(1, 3));
    });
    auto invalid = x;
    invalid.mode = int8::ScaleMode::PerRow;
    expect_throw<std::invalid_argument>([&] { (void)int8::scalar_linear(invalid, w); });
    invalid = x;
    invalid.scales.clear();
    expect_throw<std::invalid_argument>([&] { (void)int8::dequantize(invalid); });
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        invalid = x;
        invalid.scales[0] = bad;
        expect_throw<std::invalid_argument>([&] { (void)int8::scalar_linear(invalid, w); });
    }
    invalid = x;
    invalid.values(0, 0) = -128;
    expect_throw<std::invalid_argument>([&] { (void)int8::dequantize(invalid); });
    expect_throw<std::invalid_argument>([&] { (void)int8::scalar_accumulate(invalid.values, w.values); });
    invalid = x;
    invalid.scales[0] = std::numeric_limits<double>::max();
    expect_throw<std::overflow_error>([&] { (void)int8::dequantize(invalid); });
    const auto large = int8::quantize(
        int8::Matrix<float>(1, 1, {std::numeric_limits<float>::max()}), int8::ScaleMode::PerTensor);
    const auto two = int8::quantize(int8::Matrix<float>(1, 1, {2}), int8::ScaleMode::PerRow);
    expect_throw<std::overflow_error>([&] { (void)int8::scalar_linear(large, two); });
    auto moved_from = x;
    auto destination = std::move(moved_from);
    (void)destination;
    expect_throw<std::invalid_argument>([&] { (void)int8::scalar_linear(moved_from, w); });
    int8::Matrix<float> empty_source(1, 1);
    auto owned = std::move(empty_source);
    (void)owned;
    expect_throw<std::invalid_argument>([&] {
        (void)int8::quantize(empty_source, int8::ScaleMode::PerTensor);
    });
}

void test_accumulation_limit() {
    int8::Matrix<std::int8_t> x(1, int8::max_safe_k);
    int8::Matrix<std::int8_t> w(1, int8::max_safe_k);
    std::fill_n(x.data(), x.size(), static_cast<std::int8_t>(127));
    std::fill_n(w.data(), w.size(), static_cast<std::int8_t>(127));
    const auto result = int8::scalar_accumulate(x, w);
    const std::int64_t expected = static_cast<std::int64_t>(int8::max_safe_k) * 127 * 127;
    require(result(0, 0) == expected, "largest allowed worst-case dot product");
    w(0, 0) = -127;
    require(int8::scalar_accumulate(x, w)(0, 0) == expected - 2 * 127 * 127,
            "mixed-sign accumulation near the limit");
    std::fill_n(w.data(), w.size(), static_cast<std::int8_t>(-127));
    require(int8::scalar_accumulate(x, w)(0, 0) == -expected, "negative limit");
    const int8::Matrix<std::int8_t> too_wide(1, int8::max_safe_k + 1);
    expect_throw<std::overflow_error>([&] { (void)int8::scalar_accumulate(too_wide, too_wide); });
}
} // namespace

int main() {
    try {
        test_storage();
        test_quantization();
        test_known_linear();
        test_random_reference();
        test_rejections();
        test_accumulation_limit();
        std::cout << "PASS: 6 test groups (storage, quantization, known outputs, "
                     "independent references, invalid inputs, accumulation limits)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
