// Qinyan Zeng: production rotation/operator boundaries and transactional prepare.
#include "int4/blocked_kernel.hpp"
#include "int4/linear.hpp"
#include "bench/metrics.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using int4::Matrix;
using int4::LinearMode;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void rejects(Function fn) {
    try { fn(); } catch (const Exception&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}

template <typename Scale>
class ObservedKernel final : public int4::MatmulKernel<Scale> {
public:
    mutable std::size_t calls = 0;
    Matrix<std::int32_t> run(const int4::PackedInt4Matrix<Scale>& x,
                             const int4::PackedInt4Matrix<Scale>& w) const override {
        ++calls;
        return int4::ScalarKernel<Scale>().run(x, w);
    }
};

void rotation_boundaries() {
    rejects<std::invalid_argument>([] { (void)int4::HadamardRotation(2048, 0); });
    rejects<std::invalid_argument>([] {
        (void)int4::HadamardRotation(std::numeric_limits<std::size_t>::max(), 0);
    });
    int4::HadamardRotation rotation(4, 0);
    const float large = std::numeric_limits<float>::max() * 0.75F;
    Matrix<float> x(1, 4, {large * rotation.signs()[0], large * rotation.signs()[1], 0, 0});
    rotation.apply_right(x);
    require(x(0, 0) == large && x(0, 1) == 0 && x(0, 2) == large && x(0, 3) == 0,
            "double butterfly avoids spurious intermediate FP32 overflow");
    Matrix<float> w(4, 1, {large * rotation.signs()[0], large * rotation.signs()[1], 0, 0});
    rotation.apply_transpose_left(w);
    require(w(0, 0) == large && w(2, 0) == large, "left rotation avoids intermediate overflow");
    Matrix<float> overflow(1, 4);
    for (std::size_t i = 0; i < 4; ++i) {
        overflow(0, i) = std::numeric_limits<float>::max() * rotation.signs()[i];
    }
    rejects<std::overflow_error>([&] { rotation.apply_right(overflow); });
    Matrix<float> nonfinite(1, 4, {1, 2, std::numeric_limits<float>::infinity(), 4});
    rejects<std::invalid_argument>([&] { rotation.apply_right(nonfinite); });

    // Exercise all proposal K values, including K=1 and K=1024, and several seeds.
    for (std::size_t k = 1; k <= 1024; k *= 2) {
        for (std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{2},
                                   std::numeric_limits<std::uint64_t>::max()}) {
            int4::HadamardRotation r(k, seed);
            Matrix<float> a(3, k);
            Matrix<float> b(k, 5);
            for (std::size_t i = 0; i < a.size(); ++i) {
                a.data()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7;
            }
            for (std::size_t i = 0; i < b.size(); ++i) {
                b.data()[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 5;
            }
            const auto reference = bench::reference_fp64(a, b);
            r.apply_right(a);
            r.apply_transpose_left(b);
            const auto error = bench::compute_errors(bench::fp32_matmul(a, b), reference);
            require(error.rel_frobenius <= 1e-5, "unquantized identity across the full K domain");
        }
    }
}

template <typename Scale>
void lifecycle() {
    const Matrix<float> x(1, 2, {1, -1});
    const Matrix<float> w(2, 1, {7, 0});
    rejects<std::invalid_argument>([] {
        (void)int4::QuantizedLinear<Scale>(static_cast<LinearMode>(99));
    });
    for (auto mode : {LinearMode::Fp32, LinearMode::Direct, LinearMode::Rotated}) {
        int4::QuantizedLinear<Scale> op(mode, 2);
        ObservedKernel<Scale> kernel;
        require(!op.prepared(), "initially unprepared");
        rejects<std::logic_error>([&] { (void)op.weight_bytes(); });
        rejects<std::logic_error>([&] { (void)op.run(x, kernel); });
        op.prepare(w);
        require(op.prepared(), "successful prepare");
        const auto before = op.run(x, kernel);
        require(std::abs(before(0, 0) - 7.0F) < 1e-5F, "proposal fixture");
        require(kernel.calls == (mode == LinearMode::Fp32 ? 0U : 1U), "kernel dispatch only for INT4");
        rejects<std::invalid_argument>([&] { op.prepare(Matrix<float>(3, 1)); });
        rejects<std::invalid_argument>([&] { op.prepare(Matrix<float>(2048, 1)); });
        rejects<std::invalid_argument>([&] { op.prepare(Matrix<float>(2, 1025)); });
        rejects<std::invalid_argument>([&] {
            op.prepare(Matrix<float>(4, 1, {1, 2, 3, std::nanf("")}));
        });
        require(bench::identical(before, op.run(x, kernel)), "failed prepare preserves prior output");
        rejects<std::invalid_argument>([&] { (void)op.run(Matrix<float>(33, 2), kernel); });
        rejects<std::invalid_argument>([&] { (void)op.run(Matrix<float>(1, 4), kernel); });
        rejects<std::invalid_argument>([&] {
            (void)op.run(Matrix<float>(1, 2, {1, std::nanf("")}), kernel);
        });
        if (mode == LinearMode::Fp32) {
            rejects<std::logic_error>([&] { (void)op.packed_weights(); });
            rejects<std::logic_error>([&] { (void)op.quantize_input(x); });
            require(op.weight_bytes() == 8 && op.signs_bytes() == 0, "FP32 storage");
        }
        // Reprepare with a different K and odd N; the operator owns its weights.
        Matrix<float> replacement(1, 33);
        for (std::size_t i = 0; i < 33; ++i) replacement(0, i) = static_cast<float>(i);
        op.prepare(replacement);
        replacement(0, 32) = -100;
        const auto output = op.run(Matrix<float>(32, 1, std::vector<float>(32, 7)), kernel);
        require(output.rows() == 32 && output.cols() == 33 &&
                    std::abs(output(31, 32) - 224) < 1e-4F, "new shape, K=1, N=33, owned weights");
        auto moved = std::move(op);
        require(!op.prepared() && moved.prepared(), "move transfers prepared state");
        rejects<std::logic_error>([&] { (void)op.run(x, kernel); });
    }
    int4::QuantizedLinear<Scale> rotated(LinearMode::Rotated);
    rotated.prepare(w);
    const auto bytes_before = rotated.packed_weights().bytes();
    const auto signs_before = rotated.rotation()->signs();
    Matrix<float> too_large(4, 1);
    const int4::HadamardRotation signs(4, 0);
    for (std::size_t i = 0; i < 4; ++i) {
        too_large(i, 0) = std::numeric_limits<float>::max() * signs.signs()[i];
    }
    rejects<std::overflow_error>([&] { rotated.prepare(too_large); });
    require(rotated.packed_weights().bytes() == bytes_before &&
                rotated.rotation()->signs() == signs_before, "rotation failure preserves all prepared state");

    if constexpr (std::is_same_v<Scale, float>) {
        int4::QuantizedLinear<Scale> direct(LinearMode::Direct);
        direct.prepare(w);
        rejects<std::underflow_error>([&] {
            direct.prepare(Matrix<float>(1, 1, {std::numeric_limits<float>::denorm_min()}));
        });
        require(std::abs(direct.run(x, int4::ScalarKernel<Scale>())(0, 0) - 7) < 1e-5F,
                "quantization underflow preserves old weights");
    }
    int4::QuantizedLinear<Scale> fp32(LinearMode::Fp32);
    fp32.prepare(Matrix<float>(1, 1, {std::numeric_limits<float>::max()}));
    rejects<std::overflow_error>([&] {
        (void)fp32.run(Matrix<float>(1, 1, {2}), int4::ScalarKernel<Scale>());
    });
}
} // namespace

int main() {
    try {
        rotation_boundaries();
        lifecycle<float>();
        lifecycle<double>();
        std::cout << "PASS: production rotation, FP32/INT4 lifecycle, domain and numerical boundaries\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
