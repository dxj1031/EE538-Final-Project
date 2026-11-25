// Member 3: tests for the benchmark support code (include/bench/*.hpp).
//
// Covers timing statistics, reproducible workload generation, FP64 reference and error
// metrics, the FP32 baseline, the rotation stand-in against a dense Q for K <= 64 with
// shared signs, the prepared-operator lifecycle, storage accounting, and binary dumps.

#include "bench/accounting.hpp"
#include "bench/dump.hpp"
#include "bench/linear.hpp"
#include "bench/metrics.hpp"
#include "bench/stats.hpp"
#include "bench/workload.hpp"
#include "int4/blocked_kernel.hpp"
#include "int4/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using int4::Matrix;

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

bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

Matrix<float> random_matrix(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-2, 2);
    Matrix<float> m(rows, cols);
    for (std::size_t i = 0; i < m.size(); ++i) m.data()[i] = distribution(rng);
    return m;
}

// ---------------------------------------------------------------------------------
void test_stats() {
    const auto five = bench::summarize({50, 10, 40, 20, 30});
    require(five.median == 30 && five.q1 == 20 && five.q3 == 40 && five.iqr == 20 && five.count == 5,
            "five-sample median/IQR uses order statistics 2..4");
    const auto four = bench::summarize({4, 1, 3, 2});
    require(four.median == 2.5 && four.q1 == 1.75 && four.q3 == 3.25, "linear interpolation quantiles");
    const auto one = bench::summarize({7});
    require(one.median == 7 && one.iqr == 0, "single sample");
    rejects<std::invalid_argument>([] { (void)bench::summarize({}); }, "empty sample");
    rejects<std::invalid_argument>([] { (void)bench::summarize({1, std::nan("")}); }, "NaN sample");
    rejects<std::invalid_argument>([] { (void)bench::quantile_sorted({1, 2}, 1.5); }, "bad probability");

    int calls = 0;
    std::vector<double> samples;
    const int last = bench::measure([&] { return ++calls; }, 2, 5, samples);
    require(calls == 7 && last == 7 && samples.size() == 5, "2 warm-ups + 5 timed runs; last result returned");
    for (double s : samples) require(s >= 0 && std::isfinite(s), "durations are finite and non-negative");
    rejects<std::invalid_argument>([&] { (void)bench::measure([] { return 0; }, 1, 0, samples); }, "zero runs");
}

// ---------------------------------------------------------------------------------
void test_workload() {
    require(bench::outlier_count(256) == 3 && bench::outlier_count(512) == 6 && bench::outlier_count(1024) == 11,
            "ceil(0.01K) outlier columns");
    require(bench::outlier_count(1) == 1 && bench::outlier_count(100) == 1 && bench::outlier_count(101) == 2,
            "outlier count boundaries");

    const auto a = bench::generate_scale_data(256, 256, 0);
    const auto b = bench::generate_scale_data(256, 256, 0);
    const auto c = bench::generate_scale_data(256, 256, 1);
    require(a.x0.rows() == 32 && a.x0.cols() == 256 && a.w.rows() == 256 && a.w.cols() == 256, "shapes");
    require(bench::identical(a.x0, b.x0) && bench::identical(a.w, b.w) && a.outlier_cols == b.outlier_cols,
            "same seed reproduces data");
    require(!bench::identical(a.x0, c.x0) && !bench::identical(a.w, c.w), "different seeds differ");
    require(a.outlier_cols.size() == 3, "outlier column count");
    for (std::size_t i = 0; i < a.outlier_cols.size(); ++i) {
        require(a.outlier_cols[i] < 256, "outlier column in range");
        if (i > 0) require(a.outlier_cols[i - 1] < a.outlier_cols[i], "outlier columns sorted and distinct");
    }

    // Generator regression values (mt19937_64 + seed_seq are fully specified by the
    // standard; tolerance allows last-ulp libm differences across platforms).
    const auto tiny = bench::generate_scale_data(4, 4, 0);
    const double golden_x0[3] = {-1.31707227, 0.932350278, 0.513112366};
    const double golden_w[3] = {0.0500178672, -1.46872509, -0.211967513};
    for (std::size_t i = 0; i < 3; ++i) {
        require(near(tiny.x0.data()[i], golden_x0[i], 1e-6) && near(tiny.w.data()[i], golden_w[i], 1e-6),
                "portable Gaussian stream regression values");
    }
    require(tiny.outlier_cols == std::vector<std::size_t>({2}), "outlier column regression value");
    double sum = 0;
    double sum_sq = 0;
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        sum += a.w.data()[i];
        sum_sq += static_cast<double>(a.w.data()[i]) * a.w.data()[i];
    }
    const double count = static_cast<double>(a.w.size());
    const double mean = sum / count;
    const double variance = sum_sq / count - mean * mean;
    require(std::abs(mean) < 0.02 && std::abs(variance - 1) < 0.03, "W is approximately standard normal");

    // Case inputs: prefix rows of X0, outlier columns scaled by a, others unchanged.
    const Matrix<float> x8 = bench::make_case_input(a, 8, 10);
    const Matrix<float> x1 = bench::make_case_input(a, 1, 10);
    const Matrix<float> control = bench::make_case_input(a, 8, 1);
    for (std::size_t m = 0; m < 8; ++m) {
        for (std::size_t k = 0; k < 256; ++k) {
            const bool outlier = std::find(a.outlier_cols.begin(), a.outlier_cols.end(), k) != a.outlier_cols.end();
            require(x8(m, k) == (outlier ? a.x0(m, k) * 10.0F : a.x0(m, k)), "outlier scaling");
            require(control(m, k) == a.x0(m, k), "a = 1 control is unchanged");
            if (m == 0) require(x1(0, k) == x8(0, k), "first-M-rows prefix property");
        }
    }
    rejects<std::invalid_argument>([&] { (void)bench::make_case_input(a, 0, 10); }, "M = 0");
    rejects<std::invalid_argument>([&] { (void)bench::make_case_input(a, 33, 10); }, "M = 33");
    rejects<std::invalid_argument>([&] { (void)bench::make_case_input(a, 8, 0); }, "a = 0");

    const auto cases = bench::protocol_cases({256, 512, 1024});
    require(cases.size() == 15, "15 distinct cases per seed");
    std::size_t shared = 0;
    for (const auto& spec : cases) {
        require(spec.k == spec.n, "K = N");
        require((spec.a == 10.0F && (spec.m == 1 || spec.m == 8 || spec.m == 32)) ||
                    (spec.m == 8 && (spec.a == 1.0F || spec.a == 100.0F)),
                "only the two controlled sweeps");
        if (spec.m == 8 && spec.a == 10.0F) {
            require(spec.sweep == "both", "shared case label");
            ++shared;
        }
    }
    require(shared == 3, "M = 8, a = 10 listed once per scale");
}

// ---------------------------------------------------------------------------------
void test_metrics_and_fp32() {
    // Proposal example: X = [1, -1], W = [7, 0]^T -> Y = 7.
    const Matrix<float> x(1, 2, {1, -1});
    const Matrix<float> w(2, 1, {7, 0});
    const auto y64 = bench::reference_fp64(x, w);
    require(y64(0, 0) == 7.0, "FP64 reference fixture");
    bench::Fp32Linear op;
    op.prepare(w);
    require(op.run(x)(0, 0) == 7.0F, "FP32 baseline fixture");

    const Matrix<float> y(1, 3, {1, 2, 3});
    const Matrix<double> ref(1, 3, {1, 2, 5});
    const auto e = bench::compute_errors(y, ref);
    require(e.max_abs == 2.0 && near(e.rel_frobenius, 2.0 / std::sqrt(30.0), 1e-15), "error metrics");
    const auto zero = bench::compute_errors(Matrix<float>(1, 1, {1e-13F}), Matrix<double>(1, 1, {0.0}));
    require(near(zero.rel_frobenius, static_cast<double>(1e-13F) / 1e-12, 1e-12), "1e-12 denominator floor");
    rejects<std::invalid_argument>([&] { (void)bench::compute_errors(Matrix<float>(1, 2), ref); }, "shape check");
    require(bench::checksum(y) == 6.0, "checksum");

    // Unquantized FP32 relative error <= 1e-5 on fixed nonzero small fixtures.
    for (std::size_t k : {1U, 2U, 16U, 64U}) {
        const auto xr = random_matrix(8, k, 11);
        const auto wr = random_matrix(k, 33, 12);
        bench::Fp32Linear fp32;
        fp32.prepare(wr);
        const auto err = bench::compute_errors(fp32.run(xr), bench::reference_fp64(xr, wr));
        require(err.rel_frobenius <= 1e-5, "FP32 baseline relative error");
    }

    // Domain and lifecycle checks.
    bench::Fp32Linear fresh;
    rejects<std::logic_error>([&] { (void)fresh.run(x); }, "run before prepare");
    rejects<std::invalid_argument>([&] { fresh.prepare(Matrix<float>(3, 1)); }, "K not a power of two");
    rejects<std::invalid_argument>([&] { fresh.prepare(Matrix<float>(2048, 1)); }, "K > 1024");
    rejects<std::invalid_argument>([&] { fresh.prepare(Matrix<float>(2, 1025)); }, "N > 1024");
    rejects<std::invalid_argument>([&] { (void)op.run(Matrix<float>(33, 2)); }, "M > 32");
    rejects<std::invalid_argument>([&] { (void)op.run(Matrix<float>(1, 4)); }, "K mismatch");
    rejects<std::invalid_argument>([&] { (void)op.run(Matrix<float>(1, 2, {1, std::nanf("")})); }, "NaN X");
    rejects<std::invalid_argument>([&] { op.prepare(Matrix<float>(2, 1, {1, INFINITY})); }, "Inf W");
    require(op.run(x)(0, 0) == 7.0F, "failed prepare keeps previous weights");
}

// ---------------------------------------------------------------------------------
// Dense Sylvester Hadamard (unnormalized), used only as an offline reference.
std::vector<double> dense_hadamard(std::size_t k) {
    std::vector<double> h(k * k, 0);
    h[0] = 1;
    for (std::size_t size = 1; size < k; size *= 2) {
        for (std::size_t r = 0; r < size; ++r) {
            for (std::size_t c = 0; c < size; ++c) {
                const double v = h[r * k + c];
                h[r * k + c + size] = v;
                h[(r + size) * k + c] = v;
                h[(r + size) * k + c + size] = -v;
            }
        }
    }
    return h;
}

void test_rotation_against_dense_q() {
    for (std::size_t k : {1U, 2U, 4U, 8U, 16U, 32U, 64U}) {
        const bench::HadamardRotation rotation(k, 0);
        const auto h = dense_hadamard(k);
        // Q = D H / sqrt(K) with the SAME signs as the fast path.
        std::vector<double> q(k * k);
        for (std::size_t r = 0; r < k; ++r) {
            for (std::size_t c = 0; c < k; ++c) {
                q[r * k + c] = rotation.signs()[r] * h[r * k + c] / std::sqrt(static_cast<double>(k));
            }
        }
        for (std::size_t r = 0; r < k; ++r) {
            for (std::size_t c = 0; c < k; ++c) {
                double dot = 0;
                for (std::size_t i = 0; i < k; ++i) dot += q[i * k + r] * q[i * k + c];
                require(near(dot, r == c ? 1.0 : 0.0, 1e-12), "Q^T Q = I");
            }
        }
        const auto x = random_matrix(3, k, 21);
        const auto w = random_matrix(k, 5, 22);
        Matrix<float> xr = x;
        Matrix<float> wr = w;
        rotation.apply_right(xr);
        rotation.apply_transpose_left(wr);
        for (std::size_t m = 0; m < 3; ++m) {
            for (std::size_t c = 0; c < k; ++c) {
                double expected = 0;
                for (std::size_t i = 0; i < k; ++i) expected += static_cast<double>(x(m, i)) * q[i * k + c];
                require(near(xr(m, c), expected, 1e-5 * (1 + std::abs(expected))), "fast XQ matches dense XQ");
            }
        }
        for (std::size_t r = 0; r < k; ++r) {
            for (std::size_t n = 0; n < 5; ++n) {
                double expected = 0;
                for (std::size_t i = 0; i < k; ++i) expected += q[i * k + r] * static_cast<double>(w(i, n));
                require(near(wr(r, n), expected, 1e-5 * (1 + std::abs(expected))), "fast Q^T W matches dense");
            }
        }
        // Unquantized equivalence (XQ)(Q^T W) = XW within relative error 1e-5.
        Matrix<float> x_prime = xr;
        const auto rotated_product = bench::fp32_matmul(x_prime, wr);
        const auto err = bench::compute_errors(rotated_product, bench::reference_fp64(x, w));
        require(err.rel_frobenius <= 1e-5, "rotation equivalence");
    }
    const bench::HadamardRotation a(64, 0);
    const bench::HadamardRotation b(64, 0);
    const bench::HadamardRotation c(64, 1);
    require(a.signs() == b.signs() && a.signs() != c.signs(), "seeded signs are reproducible");
    std::size_t negatives = 0;
    for (auto s : a.signs()) {
        require(s == 1 || s == -1, "signs are +/-1");
        negatives += s < 0 ? 1 : 0;
    }
    require(negatives > 0 && negatives < 64, "both signs occur");
    require(a.storage_bytes() == 64, "one byte per sign");
    rejects<std::invalid_argument>([] { (void)bench::HadamardRotation(3, 0); }, "non power of two");
    rejects<std::invalid_argument>([] { (void)bench::HadamardRotation(0, 0); }, "zero size");
    Matrix<float> wrong(2, 32);
    rejects<std::invalid_argument>([&] { a.apply_right(wrong); }, "X columns != K");
    rejects<std::invalid_argument>([&] { a.apply_transpose_left(wrong); }, "W rows != K");
}

// ---------------------------------------------------------------------------------
template <typename Scale>
void test_w4a4_operator() {
    const int4::ScalarKernel<Scale> scalar;
    const int4::BlockedKernel<Scale> blocked;
    // Proposal example: Y = 7 within floating-point tolerance in every mode.
    const Matrix<float> x(1, 2, {1, -1});
    const Matrix<float> w(2, 1, {7, 0});
    for (auto mode : {bench::QuantMode::Direct, bench::QuantMode::Rotated}) {
        for (std::uint64_t seed : {0U, 1U, 2U, 3U}) {
            bench::W4A4Linear<Scale> op(mode, seed);
            op.prepare(w);
            const auto ys = op.run(x, scalar);
            const auto yb = op.run(x, blocked);
            require(std::abs(ys(0, 0) - 7.0F) < 1e-5F, "proposal example in " + std::string(bench::mode_name(mode)));
            require(bench::identical(ys, yb), "scalar and blocked agree inside the operator");
        }
    }
    // HANDOFF.md shared fixture through the direct operator.
    bench::W4A4Linear<Scale> direct(bench::QuantMode::Direct, 0);
    direct.prepare(Matrix<float>(2, 2, {7, 0, 0, 14}));
    const auto y = direct.run(Matrix<float>(1, 2, {7, -3.5F}), blocked);
    require(y(0, 0) == 49.0F && y(0, 1) == -56.0F, "HANDOFF fixture through the operator");
    require(direct.signs_bytes() == 0 && direct.rotation() == nullptr, "direct mode has no signs");

    // Lifecycle: run before prepare, strong guarantee on failed prepare, domain checks.
    bench::W4A4Linear<Scale> rotated(bench::QuantMode::Rotated, 0);
    rejects<std::logic_error>([&] { (void)rotated.run(x, blocked); }, "run before prepare");
    rotated.prepare(w);
    const auto bytes_before = rotated.packed_weights().bytes();
    const auto signs_before = rotated.rotation()->signs();
    rejects<std::invalid_argument>([&] { rotated.prepare(Matrix<float>(4, 1, {1, 2, std::nanf(""), 4})); },
                                   "nonfinite W");
    rejects<std::invalid_argument>([&] { rotated.prepare(Matrix<float>(6, 1)); }, "K not a power of two");
    require(rotated.packed_weights().bytes() == bytes_before && rotated.rotation()->signs() == signs_before &&
                rotated.rotation()->k() == 2,
            "failed prepare leaves prepared state unchanged");
    rejects<std::invalid_argument>([&] { (void)rotated.run(Matrix<float>(33, 2), blocked); }, "M > 32");
    rejects<std::invalid_argument>([&] { (void)rotated.run(Matrix<float>(1, 4), blocked); }, "K mismatch");
    require(rotated.signs_bytes() == 2, "rotated mode retains K sign bytes");

    // Proposal storage budget: K = N = 1024 packed W is 0.5 MiB plus one scale per column.
    const auto data = bench::generate_scale_data(1024, 1024, 0);
    bench::W4A4Linear<Scale> big(bench::QuantMode::Rotated, 0);
    big.prepare(data.w);
    require(big.packed_weights().bytes().size() == 524288, "packed W bytes at K = N = 1024");
    require(big.weight_bytes() == 524288 + 1024 * sizeof(Scale), "packed W + scales");
    require(big.signs_bytes() == 1024, "signs at K = 1024");
}

template <typename Scale>
void test_rotation_reduces_outlier_error() {
    // Regression expectation for the experiment's hypothesis on a fixed seed: with
    // a = 100 activation outliers, rotation lowers the W4A4 relative error.
    const auto data = bench::generate_scale_data(256, 256, 0);
    const auto x = bench::make_case_input(data, 8, 100);
    const auto y64 = bench::reference_fp64(x, data.w);
    const int4::BlockedKernel<Scale> kernel;
    bench::W4A4Linear<Scale> direct(bench::QuantMode::Direct, 0);
    bench::W4A4Linear<Scale> rotated(bench::QuantMode::Rotated, 0);
    direct.prepare(data.w);
    rotated.prepare(data.w);
    const double direct_error = bench::compute_errors(direct.run(x, kernel), y64).rel_frobenius;
    const double rotated_error = bench::compute_errors(rotated.run(x, kernel), y64).rel_frobenius;
    require(rotated_error < direct_error, "rotation reduces error with a = 100 outliers");
}

// ---------------------------------------------------------------------------------
void test_accounting() {
    const auto f = bench::fp32_bytes(8, 256, 128);
    require(f.weight == 131072 && f.input == 8192 && f.output == 4096 && f.prep_peak == 262144 &&
                f.call_peak == 131072 + 8192 + 4096 && f.resident() == 131072,
            "FP32 accounting");
    // M=8, K=256, N=128, rotated, float scales: packed W = 256*64 + 128*4, X = 8*128 + 8*4.
    const std::size_t weight = 256 * 64 + 128 * 4;
    const std::size_t act = 8 * 128 + 8 * 4;
    const auto r = bench::int4_bytes(8, 256, 128, true, weight, 256, act, 1280);
    require(r.sums == 4096 && r.output == 4096 && r.resident() == weight + 256, "INT4 components");
    require(r.prep_peak == 131072 + 131072 + 128 * 8 + weight + 256, "INT4 prepare peak");
    // Phases: quantize 8192 + 64 + act; kernel act + 4096 + 1280; rescale act + 8192.
    require(r.call_peak == weight + 256 + 8192 + (8192 + 64 + act), "INT4 call peak");
    const auto d = bench::int4_bytes(8, 256, 128, false, weight, 0, act, 1280);
    require(d.call_peak == weight + 8192 + (act + 4096 + 4096), "direct call peak uses rescale phase");
}

void test_dumps() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("int4_bench_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);
    const std::string path = (dir / "m.bin").string();
    const Matrix<float> m(2, 3, {1, -2.5F, 3, std::numeric_limits<float>::denorm_min(), -0.0F, 1e30F});
    bench::write_matrix_bin(path, m);
    require(fs::file_size(path) == 16 + 6 * 4, "dump size = 16-byte header + FP32 payload");
    const auto back = bench::read_matrix_bin(path);
    require(back.rows() == 2 && back.cols() == 3 && std::memcmp(back.data(), m.data(), 24) == 0, "dump round trip");
    {
        std::ofstream truncate(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t header[2] = {2, 3};
        truncate.write(reinterpret_cast<const char*>(header), sizeof(header));
    }
    rejects<std::runtime_error>([&] { (void)bench::read_matrix_bin(path); }, "truncated dump");
    rejects<std::runtime_error>([&] { (void)bench::read_matrix_bin((dir / "missing.bin").string()); }, "missing");
    fs::remove_all(dir);
}

} // namespace

int main() {
    try {
        test_stats();
        test_workload();
        test_metrics_and_fp32();
        test_rotation_against_dense_q();
        test_w4a4_operator<float>();
        test_w4a4_operator<double>();
        test_rotation_reduces_outlier_error<float>();
        test_rotation_reduces_outlier_error<double>();
        test_accounting();
        test_dumps();
        std::cout << "PASS: benchmark support: median/IQR, reproducible workloads, FP64 metrics, FP32 "
                     "baseline, rotation vs dense Q (K <= 64), operator lifecycle, accounting, dumps\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
