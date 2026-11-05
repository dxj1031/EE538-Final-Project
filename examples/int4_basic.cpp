#include "int4/core.hpp"

#include <iomanip>
#include <iostream>

template <typename Scale>
void example(const char* label) {
    // Per-row X scales, per-column W scales; no transpose at the API boundary.
    const int4::Matrix<float> x(2, 3, {7, -3.5F, 1, 0.3F, 0.2F, -0.1F});
    const int4::Matrix<float> w(3, 2, {0.2F, -0.3F, 0.4F, 0.1F, -0.1F, 0.2F});
    const auto qx = int4::quantize_int4<Scale>(x, int4::ScaleAxis::Row);
    const auto qw = int4::quantize_int4<Scale>(w, int4::ScaleAxis::Column);
    const int4::ScalarKernel<Scale> scalar;
    const int4::MatmulKernel<Scale>& kernel = scalar;
    const auto sums = kernel.run(qx, qw);
    const auto y = int4::rescale_output(sums, qx.scales(), qw.scales());
    std::cout << label << " scales; logical packed+scale bytes: "
              << qx.storage_bytes() + qw.storage_bytes() << '\n';
    for (std::size_t m = 0; m < y.rows(); ++m) {
        for (std::size_t n = 0; n < y.cols(); ++n) std::cout << y(m, n) << ' ';
        std::cout << '\n';
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    example<float>("float");
    example<double>("double");
}
