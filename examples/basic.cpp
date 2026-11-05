#include "int8/scalar_kernel.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const int8::Matrix<float> x(2, 3, {1.0F, -0.5F, 0.25F, 0.3F, 0.2F, -0.1F});
    const int8::Matrix<float> w(2, 3, {0.2F, 0.4F, -0.1F, -0.3F, 0.1F, 0.2F});
    const auto qx = int8::quantize(x, int8::ScaleMode::PerTensor);
    const auto qw = int8::quantize(w, int8::ScaleMode::PerRow);
    const auto y = int8::scalar_linear(qx, qw, {0.1F, -0.2F});
    std::cout << "Quantized output (FP32):\n" << std::fixed << std::setprecision(6);
    for (std::size_t m = 0; m < y.rows(); ++m) {
        for (std::size_t n = 0; n < y.cols(); ++n) std::cout << y(m, n) << ' ';
        std::cout << '\n';
    }
    std::cout << "Original unquantized output (approximately):\n"
                 "0.075000 -0.500000\n0.250000 -0.290000\n";
}
