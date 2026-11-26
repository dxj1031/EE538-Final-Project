#include "int4/io.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void rejects(const std::string& text, int4::MatrixFileKind kind) {
    std::istringstream input(text);
    try { (void)int4::read_matrix_text(input, kind); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("accepted invalid matrix: " + text);
}
} // namespace

int main() {
    try {
        using Kind = int4::MatrixFileKind;
        const int4::Matrix<float> original(1, 4, {1.25F, -0.0F,
            std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()});
        std::stringstream text;
        int4::write_matrix_text(text, original);
        const auto back = int4::read_matrix_text(text, Kind::Input);
        require(back.rows() == 1 && back.cols() == 4 &&
            std::memcmp(back.data(), original.data(), original.storage_bytes()) == 0,
            "FP32 text round trip preserves values including subnormal and signed zero");
        std::istringstream whitespace(" 1\t2\n +1e0\r\n-1 \n");
        const auto valid = int4::read_matrix_text(whitespace, Kind::Input);
        require(valid(0, 0) == 1 && valid(0, 1) == -1, "whitespace and scientific notation");
        for (const std::string bad : {"", "1", "0 2", "-1 2", "+1 2", "1.0 2", "33 2",
             "1 3", "1 2048", "18446744073709551616 2", "1 2 7", "1 2 7 0 extra",
             "1 2 nan 0", "1 2 inf 0", "1 2 1e100 0", "1 2 1junk 0", "1 2 +-1 0"}) {
            rejects(bad, Kind::Input);
        }
        for (const std::string bad : {"3 1 0 0 0", "2048 1", "2 1025", "2 0", "1 1 -inf"}) {
            rejects(bad, Kind::Weight);
        }
        require(int4::parse_unsigned("18446744073709551615", "seed") ==
                std::numeric_limits<std::uint64_t>::max(), "full uint64 seed range");
        std::ostringstream failed;
        failed.setstate(std::ios::badbit);
        bool write_rejected = false;
        try { int4::write_matrix_text(failed, original); }
        catch (const std::runtime_error&) { write_rejected = true; }
        require(write_rejected, "write failure is reported");
        std::cout << "PASS: strict matrix parsing, FP32 round trip and stream errors\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
