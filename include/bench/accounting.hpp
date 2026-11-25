#pragma once

// Member 3: logical-buffer storage accounting (proposal Sections 6.5 and 7.1).
//
// Counts resident production matrices, packed bytes, scales, signs, and scratch.
// Like storage_bytes(), it excludes vector capacity, allocator and object overhead,
// offline references (Y64), and process overhead. Peaks follow the lifetimes of the
// buffers inside prepare() and run(); see docs/benchmark.md.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace bench {

struct Bytes {
    std::size_t weight = 0;    // Retained prepared weights (FP32 copy, or packed codes + scales).
    std::size_t signs = 0;     // Retained rotation signs.
    std::size_t input = 0;     // Caller-owned FP32 X.
    std::size_t act = 0;       // Packed X codes + scales.
    std::size_t sums = 0;      // INT32 accumulators.
    std::size_t output = 0;    // FP32 Y.
    std::size_t scratch = 0;   // Kernel scratch.
    std::size_t prep_peak = 0; // Peak inside prepare(), including the caller's FP32 W.
    std::size_t call_peak = 0; // Peak inside run(), including resident state, X and Y.
    std::size_t resident() const noexcept { return weight + signs; }
};

inline Bytes fp32_bytes(std::size_t m, std::size_t k, std::size_t n) {
    Bytes bytes;
    bytes.weight = k * n * sizeof(float);
    bytes.input = m * k * sizeof(float);
    bytes.output = m * n * sizeof(float);
    bytes.prep_peak = 2 * k * n * sizeof(float); // Caller W + retained copy.
    bytes.call_peak = bytes.weight + bytes.input + bytes.output;
    return bytes;
}

// weight/signs/act/scratch come from the live objects (storage_bytes(), scratch_bytes()).
inline Bytes int4_bytes(std::size_t m, std::size_t k, std::size_t n, bool rotated, std::size_t weight,
                        std::size_t signs, std::size_t act, std::size_t scratch) {
    Bytes bytes;
    bytes.weight = weight;
    bytes.signs = signs;
    bytes.input = m * k * sizeof(float);
    bytes.act = act;
    bytes.sums = m * n * sizeof(std::int32_t);
    bytes.output = m * n * sizeof(float);
    bytes.scratch = scratch;
    // prepare(): caller W + rotated copy W' + quantizer maxima (double) + packed W + signs.
    bytes.prep_peak = k * n * sizeof(float) + (rotated ? k * n * sizeof(float) : 0) + n * sizeof(double) +
                      weight + signs;
    // run(): quantize phase (X' copy + maxima + qx), kernel phase (qx + sums + scratch),
    // rescale phase (qx + sums + Y); the largest phase on top of resident state and X.
    const std::size_t quantize_phase = (rotated ? m * k * sizeof(float) : 0) + m * sizeof(double) + act;
    const std::size_t kernel_phase = act + bytes.sums + scratch;
    const std::size_t rescale_phase = act + bytes.sums + bytes.output;
    bytes.call_peak = bytes.resident() + bytes.input + std::max({quantize_phase, kernel_phase, rescale_phase});
    return bytes;
}

} // namespace bench
