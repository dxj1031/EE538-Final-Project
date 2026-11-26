# Member 1 Handoff: INT4 Storage, Quantization, and Scalar Kernel

Status: implemented and locally tested; interface contract v1 is proposed for team adoption. This document is the integration reference for Members 2 and 3. Source declarations in `include/int4/core.hpp` are authoritative if a documentation discrepancy is found.

## Deliverables and boundaries

| Component | Status / owner |
| --- | --- |
| Owning `Matrix<T>` | Implemented by Member 1; shared storage template |
| `PackedInt4Matrix<Scale>` | Implemented by Member 1 |
| `quantize_int4<Scale>` | Implemented by Member 1; float/double scales |
| `MatmulKernel<Scale>` / `ScalarKernel<Scale>` | Interface / scalar implementation provided by Member 1 |
| `rescale_output` | Implemented by Member 1; shared by all integer kernels |
| Hadamard rotation and prepared operator | Implemented by Qinyan Zeng in `include/int4/hadamard.hpp` and `include/int4/linear.hpp` |
| File-to-file CLI | Implemented by Qinyan Zeng in `src/int4_cli.cpp`; see [docs/cli.md](docs/cli.md) |
| Blocked kernel and C++ benchmark driver | Implemented by Wooyoung Kim: `include/int4/blocked_kernel.hpp`, `bench/int4_bench.cpp`; see [docs/benchmark.md](docs/benchmark.md) |
| Independent tests | Each member tests their module; the current core has INT64/FP64 references |
| Scalar FP32 baseline | Implemented by Wooyoung Kim as `bench::fp32_matmul` / `bench::Fp32Linear` (`include/bench/linear.hpp`) |

Include `int4/core.hpp`. The INT4 implementation is header-only and reuses `include/int8/matrix.hpp`; copy both headers if importing the core without the rest of this directory. With CMake, link the `int4_core` interface target. Do not substitute the older INT8 quantizer or its `[N,K]` weight convention.

## 1. Shared data contract

```text
X: Matrix<float> [M,K], row-major
W: Matrix<float> [K,N], row-major
qx: PackedInt4Matrix<Scale> [M,K], ScaleAxis::Row
qw: PackedInt4Matrix<Scale> [K,N], ScaleAxis::Column
sums: Matrix<int32_t> [M,N]
Y: Matrix<float> [M,N]
```

The operation is `Y = XW`, without a bias term. `Scale` is either `float` or `double`, defaulting to float. Both packed operands in one call use the same scale type. FP32 inputs/outputs and INT32 accumulators are unchanged between variants.

For each activation row / weight column:

1. Find the maximum absolute input value in double.
2. Store `Scale(max_abs / 7.0)`, or `Scale(1)` for an all-zero group.
3. Convert that stored scale to double when evaluating `value / scale`.
4. Apply `std::round` (ties away from zero) and clamp to `[-7,7]`.

NaN and infinite input values are rejected. If a float scale rounds down to zero, quantization throws `std::underflow_error`; do not silently replace it with 1. Float/double variants are allowed to produce different codes at rounding boundaries. Compare each variant against references using its actual stored scales.

## 2. Packed byte contract

Every logical matrix remains row-major, regardless of its scale axis. Each row begins at a byte boundary:

```text
row_bytes = cols / 2 + cols % 2
byte_index(row,col) = row * row_bytes + col / 2
shift(col) = 4 * (col % 2)
```

Even columns occupy low nibbles; odd columns occupy high nibbles. Signed codes use four-bit two's complement: `-1 -> 0xF`, `-7 -> 0x9`. Nibble `0x8` is invalid because -8 is outside the chosen symmetric range. An odd-length row has a zero unused high nibble. Rows are never joined to share their final byte.

Examples: `[1,-1,7] -> [0xF1,0x07]`; `[7,-4] -> [0xC7]`.

For weights, `row_bytes` is derived from **N**, not K. Column-wise scales do not imply a transposed byte layout. A blocked implementation may use its own temporary tile layout, while the public input stays unchanged.

## 3. Public interfaces

The following is a declaration summary, not a separate header to include:

```cpp
template <typename T>
using Matrix = int8::Matrix<T>; // Exposed in namespace int4.

enum class ScaleAxis { Row, Column };

template <typename Scale = float>
class PackedInt4Matrix;

template <typename Scale = float>
PackedInt4Matrix<Scale> quantize_int4(const Matrix<float>& input, ScaleAxis axis);

template <typename Scale = float>
class MatmulKernel {
public:
    virtual ~MatmulKernel() = default;
    virtual Matrix<std::int32_t> run(const PackedInt4Matrix<Scale>& x,
                                     const PackedInt4Matrix<Scale>& w) const = 0;
};

template <typename Scale = float>
class ScalarKernel; // Implements MatmulKernel<Scale>.

template <typename Scale>
Matrix<float> rescale_output(const Matrix<std::int32_t>& sums,
                             const std::vector<Scale>& x_scales,
                             const std::vector<Scale>& w_scales);
```

`PackedInt4Matrix` provides `rows()`, `cols()`, `row_bytes()`, `scale_axis()`, `bytes()`, `scales()`, `code(row,col)`, `storage_bytes()`, and `validate()`. Its constructor accepts `(rows, cols, axis, bytes, scales)` and validates the representation. Byte and scale accessors return const references.

`Matrix<T>` owns its contiguous storage, supports independent copies, and empties the source on move. Matrix dimensions and element indices are checked. `data()` / `row_data()` expose non-owning pointers; their lifetime must not exceed their owner, and assignment or move may invalidate them. Packed objects also own their buffers; only destroy or reassign a moved-from packed object.

`run()` leaves inputs unchanged and returns a separately owned integer matrix. The scalar implementation validates packed operands, requires row-scaled X and column-scaled W with matching K, and enforces `49*K <= INT32_MAX`. The outer operator checks the proposal domain: `1 <= M <= 32`, `1 <= N <= 1024`, power-of-two K, and `1 <= K <= 1024`.

Reconstruction computes `double(sums[m,n]) * double(sx[m]) * double(sw[n])` and casts the result to FP32. Nonfinite or FP32-out-of-range reconstructed values are rejected. Do not cast `sums` to float before multiplication or rescale partial K blocks separately.

## 4. Member 2: prepared operator, rotation, and CLI

Production implementation: `int4::QuantizedLinear<Scale>` in `include/int4/linear.hpp`,
configured with `(LinearMode, rotation_seed)`, followed by `prepare(W)`. The lifecycle is:

```text
prepare(W, mode, rotation_seed)
    validate operator-domain dimensions and mode
    direct: use W
    rotated: compute Q^T W with the selected signs/normalized transform
    quantize prepared W with ScaleAxis::Column
    retain qw, dimensions, mode, and the rotation state

run(X, kernel)
    validate M/K and consistency with prepared weights
    direct: use X
    rotated: compute XQ using the SAME Q as prepare
    quantize the prepared X with ScaleAxis::Row
    sums = kernel.run(qx, qw)
    Y = rescale_output(sums, qx.scales(), qw.scales())
```

The implementation follows this lifecycle and also provides an independent FP32 mode.
Failed preparation preserves the complete previous state through an owning-pointer exchange.
The implicit rotation uses O(K) double workspace and checks its FP32 outputs; no dense Q is retained. Retain prepared weights only while W and the relevant configuration stay unchanged. Rebuild prepared state on changes to W, mode, rotation signs/seed, dimensions, or scale precision. Define failed-prepare behavior in your operator; constructing new state before replacing old state avoids partially updated weights.

If exposing `--scale-precision float|double`, dispatch once at the outer layer to templated code. Do not add per-element runtime precision checks. Keep the FP32 baseline independent of the INT4 path. Catch exceptions in the CLI and return a diagnostic plus nonzero status.

Before testing rotated quantized output, independently check the unquantized identity `(XQ)(Q^T W) = XW` with a small dense-Q reference and shared signs. This isolates rotation errors from quantization errors. Rotation is outside the current core test coverage.

## 5. Member 3: blocked kernel and measurements

Implement `BlockedKernel<Scale>` as a subclass of `MatmulKernel<Scale>`, with the exact `run` signature. Decode 32-by-32 tiles into your INT8 scratch buffers if following the proposal. Keep real dimensions separate from padded storage; cover K=1, N=33, odd row lengths, and partial tiles. Validate shape, codes/scales/padding, and the accumulation bound consistently with the scalar path.

First compare scalar and blocked **INT32 matrices element by element on the same qx/qw**. Only after those match, call shared reconstruction and compare FP32 results. A blocked kernel must not choose new scales, requantize inputs, or add its own floating output convention.

For experiments:

- Compare kernels on identical stored bytes/scales, precision, thread count, and source data.
- Compare float/double scale variants by separately quantizing identical raw inputs. Record `scale_precision`; different codes are legitimate.
- Time weight preparation separately. Input-call time includes online rotation when applicable, quantization/packing, kernel execution, reconstruction, and allocations.
- Record kernel-only timing separately. Scalar `run` currently includes validation scans and INT32 output allocation; it is not pure arithmetic timing.
- Count packed bytes, scales, signs, retained matrices, output, and scratch at the relevant lifetime peaks. Label logical-buffer accounting versus measured process memory.
- `storage_bytes()` excludes vector capacity, object/allocator overhead, and external matrices. A 1024-by-1024 W uses 524,288 code bytes plus 4,096 float-scale bytes or 8,192 double-scale bytes on the validated platform.

No scratch-reuse policy or blocked-kernel allocation API is prescribed yet. The return-by-value interface exposes a complete INT32 result; any future output-buffer interface requires a coordinated contract revision.

**Wooyoung Kim status (2026-11-21).** The items above are implemented:

- `int4::BlockedKernel<Scale>(tile = 32)` keeps the exact `run` signature and the scalar acceptance rules. It decodes each X tile once and each W tile `ceil(M/T)` times into INT8 scratch that is allocated per call and reused across tiles. `scratch_bytes(M)` reports `(min(M,T) + T) * T`. No allocation API or output-buffer change was needed, so the v1 contract is unchanged.
- The driver runs the scalar kernel on the same `qx`/`qw` for every case. It requires bitwise-identical INT32 sums and reconstructed Y from every blocked tile edge before it reports errors, and it records kernel-only, validation-only, and harness timings separately from the full input call.
- The benchmark needs a prepared operator and a rotation for the rotated mode. Until Member 2's `QuantizedLinear` is merged, it uses `bench::W4A4Linear` and `bench::HadamardRotation` (`include/bench/linear.hpp`). These follow Section 4 above and are tested against a dense Q for K ≤ 64. Replace them in `bench/int4_bench.cpp` when Member 2's operator lands; docs/benchmark.md lists the methods the driver uses.

## 6. Shared integration fixture

Use this exactly representable fixture before timing:

```text
X = [[7, -3.5]]
W = [[7,  0],
     [0, 14]]

qx codes = [[7, -4]]       sx = [1]
qw codes = [[7, 0],        sw = [1, 2]
            [0, 7]]
qx bytes = [0xC7]
qw bytes = [0x07, 0x70]
integer sums = [[49, -28]]
reconstructed Y = [[49, -56]]
original FP64 XW = [[49, -49]]
```

Both scale types give the same results for this fixture. The difference from original XW is expected quantization error, not a kernel failure.

Adoption steps: review this contract together, run `int4_tests`, reproduce this fixture through each implemented path, then mark v1 as accepted in the team repository. Changes to shape orientation, encoding, scale rules, return types, or ownership must update this document, the public declarations, and the tests together.

## 7. Build and verification

**Member 2 integration (2026-11-25).** The production operator and file-to-file
CLI are implemented. Proposal and Section 6 fixtures pass through both kernels
in direct and rotated modes, with both scale types (the rotated fixture has its
own expected quantization error; see docs/cli.md). Windows Clang Release and
ASan/UBSan Debug pass all 54 CTest tests; additional GCC/WSL Release and checked
Debug suites pass. Both full precision sweeps have zero kernel mismatches and
330/330 independent NumPy metric agreements in total. See
[docs/validation_qinyan.md](docs/validation_qinyan.md) and
[docs/report_qinyan_zeng.md](docs/report_qinyan_zeng.md).
This records Member 2's integration checks; team-wide contract acceptance remains
subject to the adoption steps in Section 6.

The following are the original Member 1 verification notes and Member 3's
2026-11-21 addendum:

See [README.md](README.md) for exact build commands and environment details. The tested example is `examples/int4_basic.cpp`; the current test entry point is `tests/test_int4.cpp`. Float and double variants passed Release and Debug builds with warnings treated as errors. CMake/CTest, a supported sanitizer, rotation, the blocked path, and the full benchmark sweep remain unverified here.

Wooyoung Kim addendum (2026-11-21, macOS arm64, Apple clang 21, CMake 4.4): the full CTest suite of 7 tests, including the INT4 core tests above, passes in Release (`-O3 -DNDEBUG`). It also passes in a Debug build with `-DINT4_SANITIZE=ON`, which enables AddressSanitizer and UndefinedBehaviorSanitizer, together with libc++ debug hardening. The blocked path and the full 45-case sweep were run; see `results/2026-11-21_apple-m4/`. GCC and MSVC builds of the new files were not available in that environment and remain unverified.

Core testing includes exhaustive 225 encoding pairs and 225 one-term dot products per scale type, row/column scaling, ties, constants/zeros, odd padding, N=33, malformed representations, independent INT64/FP64 references, and analytic maximum-K guard checks. Passing these tests does not establish full-model accuracy or a speedup.
