# Hadamard-Rotated INT4 Linear Operator on CPUs

C++17 matrix storage, symmetric INT4 quantization, and a scalar integer matrix-multiplication kernel for `Y = XW` on CPUs. Both `float` and `double` scale storage are supported by the same implementation.

The current project uses **W4A4**: both weights and activations are stored as packed 4-bit codes. The current public entry point is `include/int4/core.hpp`; an earlier INT8 baseline is retained separately.

## Documentation

- [Team handoff and interface contract](HANDOFF.md): integration instructions for Members 2 and 3.
- [Blocked kernel and benchmark driver](docs/benchmark.md) (Wooyoung Kim): kernel design, measurement protocol, CSV schema, and reproduction commands.
- [Wooyoung Kim's report section draft](docs/report_wooyoung_kim.md): kernel design and preliminary results.
- [Prepared operator and CLI](docs/cli.md) (Qinyan Zeng): API, input files, options, timing and tests.

## Implemented scope

- Owning, contiguous row-major `Matrix<T>` with RAII, checked shapes, deep copies, and moves.
- `PackedInt4Matrix<Scale>` with private byte/scale buffers and checked metadata.
- Per-row activation scales and per-column weight scales.
- Symmetric codes in `[-7, 7]`, packed two per byte with signed decoding.
- `ScalarKernel<Scale>` implementing the shared `MatmulKernel<Scale>` interface.
- INT32 accumulation followed by shared FP32 output reconstruction.
- Tests using known encodings, independent INT64 accumulation, and FP64 output references.

Wooyoung Kim added `BlockedKernel<Scale>` (`include/int4/blocked_kernel.hpp`), the scalar FP32 baseline, and the `int4_bench` driver with its NumPy reference check; see [docs/benchmark.md](docs/benchmark.md). Qinyan Zeng added the production Hadamard rotation, the three-mode `QuantizedLinear<Scale>` operator, and the file-to-file `int4_cli`. The benchmark now uses these production classes; the temporary rotation and operator have been removed. The example is a correctness demonstration, not a benchmark. Scalar arithmetic decodes codes to ordinary integer operands; this implementation does not use a hardware INT4 instruction.

## Build and run

Run the following from this directory. A C++17 compiler is the only requirement for the header-only INT4 module.

### Windows / PowerShell with GCC

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -I include tests/test_int4.cpp -o build/int4_tests.exe
if ($LASTEXITCODE -ne 0) { throw "INT4 test build failed" }
.\build\int4_tests.exe
if ($LASTEXITCODE -ne 0) { throw "INT4 tests failed" }

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -I include examples/int4_basic.cpp -o build/int4_example.exe
if ($LASTEXITCODE -ne 0) { throw "INT4 example build failed" }
.\build\int4_example.exe
```

### CMake / CTest

```text
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release
ctest --test-dir build/cmake -C Release --output-on-failure -R int4_core_tests
```

CMake exposes the interface library `int4_core` and executables `int4_example`, `int4_tests` and `int4_cli`. Select a generator matching the local compiler on Windows. CMake/CTest commands were not validated in Member 1's environment because CMake was unavailable there.

Wooyoung Kim's targets: `int4_blocked_tests`, `int4_bench_tests`, and the driver `int4_bench`. Run the whole suite with `ctest --test-dir build/cmake -C Release --output-on-failure`. With Python and NumPy it has 52 tests: the original seven suites/checks, production operator and parser suites, 24 CLI fixture combinations (each with four seeds), and 19 CLI rejection cases. The numerical CLI checks require only the Python standard library; the independent benchmark reference requires NumPy. Tests that need these packages are registered when they are available. Single-config generators now default to `Release`. `-DINT4_SANITIZE=ON` adds AddressSanitizer and UndefinedBehaviorSanitizer on GCC/Clang. The complete benchmark pipeline is `scripts/run_benchmark.sh`.

## File-to-file CLI

```text
build/release/int4_cli --x tests/data/proposal_x.txt --w tests/data/proposal_w.txt --out results/example/Y.txt --csv results/example/timings.csv --mode rotated --kernel blocked --seed 0 --scale-precision float
```

Use the `.exe` path on Windows, adding `Release/` for multi-config builds. The
proposal example returns approximately 7. Inputs are dimensions followed by
row-major FP32 values. See [docs/cli.md](docs/cli.md) for all three modes, both
kernels, float/double scales, strict input rules and timing/storage columns.

## Minimal usage

```cpp
#include "int4/core.hpp"
#include <iostream>

int main() {
    using Scale = float; // Change to double to use FP64 scale storage.
    const int4::Matrix<float> x(1, 2, {1, -1});
    const int4::Matrix<float> w(2, 1, {7, 0});
    const auto qx = int4::quantize_int4<Scale>(x, int4::ScaleAxis::Row);
    const auto qw = int4::quantize_int4<Scale>(w, int4::ScaleAxis::Column);
    const int4::ScalarKernel<Scale> kernel;
    const auto sums = kernel.run(qx, qw);
    const auto y = int4::rescale_output(sums, qx.scales(), qw.scales());
    std::cout << y(0, 0) << '\n'; // Approximately 7.
}
```

Weights use shape `[K,N]`, not `[N,K]`. No transpose or bias is part of this interface. A prepared operator can retain `qw` across calls when the prepared weights and configuration are unchanged.

## Numerical contract

| Item | Contract |
| --- | --- |
| Shapes | X: `[M,K]`; W: `[K,N]`; integer sums and Y: `[M,N]` |
| Input/output | FP32 matrices; positive dimensions |
| Activation scales | One per row of X |
| Weight scales | One per column of W |
| Stored scale | `Scale(max_abs / 7.0)`; all-zero groups use 1 |
| Code generation | Divide by the stored scale in double; round ties away from zero; clamp to `[-7,7]` |
| Packing | Row-aligned; low nibble holds even columns, high nibble holds odd columns |
| Signed codes | Four-bit two's complement; nibble `0x8` is forbidden |
| Odd row length | The unused high nibble is zero |
| Accumulation | INT32; require `49*K <= INT32_MAX` |
| Reconstruction | Compute `sum[m,n] * sx[m] * sw[n]` in double, then cast to FP32 |
| Errors | Standard exceptions for invalid shapes, metadata, codes, nonfinite data, and unsafe ranges |

`float` and `double` refer to scale storage, not the input/output or accumulator type. Encoding uses the stored scale, so the two variants can produce different codes near rounding boundaries. For `[1,0.5]`, the second code is 3 with float scales and 4 with double scales under the current contract. A float scale that underflows to zero is rejected explicitly; the double variant supports those smaller FP32 inputs.

The core accepts safe positive dimensions beyond the proposal's CLI domain. The outer layer must enforce `1 <= M <= 32`, `1 <= N <= 1024`, and power-of-two `K <= 1024` if adopting that domain.

## Source map

| Path | Purpose |
| --- | --- |
| `include/int4/core.hpp` | Current packed format, quantizer, kernel interface, scalar implementation, and reconstruction |
| `include/int8/matrix.hpp` | Shared owning `Matrix<T>` implementation, aliased into namespace `int4` |
| `examples/int4_basic.cpp` | Both scale variants through the virtual kernel interface |
| `tests/test_int4.cpp` | INT4 encoding, arithmetic, reference, and rejection tests |
| `CMakeLists.txt` | C++17 build targets and CTest integration |
| `include/int4/hadamard.hpp`, `include/int4/linear.hpp` | Qinyan Zeng: implicit rotation and prepared three-mode operator |
| `include/int4/io.hpp`, `src/int4_cli.cpp` | Qinyan Zeng: strict text matrix I/O and file-to-file CLI |
| `tests/test_linear.cpp`, `tests/test_io.cpp`, `tests/test_cli.py` | Qinyan Zeng: operator/parser and CLI end-to-end tests |
| `include/int4/blocked_kernel.hpp` | Wooyoung Kim: blocked INT4 kernel (32-by-32 tiles decoded into reusable INT8 scratch) |
| `include/bench/*.hpp` | Wooyoung Kim: timing statistics, workloads, metrics, independent FP32 baseline, accounting, dumps |
| `bench/int4_bench.cpp` | Wooyoung Kim: benchmark driver (CSV and metadata output) |
| `tests/test_blocked_kernel.cpp`, `tests/test_bench.cpp` | Wooyoung Kim: blocked-kernel and benchmark-support tests |
| `scripts/` | Wooyoung Kim: `run_benchmark.sh`, `reference_check.py` (NumPy FP64), `plot_results.py` |
| `results/` | Raw benchmark CSVs, metadata, plots, and summaries (`dumps/` is git-ignored) |

The earlier INT8 baseline remains under `include/int8`, `src`, `examples/basic.cpp`, and `tests/test_core.cpp`. Its weight orientation and quantization rules differ; use `int4/core.hpp` and `HANDOFF.md` for the current project.

## Validation status

Verified on 2026-11-04 with Windows / MSYS2 UCRT64 GCC 16.1.0:

- Release: `-O2`.
- Debug: `-O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS`.
- Both builds passed with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
- Each scale type covers all 225 adjacent code pairs and 225 one-term signed dot products, plus scale axes, padding, `N=33`, same-code INT64/FP64 references, malformed input, and accumulator-bound checks.
- The example ran with both scale types.

The INT4 maximum-K guard is checked analytically; the test does not allocate and multiply at that enormous limit. No performance sweep, sanitizer run, rotation test, or full team integration is claimed.

## Complexity and memory accounting

Quantization takes `O(rows*cols)` time and owns the packed output plus per-group scales; it also uses temporary maxima. Scalar multiplication takes `O(M*K*N)` arithmetic work plus input validation. Reconstruction takes `O(M*N)` time.

Packed storage is `rows * ceil(cols/2)` bytes, plus `scale_count * sizeof(Scale)`. For a 1024-by-1024 weight matrix, codes occupy 524,288 bytes; column scales occupy 4,096 bytes with float or 8,192 bytes with double on the validated platform.

`storage_bytes()` reports logical buffer contents, excluding capacity, allocator overhead, and object metadata. The integer output and reconstructed FP32 output can coexist, adding `8*M*N` logical bytes on the validated platform. Retained FP32 inputs, temporary rotation data, packed operands, and scratch must be accounted for separately. The implementation makes no claim of a particular process-memory reduction or speedup.
