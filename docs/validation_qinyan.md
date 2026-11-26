# Member 2 validation record

Date: 2026-11-25. Branch: `qinyan/rotation-operator-cli`, based on
`origin/wooyoung-kim/blocked-kernel-benchmark` (PR #2, commit `38ce9e5`).
Member 1's `include/int4/core.hpp` and packed format are unchanged.

## Executed checks

| Environment | Configuration | Observed result |
| --- | --- | --- |
| Windows x64, LLVM-MinGW UCRT Clang 23.1.3, CMake 4.4.5/Ninja | Release `-O3 -DNDEBUG`, requested warnings plus `-Werror` | 54/54 CTest tests pass |
| Same Windows toolchain | Debug `-g`, `INT4_SANITIZE=ON` (ASan+UBSan), warnings plus `-Werror` | 54/54 CTest tests pass |
| Ubuntu 22.04 / WSL2, GCC 11.4.0 | `-O2 -DNDEBUG`, requested warnings plus `-Werror` | Six C++ test executables, benchmark smoke and 24 CLI fixture combinations pass |
| Same GCC toolchain | `-O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS`, warnings plus `-Werror` | INT4 core, blocked, benchmark-support, linear and I/O C++ suites pass |
| Windows/Git Bash | Full shell pipeline, float scales | 45 cases; 165/165 NumPy metric checks, 60/60 direct same-code matches; zero kernel mismatches |
| Windows/Git Bash | Full shell pipeline, double scales | 45 cases; 165/165 NumPy metric checks, 60/60 direct same-code matches; zero kernel mismatches |
| Windows/PowerShell | `run_benchmark.ps1 -Quick -ScalePrecision double` | Build, 53-test suite at that stage, smoke sweep, reference and all plots pass |

Warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
MSVC was not available and is not claimed as tested. GCC validation is Linux
under WSL2, not a claim of a native MSYS2 GCC build.

## Reference coverage

- Existing dense Sylvester Q tests now target production classes, including
  XQ, Q^T W, Q^T Q=I and unquantized relative product error <=1e-5 for K<=64.
- New rotation tests exercise all power-of-two K through 1024 and uint64 seed
  boundaries, intermediate/result overflow, nonfinite values and wrong shapes.
- Operator tests cover all modes, kernel dispatch, owning/moved state, changed
  dimensions, K=1/N=33, and preservation after validation, rotation or quantizer
  failure. FP32 output overflow is rejected.
- Strict text I/O tests include exact FP32 round trips at FLT_MAX, subnormals
  and signed zero; scientific notation; invalid/huge headers; short/extra values;
  malformed/nonfinite/out-of-range tokens and failed output streams.
- The 24 supplied-fixture CLI combinations each test seeds 0,1,2,UINT64_MAX,
  numeric Y, timing CSV statistics/storage and Windows case-alias rejection.
- NumPy CLI reference: 72 additional outputs at M=3, N=33, K=1/2/64, seeds
  0/7/UINT64_MAX, both quantized modes/kernels/precisions. Python independently
  builds the MT19937-64 sign stream and dense Q, quantizes and accumulates INT64.
- 19 `WILL_FAIL` CLI tests cover malformed/nonfinite/truncated/trailing input,
  invalid domains/options/seeds, missing files, zero runs and input/output aliases.
- Median aggregation tests reject missing pairs/seeds, duplicate primary rows,
  unverified data and nonfinite metrics; other kernels/tiles cannot bias medians.

## Reproduction

Use a compiler/CMake/Python environment on PATH. Python 3 with NumPy and
Matplotlib is needed for complete independent checks and figures.

```text
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-Werror -DPython3_EXECUTABLE=<python-path>
cmake --build build/release --parallel 4
ctest --test-dir build/release -C Release --output-on-failure

cmake -S . -B build/debug-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-Werror -DINT4_SANITIZE=ON -DPython3_EXECUTABLE=<python-path>
cmake --build build/debug-sanitize --parallel 4
ctest --test-dir build/debug-sanitize -C Debug --output-on-failure

scripts/run_benchmark.sh results/<date>_<machine>-float --scale-precision float
scripts/run_benchmark.sh results/<date>_<machine>-double --scale-precision double
```

For direct GCC validation, compile `tests/test_int4.cpp`,
`tests/test_blocked_kernel.cpp`, `tests/test_bench.cpp`, `tests/test_linear.cpp`
and `tests/test_io.cpp` separately with `-std=c++17 -I include` and the flags
above. `tests/test_core.cpp` additionally links `src/quantization.cpp` and
`src/scalar_kernel.cpp`. Compile CLI/driver from `src/int4_cli.cpp` and
`bench/int4_bench.cpp`, run the driver with `--quick`, and run
`tests/test_cli.py --exe <absolute-cli-path> --fixture proposal|handoff --mode
fp32|direct|rotated --kernel scalar|blocked --precision float|double` for all
24 combinations. Test flags are shown literally; replace the vertical-bar choices
with one value per invocation.

Per-run compiler flags, versions, hardware and seeds are retained beside the
result CSVs. Figures were also visually inspected. These checks support the
tested implementation and inputs, without asserting that all possible bugs or
platform differences have been excluded.
