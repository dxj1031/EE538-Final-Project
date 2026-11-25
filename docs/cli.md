# Prepared operator and file-to-file CLI

Owner: Qinyan Zeng (Member 2). The production rotation adapts Wooyoung Kim's
temporary benchmark implementation, retaining its seeded-sign convention while
adding a bounded domain, double butterfly workspace and checked FP32 results.
`include/int4/core.hpp` and its packed v1 contract are unchanged.

## Build and example

```text
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
build/release/int4_cli --x tests/data/proposal_x.txt --w tests/data/proposal_w.txt --out results/example/Y.txt --csv results/example/timings.csv --mode rotated --kernel blocked --seed 0
```

On Windows use `build/release/int4_cli.exe`, or `build/release/Release/int4_cli.exe`
with a multi-configuration generator. `--help` lists all options. The executable
has no Python dependency. C++17 floating-point `from_chars` support is required.

Each input contains two positive decimal dimensions followed by exactly that
many row-major FP32 values, separated by whitespace:

```text
# X.txt (do not include this comment in the actual file)
1 2
1 -1

# W.txt
2 1
7
0
```

The output is `1 1` followed by approximately `7` in all modes and with either
INT4 kernel. Input files do not support comments. Numeric data may use signs and
scientific notation; dimensions must be unsigned decimal integers. Output uses
FP32 `max_digits10` precision for round trips.

| Option | Values / default |
| --- | --- |
| `--x`, `--w`, `--out`, `--csv` | Required X, W, Y and timing CSV paths |
| `--mode` | `fp32`, `direct`, `rotated`; default `rotated` |
| `--kernel` | `scalar`, `blocked`; default `blocked` (tile 32) |
| `--seed` | Unsigned 64-bit sign seed, including 18446744073709551615; default 0 |
| `--scale-precision` | `float` or `double`; default `float` |
| `--warmups` | 0..1000 untimed repetitions; default 2 |
| `--runs` | 1..1000 timed repetitions; default 5 |

FP32 mode uses the independent FP32 baseline; its CSV records `kernel=fp32`.
The scale type is dispatched once before numerical computation. X must have
1..32 rows, W must have 1..1024 columns, and their shared K must be a power of two
in 1..1024. Header bounds are checked before allocating any matrix.

Missing/malformed files, missing or extra values, nonfinite or FP32-out-of-range
values, invalid shapes/options, quantization underflow and output overflow produce
a stderr diagnostic and nonzero exit status. Output paths may not alias an input
or each other. Numerical processing finishes before any output file is opened;
parent output directories are created as needed. Existing output files are replaced.

## Prepared operator

```cpp
#include "int4/linear.hpp"
#include "int4/blocked_kernel.hpp"

int4::QuantizedLinear<double> op(int4::LinearMode::Rotated, 0);
op.prepare(w);                         // W is [K,N], retained as packed codes/scales
int4::BlockedKernel<double> kernel;
auto y = op.run(x, kernel);            // X is [M,K], Y is [M,N]
```

`prepare` is transactional: a new state is fully built before the old state is
exchanged by a nonthrowing owning-pointer swap. Failure leaves weights, dimensions
and signs unchanged. The move-only operator owns its state; moves transfer that
state and leave the source unprepared. Run and storage access before prepare throw
`logic_error`. Packed accessors also reject FP32 mode.

The driver uses `quantize_input(X)`, `packed_weights()`, `run(X,kernel)`,
`weight_bytes()`, `signs_bytes()`, and `rotation_scratch_bytes()`. All retain the
core's column-scaled `[K,N]` weights, row-scaled `[M,K]` inputs and shared
`rescale_output` reconstruction. `HadamardRotation::apply_right` computes XQ;
`apply_transpose_left` computes Q^T W using the same signs. These in-place helpers
may partially change their argument on result overflow; the operator invokes
them only on fresh copies, preserving its transactional preparation guarantee.

## Timing and logical storage

The CSV has one row. `prep_*_ns` and `call_*_ns` report median, Q1, Q3 and IQR from
`bench::measure` / `bench::summarize`. Preparation includes construction, validation,
rotation and weight quantization. Calls include input validation, online rotation,
quantization, kernel, reconstruction and allocations. File I/O and storage-only
inspection are outside timing. A Debug build is useful for validation; use Release
for timing comparisons. `--runs 1` works, with IQR zero.

Storage fields use `bench/accounting.hpp`: retained weights/signs, caller X,
packed activations, INT32 sums, FP32 output, blocked-kernel scratch and the
`8*K`-byte double rotation workspace. The prepare/call peaks account for separate
buffer lifetimes; these are logical contents, excluding vector capacity,
allocator/object overhead, harness-held results and measured process memory.
Preparation describes a fresh operator. A reprepare additionally retains the old
state until the new state succeeds. The caller's original W is included in the
preparation peak, but not in the run peak (it can be released after prepare).

## Correctness fixtures

For `tests/data/handoff_*.txt`, expected Y is:

| Mode | Y |
| --- | --- |
| FP32 | `[49, -49]` |
| Direct W4A4 | `[49, -56]` |
| Rotated W4A4 | approximately `[47.25, -52.5]` |

The rotated K=2 input has code magnitudes 7 and 2; its error differs from the
direct rounding-tie example. Both kernels must agree within each mode.
CTest runs both fixtures in three modes, two kernels and both scale types;
each case exercises seeds 0, 1, 2 and the maximum uint64 seed. Numeric CLI tests
use Python's standard library when an interpreter is found. Strict parser and
operator tests, and `WILL_FAIL` rejection cases, also run without Python.
