# Member 3: Blocked Kernel and Benchmark Driver

Owner: Member 3 (Wooyoung Kim). This document covers `BlockedKernel`, the C++ benchmark driver `int4_bench`, the NumPy reference check, and the plotting script. Declarations in the headers are authoritative if this document disagrees with them.

## Components

| Path | Purpose |
| --- | --- |
| `include/int4/blocked_kernel.hpp` | `int4::BlockedKernel<Scale>`: a second `MatmulKernel<Scale>` implementation |
| `include/bench/stats.hpp` | `steady_clock` timing harness, median / Q1 / Q3 / IQR |
| `include/bench/workload.hpp` | Seeded workloads (proposal Section 6.3), case list (Section 6.4) |
| `include/bench/metrics.hpp` | FP64 reference product, max-abs and relative Frobenius error (Eq. 3) |
| `include/bench/linear.hpp` | FP32 baseline, a **stand-in** rotation and prepared W4A4 operator (see below) |
| `include/bench/accounting.hpp` | Logical-buffer storage accounting |
| `include/bench/dump.hpp` | Binary matrix dumps for the Python reference check |
| `bench/int4_bench.cpp` | Benchmark driver executable |
| `tests/test_blocked_kernel.cpp` | Blocked-kernel tests (CTest `int4_blocked_kernel_tests`) |
| `tests/test_bench.cpp` | Benchmark support tests (CTest `int4_bench_support_tests`) |
| `scripts/reference_check.py` | Independent NumPy FP64 reference errors; appends `np_*` columns |
| `scripts/plot_results.py` | Timing plots and `summary.md` tables |
| `scripts/run_benchmark.sh` | Whole pipeline: build, CTest, sweep, reference check, plots |

## Blocked kernel

`BlockedKernel<Scale>` derives from `MatmulKernel<Scale>` and keeps the exact `run(qx, qw)` contract of `ScalarKernel`. It takes the same packed operands, applies the same acceptance rules, and returns the same INT32 sums:

- **Accepted operands:** `validate()` on both operands, row-scaled X `[M,K]`, column-scaled W `[K,N]`, matching K, and the `49*K <= INT32_MAX` guard (`check_inner_dimension`).
- **Algorithm:** loop over `T x T` tiles (default `T = 32`; any `1 <= T <= 1024` is accepted). Each X tile (`min(M,T) x T`) and each W tile (`T x T`) is decoded once into INT8 scratch. The inner loop multiplies the decoded INT8 values into the INT32 output row, so it never decodes nibbles. Tiles that start on an even column decode one byte into two codes with a 256-entry compile-time table. Odd tile edges fall back to per-nibble sign extension `(nibble ^ 8) - 8`.
- **Decode counts:** X is decoded exactly once and W `ceil(M/T)` times. In the proposal domain (`M <= 32 = T`) every code is therefore decoded once per call. The scalar kernel instead decodes two nibbles, with bounds checks, for each of the `M*K*N` multiply-adds.
- **Partial tiles:** real extents (`mt`, `kt`, `nt`) are kept separate from the scratch stride `T`. Padding nibbles of odd-length rows are never read, and the result never mixes padded entries.
- **Scratch:** allocated once per `run()` and reused for every tile. `scratch_bytes(M) = (min(M,T) + T) * T` bytes, which is 1,280 B for M = 8 and T = 32. Per-call allocation keeps the `const` `run()` thread-safe. No mutable state is shared across calls.
- **Output and scaling:** reconstruction stays in the shared `int4::rescale_output`. The blocked kernel never chooses scales or requantizes its inputs.

`BlockedKernel` uses only standard C++. It contains no intrinsics, SIMD code, or threads. The compiler may auto-vectorize the contiguous inner loop at `-O2`.

## Stand-in rotation and prepared operator

The proposal assigns `QuantizedLinear::prepare/run` and the Hadamard rotation to Member 2. Neither was in the repository when this benchmark was written. To measure all three modes, `include/bench/linear.hpp` implements the lifecycle from HANDOFF.md Section 4 as `bench::W4A4Linear<Scale>` and `bench::HadamardRotation`:

- `Q = D H_K`, with seeded signs `D_ii` drawn from `std::mt19937_64(rotation_seed)`, top bit set meaning -1. `H_K` is the orthonormal Sylvester Walsh–Hadamard matrix, applied in place by the fast transform in FP32. A dense Q is never allocated.
- `X' = XQ` scales the columns by `d`, then applies the FWHT to each row. `W' = Q^T W` scales the rows by `d`, then applies the FWHT along K.
- `prepare(W)` validates the domain (K a power of two ≤ 1024, `N ≤ 1024`, finite values). It rotates if needed and quantizes per column. New state is built before it replaces the old state, so a failed `prepare` leaves the previous state intact. `run(X, kernel)` validates `M ≤ 32` and K, rotates with the same Q, quantizes per row, calls the kernel, and rescales.
- Tests compare the fast transform with a dense `Q` for K ≤ 64 using shared signs. They also check `Q^T Q = I` and that the unquantized identity `(XQ)(Q^T W) = XW` holds to relative error 1e-5.

**When Member 2's operator is merged**, replace the two `bench::` classes in `bench/int4_bench.cpp`. The driver calls only `prepare(W)`, `quantize_input(X)`, `packed_weights()`, `run(X, kernel)`, `weight_bytes()`, and `signs_bytes()`. Rotated-mode error numbers depend on the sign generator, so rerun the sweep after the swap. FP32, direct, and kernel-comparison numbers do not depend on it.

## Workloads (proposal Sections 6.3–6.4)

- For each scale `K = N` in {256, 512, 1024} and each data seed in {0, 1, 2}, a Gaussian `X0 [32,K]` and `W [K,N]` are drawn once. Each case takes the first M rows of `X0` and multiplies the same `ceil(0.01K)` columns by `a`. These are 3, 6, and 11 columns, chosen by a seeded partial Fisher–Yates shuffle and recorded in the metadata file.
- The cases are controlled sweeps, not the Cartesian product: `M ∈ {1, 8, 32}` at `a = 10`, and `a ∈ {1, 10, 100}` at `M = 8`. The shared case `(M = 8, a = 10)` is listed once with `sweep=both`. This gives 15 cases per seed and 45 in total. `a = 1` is the control. The rotation seed is 0.
- Generation uses only `std::mt19937_64` and `std::seed_seq`, which the standard specifies exactly, plus explicit Box–Muller, so the data match across compilers. `std::normal_distribution` is implementation-defined and is not used.

## Measurement protocol (proposal Section 6.6)

All timing uses `std::chrono::steady_clock` on one thread from a Release build, with 2 warm-ups, 5 timed runs, and the median and IQR reported. Quantiles use linear interpolation, so for 5 runs Q1, the median, and Q3 are the 2nd, 3rd, and 4th order statistics. Only the call itself is timed. Each result is destroyed outside the timed region, and the last result is consumed after timing for the checksum, the error, and the exactness checks. Input generation, the FP64 reference, and file dumps are never timed.

| Column group | What is inside the timed region |
| --- | --- |
| `prep_*` | Construct the operator plus `prepare(W)`: domain checks, a copy (FP32), or rotation, quantization and packing (W4A4) |
| `call_*` | `run(X)`: domain checks, online rotation, quantization/packing of X, kernel, rescaling, and every allocation |
| `kernel_*` | `kernel.run(qx, qw)` on pre-built packed operands: validation scans and INT32 output allocation included. For FP32 rows, `fp32_matmul` only |
| `validate_median_ns` | `qx.validate(); qw.validate();` alone. This is the share of `kernel_*` that is not arithmetic |
| `harness_median_ns` | An empty timed call, giving the `steady_clock` overhead and resolution (about 42 ns ticks on Apple silicon) |

A speedup is not a project requirement, and timing claims concern only these implementations.

## Correctness inside the driver

For every W4A4 case and mode, the driver runs the scalar kernel once, untimed, on the identical packed operands. Every timed kernel's INT32 sums (`sums_match_scalar`) and every reconstructed Y (`y_match_scalar`) must be bitwise identical to that result. A mismatch is reported on stderr, and the driver exits with status 3 after finishing the sweep.

`scripts/reference_check.py` then performs three checks independently in NumPy:

- It recomputes `Y64 = FP64(X) @ FP64(W)`, the max-abs error, and the relative Frobenius error for every row. These must match the C++ values within a rigorous summation-order bound of `2 K eps sum|x||w|`.
- It confirms that each dumped Y matches the CSV checksum.
- For direct rows, it rebuilds the output with an independent NumPy quantizer (ties away from zero) and INT64 accumulation, and requires `rtol 1e-5, atol 1e-6` agreement. On this platform the match is bitwise.

## CSV schema (`bench.csv`)

There is one row per (case, mode, kernel, tile). Times are in nanoseconds and sizes in logical bytes.

| Columns | Meaning |
| --- | --- |
| `case_id` | `K{K}_M{M}_a{a}_s{seed}`, also the dump file stem |
| `data_seed, rotation_seed, K, N, M, a, sweep, outlier_cols` | Workload identity (`sweep` ∈ batch / outlier / both) |
| `mode` | `fp32`, `direct`, or `rotated` |
| `kernel, tile` | `fp32`; `blocked` with its tile edge; or `scalar` (empty tile) |
| `scale_precision, warmups, runs` | Run configuration |
| `prep_*, call_*, kernel_*` | `median, q1, q3, iqr` of the timed runs |
| `validate_median_ns, harness_median_ns` | See the protocol table |
| `max_abs_err, rel_frob_err` | Against `Y64`, proposal Eq. (3) with the `1e-12` floor |
| `sums_match_scalar, y_match_scalar` | `1` when bitwise equal to the scalar kernel on the same operands (empty for FP32) |
| `weight_bytes, signs_bytes` | Retained prepared state: an FP32 copy, or packed codes + scales, plus K sign bytes |
| `input_bytes, act_bytes, sums_bytes, output_bytes, scratch_bytes` | Caller X, packed X + scales, INT32 sums, FP32 Y, kernel scratch |
| `resident_bytes` | `weight_bytes + signs_bytes` (kept between calls) |
| `prep_peak_bytes` | Peak inside `prepare`: caller W, rotated copy W' (if rotated), quantizer maxima (8 B per column), packed W, signs |
| `call_peak_bytes` | Resident state plus caller X plus the largest call phase. The phases are quantize (X' copy if rotated, maxima, packed X), kernel (packed X, sums, scratch), and rescale (packed X, sums, Y) |
| `checksum` | Sum of Y (consumes the output after timing) |

`reference_check.py` adds `np_max_abs_err`, `np_rel_frob_err`, `np_metrics_agree`, `np_direct_samecode_ok`, and `np_direct_samecode_exact_frac`.

The accounting covers logical buffer contents in the same sense as `storage_bytes()`. It excludes vector capacity, allocator and object overhead, offline references (`Y64`), and process memory. It is not a measurement of resident set size.

## Running

```text
# Whole pipeline (macOS/Linux; needs CMake, a C++17 compiler, Python 3 with NumPy + Matplotlib):
scripts/run_benchmark.sh results/<date>_<machine>

# Manually (any platform, including Windows/PowerShell with the same arguments):
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
build/release/int4_bench --csv results/run/bench.csv --dump-dir results/run/dumps --tuning-tiles 16,64
python3 scripts/reference_check.py results/run/bench.csv results/run/dumps --out results/run/bench_with_reference.csv
python3 scripts/plot_results.py results/run/bench_with_reference.csv --out-dir results/run
```

Run `int4_bench --help` to list the options: `--scales`, `--seeds`, `--rotation-seed`, `--warmups`, `--runs`, `--tile`, `--tuning-tiles`, `--scalar-timing protocol|all|none`, `--scale-precision float|double`, `--quick`. Invalid options print a diagnostic and exit with status 2. Runtime errors exit with status 1.

Keep `bench.csv`, `bench.csv.meta.txt` (seeds, outlier columns, compiler, build type, clock), `environment.txt` (CPU, memory, CMake cache flags, tool versions), `bench_with_reference.csv`, the plots, and `summary.md`. The `dumps/` directory is git-ignored and can be regenerated with the same command. A full sweep's dumps take about 25 MB.

For a sanitizer build, configure with `-DINT4_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug` (GCC/Clang). This enables AddressSanitizer and UndefinedBehaviorSanitizer for every target.
