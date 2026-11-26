# Blocked Kernel and Benchmark Driver

Owner: Wooyoung Kim. This document covers `BlockedKernel`, the C++ benchmark driver `int4_bench`, the NumPy reference check, and the plotting script. Declarations in the headers are authoritative if this document disagrees with them.

## Components

| Path | Purpose |
| --- | --- |
| `include/int4/blocked_kernel.hpp` | `int4::BlockedKernel<Scale>`: a second `MatmulKernel<Scale>` implementation |
| `include/bench/stats.hpp` | `steady_clock` timing harness, median / Q1 / Q3 / IQR |
| `include/bench/workload.hpp` | Seeded workloads (proposal Section 6.3), case list (Section 6.4) |
| `include/bench/metrics.hpp` | FP64 reference product, max-abs and relative Frobenius error (Eq. 3) |
| `include/bench/linear.hpp` | Independent scalar FP32 baseline (Wooyoung Kim) |
| `include/int4/hadamard.hpp`, `include/int4/linear.hpp` | Production rotation and `int4::QuantizedLinear<Scale>` (Qinyan Zeng) |
| `include/bench/accounting.hpp` | Logical-buffer storage accounting |
| `include/bench/dump.hpp` | Binary matrix dumps for the Python reference check |
| `bench/int4_bench.cpp` | Benchmark driver executable |
| `tests/test_blocked_kernel.cpp` | Blocked-kernel tests (CTest `int4_blocked_kernel_tests`) |
| `tests/test_bench.cpp` | Benchmark support tests (CTest `int4_bench_support_tests`) |
| `scripts/reference_check.py` | Independent NumPy FP64 reference errors; appends `np_*` columns |
| `scripts/plot_results.py` | Timing plots and `summary.md` tables |
| `scripts/plot_rotation_error.py` | Qinyan Zeng: paired relative/absolute error plots and seed medians |
| `scripts/run_benchmark.ps1` | Qinyan Zeng: native PowerShell pipeline |
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

## Production rotation and prepared operator

The benchmark now uses Qinyan Zeng's `int4::QuantizedLinear<Scale>` and
`int4::HadamardRotation`; the temporary `bench::W4A4Linear` and
`bench::HadamardRotation` have been removed. The independent FP32 baseline stays
in `include/bench/linear.hpp` and is also reused by the operator's FP32 mode.

- `Q = D H_K`; signs use the original `std::mt19937_64` top-bit mapping, so sign
  seeds retain their meaning. XQ applies signs before a row FWHT; Q^T W applies
  signs before a column FWHT. Production code never constructs dense Q.
- Butterflies and normalization use a reusable K-element double workspace; each
  transformed result is checked and cast to FP32 once. This avoids spurious
  overflow in unnormalized FP32 butterflies. True FP32 result overflow is rejected.
- `prepare(W)` enforces power-of-two K in [1,1024], N in [1,1024], and finite
  inputs. It builds a complete new state, then exchanges owning pointers without
  throwing. Failed validation, allocation, rotation or quantization keeps the old
  state. The operator is move-only and owns its retained weights.
- `run(X,kernel)` enforces M in [1,32] and matching K. INT4 modes rotate if needed,
  quantize rows, call the shared kernel interface, and use `rescale_output` once.
  FP32 mode uses the independent baseline and rejects nonfinite output.
- Driver accessors are `quantize_input`, `packed_weights`, `weight_bytes`,
  `signs_bytes`, and `rotation_scratch_bytes`. Packed accessors reject FP32 mode.
- The existing dense-Q, orthogonality and unquantized-identity tests now exercise
  the production classes. Additional tests cover all allowed K values, extreme
  seeds, overflow, failed reprepare, ownership and all three modes.

Rerun the full sweep after this integration: double butterfly intermediates can
change rounding and quantized codes relative to the original temporary FP32 path.

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

The rotation workspace is `8*K` logical bytes and is released before quantization.
Preparation and call peaks take the maximum of the rotation, quantization, kernel
and reconstruction phases; the workspace is not added to simultaneously dead
quantizer buffers. Preparation estimates describe an initially unprepared operator;
repreparing an existing operator also retains its old state until commit.

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
python3 scripts/plot_rotation_error.py results/run/bench_with_reference.csv --out-dir results/run
```

Run `int4_bench --help` to list the options: `--scales`, `--seeds`, `--rotation-seed`, `--warmups`, `--runs`, `--tile`, `--tuning-tiles`, `--scalar-timing protocol|all|none`, `--scale-precision float|double`, `--quick`. Invalid options print a diagnostic and exit with status 2. Runtime errors exit with status 1.

Keep `bench.csv`, `bench.csv.meta.txt` (seeds, outlier columns, compiler, build type, clock), `environment.txt` (CPU, memory, CMake cache flags, tool versions), `bench_with_reference.csv`, the plots, and `summary.md`. The `dumps/` directory is git-ignored and can be regenerated with the same command. A full sweep's dumps take about 25 MB.

For a sanitizer build, configure with `-DINT4_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug` (GCC/Clang). This enables AddressSanitizer and UndefinedBehaviorSanitizer for every target.

## Rotation error experiments (Qinyan Zeng)

Run each precision into its own result directory to keep cases and references
unambiguous. The default full sweep uses data seeds 0,1,2, rotation seed 0,
K=N in {256,512,1024}, M in {1,8,32} at a=10, and a in {1,10,100} at M=8.

```text
scripts/run_benchmark.sh results/<date>_<machine>-float --scale-precision float
scripts/run_benchmark.sh results/<date>_<machine>-double --scale-precision double
```

Native PowerShell (CMake, compiler and Python must be available):

```powershell
./scripts/run_benchmark.ps1 -OutputDir results/run-float -ScalePrecision float
./scripts/run_benchmark.ps1 -OutputDir results/run-double -ScalePrecision double
# Add -Quick to validate the pipeline on K=16,64 and seed 0.
```

Both pipeline scripts run the production operator, independent reference check,
timing plots and rotation-error plots. The latter filters to one tile-32 blocked
row per seed/mode; scalar and tuning duplicates never enter the medians. It
requires validated metrics and scalar agreement, rejects duplicate/missing seed
pairs, and groups scale precisions separately. A quick run labels its actual seed
set; full report results use seeds 0,1,2.

For each scale precision it writes relative Frobenius error and maximum absolute
error against a (M=8) and against M (a=10), with one panel per K=N. These four plots
are saved as PNG and vector PDF. `rotation_error_medians.csv` contains the exact
seed medians and `rotation_error_summary.md` lists both error metrics. At a=100,
the outlier terms dominate the output norm; maximum absolute error must be read
alongside the relative metric. A gain in one does not guarantee a gain in the other.
