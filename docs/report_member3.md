# Report Section Draft: Blocked Kernel and Benchmark Driver (Member 3)

Status: draft for the final report, written 2026-11-21. Every number comes from `results/2026-11-21_apple-m4/`, the full 45-case protocol run on one Apple M4 core (Apple clang 21, `-O3 -DNDEBUG`, CMake Release). Rotated-mode numbers use the benchmark's stand-in rotation (see Limitations) and should be regenerated after Member 2's operator is merged.

## Design

**Blocked kernel.** `BlockedKernel<Scale>` implements the same virtual `MatmulKernel<Scale>::run(qx, qw)` contract as the scalar kernel, so both kernels run on identical packed operands. It applies the scalar path's acceptance rules: operand validation, row/column scale axes, matching K, and the `49K ≤ INT32_MAX` bound. It walks `T × T` tiles (`T = 32`). Each X tile and each W tile is decoded once into INT8 scratch, which is allocated once per call and reused for every tile. A 256-entry compile-time table turns one packed byte into two signed codes. The inner loop then multiplies decoded INT8 values into the INT32 output row. In the proposal domain (`M ≤ 32`) every code is decoded exactly once per call, while the scalar kernel decodes two nibbles with bounds checks for each of the `MKN` multiply-adds. Partial tiles iterate only over real extents, and padding nibbles of odd-length rows are never read. Scratch is `(min(M,T) + T)·T` bytes, which is 1,280 B at M = 8. Allocating it per call keeps the `const` `run()` thread-safe. Output reconstruction remains the shared `rescale_output`.

**Benchmark driver.** `int4_bench` generates the controlled workloads of Section 6.3 with a portable generator: `mt19937_64` and `seed_seq` with explicit Box–Muller, rather than the implementation-defined `std::normal_distribution`. This lets a teammate's machine reproduce the same matrices. For each case it measures FP32, direct W4A4, and rotated W4A4. All timing uses `steady_clock` on one thread with 2 warm-ups and 5 timed runs, and reports the median, Q1, Q3, and IQR. The driver records, separately:

- weight-preparation time;
- full input-call latency (rotation, quantization/packing, kernel, rescale, and allocations);
- kernel-only time on pre-built operands;
- the operand-validation share of the kernel time;
- the empty-harness overhead.

Results are destroyed outside the timed region and consumed after timing. The driver also writes logical storage accounting, covering resident state and the peaks inside `prepare` and `run`, plus metadata (seeds, outlier columns, compiler, build type).

## Verification

- **Unit tests.** `int4_blocked_kernel_tests` cover:
  - all 225 code pairs, and every pair in both nibble positions;
  - the HANDOFF and proposal fixtures;
  - exact agreement with the scalar kernel and with an independent INT64 byte-level reference across M ∈ {1,…,65}, K ∈ {1,…,100}, N ∈ {1,…,65}, and tile edges {1, 2, 3, 7, 16, 32, 64} (partial and odd-start tiles, padded rows, `K = 1`, `N = 33`);
  - all-zero rows and columns, the `±49·1024` extremes, and the rejection rules.

  Deliberate faults in decoding, tile bounds, nibble parity, and the K loop all made the suite fail.
- **In-run checks.** For every case the driver requires bitwise-identical INT32 sums and Y from every blocked configuration versus the scalar kernel on the same operands. Result: 0 mismatches over 165 rows.
- **Independent reference.** `reference_check.py` recomputes `Y64` and both error metrics with NumPy FP64. All 165 rows agree with the C++ values within a summation-order bound. All 60 direct-mode outputs are bitwise equal to an independent NumPy quantizer with INT64 accumulation, which exceeds the proposal's rtol 1e-5 / atol 1e-6 requirement.
- **Build hygiene.** The full CTest suite (7 tests) passes in Release and under AddressSanitizer + UndefinedBehaviorSanitizer with libc++ hardening.

## Results (median across seeds 0–2)

**Does blocking reduce repeated unpacking?** Yes. On identical operands at `M = 8, a = 10`, the blocked kernel is 6.2×, 6.1×, and 7.3× faster than the scalar kernel at `K = N` = 256, 512, and 1024 (direct mode: 69.5 vs 429.4 µs, 292.8 vs 1,786.4 µs, 1,168.8 vs 8,569.5 µs). The INT32 sums are identical in every case. Operand validation, which both kernels perform on every call, accounts for 178 µs of the 1,169 µs at K = 1024. In the optional tuning at `K = N = 512`, `T = 64` (214 µs) beats `T = 32` (293 µs) and `T = 16` (328 µs). Longer contiguous inner loops amortize per-tile overhead, at the cost of 4,608 B of scratch.

**Cost relative to FP32.** The W4A4 input call is about 2.4× slower than the scalar FP32 baseline at `M = 8`: 1,198 vs 500 µs at K = 1024. With so few activation rows, decoding and validating the `K × N` weight codes is amortized over only M rows. The FP32 loop reads weights directly. Online rotation adds little to the call (1,217 vs 1,198 µs) because `O(MK log K)` is small next to `O(MKN)`. It adds about 41% to weight preparation (2.50 vs 1.77 ms), which is paid once per weight. Speedup was not a project goal.

**Storage.** At K = N = 1024 the prepared W4A4 weights occupy 528,384 B (524,288 B of codes plus 4,096 B of float scales), versus 4,194,304 B for FP32, a 7.9× reduction. Rotation adds 1,024 sign bytes. The peak inside `prepare` is higher for the rotated mode (8.9 MB vs 4.7 MB) because it holds a rotated FP32 copy of W during quantization.

**Error summary.** These numbers are for context only; the rotation-error analysis is Member 2's section.

| a | K = N = 1024, M = 8 | E_F direct | E_F rotated |
| ---: | --- | ---: | ---: |
| 1 (control) | | 0.200 | 0.201 |
| 10 | | 0.558 | 0.198 |
| 100 | | 0.195 | 0.181 |

FP32 error stays below 6.5e-7 in every row. At `M = 8, a = 10`, rotation reduced `E_F` by 2.1–2.9× per seed. Over all `a = 10` cases the per-seed reduction ranges from 1.1× to 3.2×, with the smallest value at `M = 1, K = 256`, seed 0. Rotation is neutral on the `a = 1` control. At `a = 100` the outlier columns dominate `‖Y64‖_F`. Direct quantization represents those columns well, so `E_F` shows only a small gain from rotation (0.99–1.24× per seed). Maximum absolute error should be read alongside `E_F` there.

## Limitations

- **Stand-in rotation.** Rotated-mode results use the benchmark's documented stand-in for Member 2's rotation and prepared operator, which is FWHT-based and tested against a dense Q. Swap in the team operator and rerun before reporting final numbers.
- **One machine and compiler.** Results come from a single Apple M4 with Apple clang. GCC/MSVC builds of the new files were not verified, and absolute times will differ on the fallback machine.
- **Clock resolution.** `steady_clock` has about 42 ns resolution on this platform, which matters only for the smallest calls (harness overhead rounds to 0 ns). Call-time IQR was at most 12.6% of the median, and 1.9% in the median row.
- **Validation is inside the kernel timing.** Kernel time includes per-call operand validation and output allocation, as required by the shared interface. The separately reported validation time bounds that share.
- **Logical accounting.** Storage figures are logical buffer sizes, not measured process memory.

## Reproduction

`scripts/run_benchmark.sh results/<date>_<machine>` builds, tests, runs the sweep with `--tuning-tiles 16,64`, performs the NumPy reference check, and writes the plots and `summary.md`. docs/benchmark.md gives the manual commands and the CSV schema.

## Contributions and AI use (Member 3, to be finalized)

- **Contributions.** Member 3: blocked kernel, FP32 baseline, benchmark driver, workload generator, storage accounting, NumPy reference check, plotting, tests, and this section.
- **AI use.** To be completed before submission.
