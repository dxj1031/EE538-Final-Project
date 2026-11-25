# Report Section Draft: Rotation, Prepared Operator and CLI (Qinyan Zeng)

Status: draft for integration into the team report, 2026-11-25. Results use the
production operator in `results/2026-11-25_windows-x64-float/` and
`results/2026-11-25_windows-x64-double/`. Each directory contains the full 45-case
sweep, raw/NumPy-checked CSVs, environment metadata, timing plots and four
rotation-error figures in PNG and PDF. No full LLM inference is evaluated.

## Rotation and operator design

We apply an implicit orthogonal rotation Q = D H_K. D contains seeded signs in
{-1,+1}; H_K is the normalized Sylvester Walsh-Hadamard transform. Consequently,
X' = XQ = (XD)H_K and W' = Q^T W = H_K(DW), preserving X'W' = XW before
quantization. `HadamardRotation` evaluates these transforms with O(K log K)
work per vector and one reusable K-element double workspace. It stores K sign
bytes and never allocates dense Q. Double butterfly intermediates avoid
unnormalized FP32 overflow; normalization occurs once, followed by a checked
FP32 cast. This still quantizes FP32 transformed inputs, with float or double
scale storage selected separately.

The production implementation adapts Wooyoung Kim's temporary rotation and
retains its portable `mt19937_64` top-bit sign convention. The temporary benchmark
classes were removed, and their existing dense-Q and lifecycle tests now exercise
the production classes. Dense Q appears only in independent tests.

`QuantizedLinear<Scale>` supports FP32, direct W4A4 and rotated W4A4 modes.
`prepare(W)` validates power-of-two K in [1,1024], N in [1,1024], and finite
weights. INT4 modes retain column-scaled packed weights and, when rotated,
the corresponding signs. A complete new state is constructed before a
nonthrowing owning-pointer exchange. Thus invalid dimensions, nonfinite data,
allocation failure, rotation overflow or scale underflow cannot replace the
previous prepared state. The operator is move-only and owns its buffers.

`run(X,kernel)` validates M in [1,32] and matching K. It optionally applies the
same rotation, quantizes each activation row, dispatches through
`MatmulKernel<Scale>::run`, and reconstructs once with the shared `rescale_output`.
The FP32 mode reuses Wooyoung's independent FP32 baseline. Packed layout, signed
nibbles, scale rules and INT32 accumulation remain Member 1's unchanged v1
contract. The benchmark uses this operator's preparation, quantization, packed
weight and storage accessors for both scalar and blocked paths.

## File-to-file interface and verification

`int4_cli` reads dimension headers followed by exactly the required row-major
FP32 values and writes Y in the same format. Options select mode, kernel, sign
seed and scale precision. Precision is dispatched once outside numerical loops.
The parser checks shape bounds before allocation and rejects malformed, missing,
extra, nonfinite and out-of-range values. Diagnostics use stderr and failures
return nonzero. Output paths cannot overwrite inputs or alias each other,
including new Windows filenames differing only in case. A separate CSV reports
weight-preparation and input-call median/Q1/Q3/IQR, plus logical storage estimates,
using the shared timing and accounting utilities. File I/O is excluded from timing.

The 54-test CTest suite passed Windows x64 Clang Release with all requested
warnings treated as errors, and Debug with AddressSanitizer and
UndefinedBehaviorSanitizer. GCC 11.4 on Ubuntu 22.04/WSL2 additionally passed the
C++ suites in Release and Debug with libstdc++ checking, and all 24 CLI fixture
combinations in Release. MSVC was unavailable. Details are in
`docs/validation_qinyan.md`.

For K<=64, tests compare the fast transform to dense Q with identical signs,
verify Q^T Q=I, and require unquantized product relative error <=1e-5. Additional
tests cover every allowed K through 1024, extreme seeds, K=1, N=33, invalid
domains, finite-value boundaries, reprepare failure and moved-state ownership.
Both supplied CLI fixtures run in three modes, two kernels and both scale
types, each with four seeds. Another 72 nontrivial CLI outputs match independent
Python MT19937-64 signs, dense-Q transforms and NumPy quantization/INT64 sums
at K=1,2,64, M=3 and N=33.

## Error results

The experiment uses Gaussian inputs/weights at K=N in {256,512,1024}, data seeds
0,1,2 and rotation seed 0. It scales ceil(0.01K) activation columns by a. We sweep
a in {1,10,100} at M=8 and M in {1,8,32} at a=10. Each precision produces 165
CSV rows: zero scalar/blocked mismatches, 165/165 NumPy FP64 metric agreements,
and 60/60 direct outputs bitwise equal to independent same-code NumPy results.
Plots take one tile-32 blocked result per seed and report medians across seeds;
scalar and tile-tuning duplicates are excluded.

At M=8 and a=10:

| Scale storage | K=N | Relative direct | Relative rotated | Max abs direct | Max abs rotated |
| --- | ---: | ---: | ---: | ---: | ---: |
| float | 256 | 0.377056 | 0.174251 | 44.8017 | 16.9680 |
| float | 512 | 0.459842 | 0.180605 | 81.3556 | 29.7764 |
| float | 1024 | 0.558300 | 0.197570 | 100.8918 | 35.4238 |
| double | 256 | 0.377056 | 0.174251 | 44.8017 | 16.9680 |
| double | 512 | 0.459842 | 0.180605 | 81.3556 | 29.7764 |
| double | 1024 | 0.558300 | 0.197570 | 100.8918 | 35.4238 |

Rotation reduces the median relative error by approximately 2.16x, 2.55x and
2.83x respectively. Float/double scale variants differ at finer precision, but
not materially at the table's resolution. FP32 relative error remains below
6.5e-7 across the sweep.

The a=1 control shows little benefit and can slightly favor direct quantization.
At a=100, outlier contributions dominate the FP64 output norm; a lower relative
error does not ensure smaller absolute error. For example, at K=N=256, M=8 with
float scales, the relative median improves from 0.177360 to 0.147416, while the
maximum-absolute-error median rises from 107.077 to 120.950. The figures therefore
show both metrics against both a and M.

## Limits, reproducibility and contributions

These synthetic CPU experiments test one linear operator and three data seeds;
they establish neither full-model accuracy nor a universal error reduction.
Rotation adds online work and a temporary FP32 copy plus 8K double-workspace
bytes. Logical storage peaks distinguish rotation, quantization, kernel and
reconstruction lifetimes; they exclude capacity, allocator/object and process
overhead. Repreparing an existing operator also retains old state until commit.
Timing uses one software thread, two warm-ups and five measured repetitions on
an Intel Core i9-14900HX, Windows x64, Clang 23.1.3, Release `-O3 -DNDEBUG`.
Absolute times remain specific to this compiler and machine.

Reproduce each precision with `scripts/run_benchmark.sh OUTPUT --scale-precision
float|double`, or the PowerShell counterpart documented in `docs/benchmark.md`.
The scripts build, run CTest, benchmark, independently validate and plot. Binary
dumps are regenerable and ignored by Git; CSVs, metadata and figures are retained.

Qinyan's module integrates rotation, transactional preparation, the three-mode
operator, strict CLI, additional tests, rotation-error plots and this report
section. It reuses Member 1's core and Wooyoung's baseline, benchmark utilities
and initial rotation design. Codex assisted with implementation, tests, analysis
and documentation; the validation results above are from executed tools.
