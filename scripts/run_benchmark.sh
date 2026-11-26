#!/usr/bin/env bash
# Wooyoung Kim: reproducible benchmark pipeline.
#   configure + build (Release) -> CTest -> int4_bench (full protocol + optional tile
#   tuning) -> NumPy FP64 reference check -> plots and summary tables.
#
# Usage: scripts/run_benchmark.sh [OUTPUT_DIR] [extra int4_bench options...]
#   OUTPUT_DIR defaults to results/<date>_<machine>. Environment overrides:
#   BUILD_DIR (default build/release), PYTHON (default python3).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results/$(date +%Y-%m-%d)_$(uname -m)}"
[ $# -gt 0 ] && shift
BUILD="${BUILD_DIR:-$ROOT/build/release}"
PYTHON="${PYTHON:-python3}"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON"
cmake --build "$BUILD" --config Release
ctest --test-dir "$BUILD" -C Release --output-on-failure

BENCH=""
for candidate in "$BUILD/int4_bench" "$BUILD/Release/int4_bench" "$BUILD/int4_bench.exe" "$BUILD/Release/int4_bench.exe"; do
    if [ -x "$candidate" ]; then BENCH="$candidate"; break; fi
done
[ -n "$BENCH" ] || { echo "run_benchmark: int4_bench not found under $BUILD" >&2; exit 1; }

mkdir -p "$OUT"
{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "os=$(uname -s -r -m)" # No hostname: results are committed to a shared repository.
    if [ "$(uname)" = "Darwin" ]; then
        echo "cpu=$(sysctl -n machdep.cpu.brand_string)"
        echo "logical_cpus=$(sysctl -n hw.ncpu)"
        echo "memory_bytes=$(sysctl -n hw.memsize)"
    elif [ -r /proc/cpuinfo ]; then
        echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "logical_cpus=$(grep -c '^processor' /proc/cpuinfo)"
        echo "memory=$(grep MemTotal /proc/meminfo)"
    else
        echo "cpu=${PROCESSOR_IDENTIFIER:-unknown}"
    fi
    grep -E '^CMAKE_(CXX_COMPILER|CXX_FLAGS|CXX_FLAGS_RELEASE|BUILD_TYPE|GENERATOR)(:[A-Z]+)?=' "$BUILD/CMakeCache.txt" || true
    "${CXX:-c++}" --version 2>/dev/null | head -1 | sed 's/^/cxx_version=/' || true
    echo "cmake=$(cmake --version | head -1)"
    echo "python=$("$PYTHON" --version 2>&1)"
    "$PYTHON" -c 'import numpy, matplotlib; print("numpy=" + numpy.__version__); print("matplotlib=" + matplotlib.__version__)' || true
} > "$OUT/environment.txt"

"$BENCH" --csv "$OUT/bench.csv" --dump-dir "$OUT/dumps" --tuning-tiles 16,64 "$@"
"$PYTHON" "$ROOT/scripts/reference_check.py" "$OUT/bench.csv" "$OUT/dumps" --out "$OUT/bench_with_reference.csv"
"$PYTHON" "$ROOT/scripts/plot_results.py" "$OUT/bench_with_reference.csv" --out-dir "$OUT"
"$PYTHON" "$ROOT/scripts/plot_rotation_error.py" "$OUT/bench_with_reference.csv" --out-dir "$OUT"
echo "run_benchmark: results in $OUT (dumps/ is regenerable and git-ignored)"
