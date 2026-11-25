#!/usr/bin/env python3
"""Wooyoung Kim: independent NumPy FP64 reference check for int4_bench output.

For every row of the benchmark CSV this script loads the dumped X, W and Y, recomputes
the ground truth Y64 = FP64(X) @ FP64(W) with NumPy, and recomputes the maximum absolute
error and the relative Frobenius error of proposal Eq. (3). Both must agree with the C++
values within a rigorous summation-order bound.

For direct W4A4 rows it also rebuilds the output from an independent NumPy quantizer
(stored-scale rule s = max|v|/7 cast to the scale precision, s = 1 for zero groups,
round ties away from zero, clip to [-7, 7]) with integer accumulation, and requires
agreement with rtol 1e-5 and atol 1e-6 (proposal Section 5.3).

Usage:
    python3 scripts/reference_check.py RESULTS/bench.csv RESULTS/dumps [--out OUT.csv]

Writes the CSV with added np_* columns (default: <csv stem>_with_reference.csv) and
exits with status 1 on any disagreement.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

EPS64 = np.finfo(np.float64).eps


def load_matrix(path: Path) -> np.ndarray:
    """Reads the int4_bench dump format: uint64 rows, uint64 cols, FP32 row-major data."""
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"{path}: truncated header")
    rows, cols = (int(v) for v in np.frombuffer(raw[:16], dtype="<u8"))
    data = np.frombuffer(raw[16:], dtype="<f4")
    if rows == 0 or cols == 0 or data.size != rows * cols:
        raise ValueError(f"{path}: payload does not match {rows}x{cols} header")
    return data.reshape(rows, cols)


def round_half_away(values: np.ndarray) -> np.ndarray:
    """std::round semantics (ties away from zero); np.round would round ties to even."""
    whole = np.trunc(values)
    fraction = values - whole  # Exact for |v| < 2**52.
    return whole + np.where(np.abs(fraction) >= 0.5, np.sign(values), 0.0)


def quantize(matrix: np.ndarray, axis: str, scale_dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ('row') or per-column ('col') symmetric INT4 codes and stored scales."""
    values = matrix.astype(np.float64)
    max_abs = np.abs(values).max(axis=1 if axis == "row" else 0)
    scales = np.where(max_abs == 0, 1.0, max_abs / 7.0).astype(scale_dtype)
    stored = scales.astype(np.float64)
    ratio = values / (stored[:, None] if axis == "row" else stored[None, :])
    codes = np.clip(round_half_away(ratio), -7, 7).astype(np.int64)
    return codes, scales


def direct_same_code_reference(x: np.ndarray, w: np.ndarray, precision: str) -> np.ndarray:
    scale_dtype = np.float32 if precision == "float" else np.float64
    cx, sx = quantize(x, "row", scale_dtype)
    cw, sw = quantize(w, "col", scale_dtype)
    sums = cx @ cw  # Exact INT64 accumulation (no BLAS); |sum| <= 49 K.
    value = (sums.astype(np.float64) * sx.astype(np.float64)[:, None]) * sw.astype(np.float64)[None, :]
    return value.astype(np.float32)


def error_metrics(y: np.ndarray, y64: np.ndarray) -> tuple[float, float]:
    diff = y.astype(np.float64) - y64
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = float(np.linalg.norm(diff) / max(np.linalg.norm(y64), 1e-12))
    return max_abs, rel_frob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="int4_bench CSV")
    parser.add_argument("dumps", type=Path, help="int4_bench --dump-dir directory")
    parser.add_argument("--out", type=Path, help="augmented CSV path")
    args = parser.parse_args()
    out_path = args.out or args.csv.with_name(args.csv.stem + "_with_reference.csv")

    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"reference_check: {args.csv} has no rows", file=sys.stderr)
        return 1

    cache: dict[Path, np.ndarray] = {}

    def get(name: str) -> np.ndarray:
        path = args.dumps / name
        if path not in cache:
            cache[path] = load_matrix(path)
        return cache[path]

    failures: list[str] = []
    y64_cache: dict[str, tuple[np.ndarray, float]] = {}
    for row in rows:
        case_id, mode = row["case_id"], row["mode"]
        k, n, m = int(row["K"]), int(row["N"]), int(row["M"])
        x = get(f"X_{case_id}.bin")
        w = get(f"W_K{k}_s{row['data_seed']}.bin")
        # Only the tile-32 blocked (or FP32) output is dumped per mode; other kernels are
        # bitwise equal to it when y_match_scalar == 1, which is required here.
        y = get(f"Y_{case_id}_{mode}.bin")
        if x.shape != (m, k) or w.shape != (k, n) or y.shape != (m, n):
            failures.append(f"{case_id} {mode}: dump shapes do not match CSV")
            continue
        if mode != "fp32" and row["y_match_scalar"] != "1":
            failures.append(f"{case_id} {mode} {row['kernel']}{row['tile']}: C++ reported a kernel mismatch")

        if case_id not in y64_cache:
            x64, w64 = x.astype(np.float64), w.astype(np.float64)
            # Some BLAS builds (e.g. NumPy 2.0 with Apple Accelerate) raise spurious FP
            # flags inside matmul; ignore the flags and check the results explicitly.
            with np.errstate(all="ignore"):
                product = x64 @ w64
                magnitude = np.abs(x64) @ np.abs(w64)
            if not (np.all(np.isfinite(product)) and np.all(np.isfinite(magnitude))):
                failures.append(f"{case_id}: nonfinite FP64 reference")
                continue
            # Two different FP64 summation orders of the same K products differ by at most
            # about 2 K eps sum|x||w| (standard gamma_K bound); used as the agreement bound.
            bound = 2.0 * k * EPS64 * float(np.max(magnitude))
            y64_cache[case_id] = (product, bound)
        y64, bound = y64_cache[case_id]

        max_abs, rel_frob = error_metrics(y, y64)
        norm = max(float(np.linalg.norm(y64)), 1e-12)
        cpp_max_abs, cpp_rel = float(row["max_abs_err"]), float(row["rel_frob_err"])
        metrics_ok = (abs(cpp_max_abs - max_abs) <= bound + 1e-12 * max_abs and
                      abs(cpp_rel - rel_frob) <= bound * math.sqrt(m * n) / norm + 1e-12 * rel_frob)
        checksum_ok = math.isclose(float(row["checksum"]), math.fsum(y.astype(np.float64).ravel()),
                                   rel_tol=1e-9, abs_tol=1e-9 * float(np.sum(np.abs(y))) + 1e-12)
        if not metrics_ok:
            failures.append(f"{case_id} {mode}: C++ metrics ({cpp_max_abs}, {cpp_rel}) "
                            f"vs NumPy ({max_abs}, {rel_frob})")
        if not checksum_ok:
            failures.append(f"{case_id} {mode}: dumped Y does not match the CSV checksum")

        row["np_max_abs_err"] = repr(max_abs)
        row["np_rel_frob_err"] = repr(rel_frob)
        row["np_metrics_agree"] = "1" if metrics_ok and checksum_ok else "0"
        row["np_direct_samecode_ok"] = ""
        row["np_direct_samecode_exact_frac"] = ""
        if mode == "direct":
            reference = direct_same_code_reference(x, w, row["scale_precision"])
            close = bool(np.allclose(y, reference, rtol=1e-5, atol=1e-6))
            row["np_direct_samecode_ok"] = "1" if close else "0"
            row["np_direct_samecode_exact_frac"] = repr(float(np.mean(y == reference)))
            if not close:
                failures.append(f"{case_id} direct: output differs from the NumPy same-code reference")

    fields = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    direct_rows = [r for r in rows if r["mode"] == "direct"]
    exact = sum(float(r["np_direct_samecode_exact_frac"]) == 1.0 for r in direct_rows)
    print(f"reference_check: {len(rows)} rows, {len(y64_cache)} cases; "
          f"{sum(r['np_metrics_agree'] == '1' for r in rows)} rows agree with NumPy FP64 metrics; "
          f"{exact}/{len(direct_rows)} direct rows bitwise equal to the NumPy same-code reference")
    print(f"reference_check: wrote {out_path}")
    for failure in failures:
        print(f"reference_check: FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
