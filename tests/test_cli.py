#!/usr/bin/env python3
"""Qinyan Zeng: file-to-file numerical fixtures, CSV checks and seed portability.

Uses only the Python standard library; NumPy is not needed for CLI tests.
The HANDOFF fixture has different expected errors in direct and rotated modes.
For K=2 its rotated code ratio is 7:2, giving Y=[47.25,-52.5].
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--fixture", choices=("proposal", "handoff"), required=True)
    parser.add_argument("--mode", choices=("fp32", "direct", "rotated"), required=True)
    parser.add_argument("--kernel", choices=("scalar", "blocked"), required=True)
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    args = parser.parse_args()
    data = Path(__file__).resolve().parent / "data"
    if args.fixture == "proposal":
        expected = [7.0]
    else:
        expected = {"fp32": [49.0, -49.0], "direct": [49.0, -56.0],
                    "rotated": [47.25, -52.5]}[args.mode]
    with tempfile.TemporaryDirectory(prefix="int4 cli ") as temporary:
        directory = Path(temporary)
        # Paths with spaces and the entire uint64 seed range work on Windows too.
        for seed in (0, 1, 2, 2**64 - 1):
            output, timings = directory / "output Y.txt", directory / "timings.csv"
            command = [str(args.exe), "--x", str(data / f"{args.fixture}_x.txt"),
                       "--w", str(data / f"{args.fixture}_w.txt"), "--out", str(output),
                       "--csv", str(timings), "--mode", args.mode, "--kernel", args.kernel,
                       "--seed", str(seed), "--scale-precision", args.precision,
                       "--warmups", "0", "--runs", "2"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(f"CLI failed: {result.stderr}")
            tokens = output.read_text().split()
            assert tokens[:2] == ["1", str(len(expected))], tokens
            values = [float(value) for value in tokens[2:]]
            assert len(values) == len(expected)
            assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5)
                       for a, b in zip(values, expected)), (values, expected)
            with timings.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1
            row = rows[0]
            assert row["mode"] == args.mode and row["scale_precision"] == args.precision
            assert row["kernel"] == ("fp32" if args.mode == "fp32" else args.kernel)
            assert int(row["rotation_seed"]) == seed and row["M"] == "1" and row["K"] == "2"
            assert row["N"] == str(len(expected)) and row["runs"] == "2"
            for phase in ("prep", "call"):
                q1, median, q3 = (float(row[f"{phase}_{name}_ns"]) for name in ("q1", "median", "q3"))
                assert 0 <= q1 <= median <= q3 and math.isfinite(q3)
                assert math.isclose(float(row[f"{phase}_iqr_ns"]), q3 - q1, abs_tol=1e-6)
            assert int(row["resident_bytes"]) == int(row["weight_bytes"]) + int(row["signs_bytes"])
            assert int(row["signs_bytes"]) == (2 if args.mode == "rotated" else 0)
            assert int(row["rotation_scratch_bytes"]) == (16 if args.mode == "rotated" else 0)
            assert int(row["output_bytes"]) == 4 * len(expected)
            for field in ("weight_bytes", "resident_bytes", "prep_peak_bytes", "call_peak_bytes"):
                assert int(row[field]) > 0
        if os.name == "nt":
            # Both paths are new, so filesystem::equivalent cannot detect this
            # Windows alias until the files have been created (too late).
            command[command.index("--out") + 1] = str(directory / "case_Y.txt")
            command[command.index("--csv") + 1] = str(directory / "CASE_Y.TXT")
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert rejected.returncode != 0 and "paths must differ" in rejected.stderr
            assert not (directory / "case_Y.txt").exists()
    print(f"PASS: {args.fixture}, {args.mode}, {args.kernel}, {args.precision}, four seeds")


if __name__ == "__main__":
    main()
