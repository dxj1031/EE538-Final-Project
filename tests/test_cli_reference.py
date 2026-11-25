#!/usr/bin/env python3
"""Independent NumPy/dense-Q checks of nontrivial file-to-file CLI outputs.

The Python MT19937-64 sign stream and dense Sylvester matrix share no production
C++ code. Dyadic inputs isolate quantization from decimal round-trip noise.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reference_check import quantize


class MT64:
    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.state = [seed]
        for i in range(1, 312):
            last = self.state[-1]
            self.state.append((6364136223846793005 * (last ^ (last >> 62)) + i) & self.MASK)
        self.index = 312

    def next(self):
        if self.index == 312:
            for i in range(312):
                combined = (self.state[i] & 0xFFFFFFFF80000000) | (self.state[(i + 1) % 312] & 0x7FFFFFFF)
                self.state[i] = self.state[(i + 156) % 312] ^ (combined >> 1)
                if combined & 1:
                    self.state[i] ^= 0xB5026F5AA96619E9
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        value ^= value >> 43
        return value & self.MASK


def dense_q(k, seed):
    engine = MT64(seed)
    signs = np.array([-1 if engine.next() >> 63 else 1 for _ in range(k)], dtype=np.float64)
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < k:
        h = np.block([[h, h], [h, -h]])
    return signs[:, None] * h / np.sqrt(k)


def write_matrix(path, matrix):
    with path.open("w") as handle:
        handle.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        np.savetxt(handle, matrix, fmt="%.9g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    args = parser.parse_args()
    # Standard MT19937-64 first output with its default seed.
    assert MT64(5489).next() == 14514284786278117030
    generator = np.random.default_rng(538)
    checked = 0
    with tempfile.TemporaryDirectory(prefix="int4 independent ") as temporary:
        directory = Path(temporary)
        for k in (1, 2, 64):
            x = (generator.integers(-80, 81, size=(3, k)) / 16).astype(np.float32)
            w = (generator.integers(-80, 81, size=(k, 33)) / 16).astype(np.float32)
            write_matrix(directory / "X.txt", x)
            write_matrix(directory / "W.txt", w)
            for seed in (0, 7, 2**64 - 1):
                q = dense_q(k, seed)
                assert np.allclose(q.T @ q, np.eye(k), rtol=0, atol=1e-12)
                for mode in ("direct", "rotated"):
                    xp = x if mode == "direct" else (x.astype(np.float64) @ q).astype(np.float32)
                    wp = w if mode == "direct" else (q.T @ w.astype(np.float64)).astype(np.float32)
                    for precision, dtype in (("float", np.float32), ("double", np.float64)):
                        cx, sx = quantize(xp, "row", dtype)
                        cw, sw = quantize(wp, "col", dtype)
                        reference = (((cx @ cw).astype(np.float64) * sx.astype(np.float64)[:, None]) *
                                     sw.astype(np.float64)[None, :]).astype(np.float32)
                        for kernel in ("scalar", "blocked"):
                            result = subprocess.run([str(args.exe), "--x", str(directory / "X.txt"),
                                "--w", str(directory / "W.txt"), "--out", str(directory / "Y.txt"),
                                "--csv", str(directory / "timings.csv"), "--mode", mode,
                                "--kernel", kernel, "--seed", str(seed), "--scale-precision", precision,
                                "--warmups", "0", "--runs", "1"],
                                text=True, capture_output=True, timeout=60)
                            if result.returncode:
                                raise AssertionError(result.stderr)
                            tokens = (directory / "Y.txt").read_text().split()
                            assert tokens[:2] == ["3", "33"]
                            actual = np.array(tokens[2:], dtype=np.float32).reshape(3, 33)
                            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6,
                                err_msg=f"K={k}, seed={seed}, {mode}, {precision}, {kernel}")
                            checked += 1
    print(f"PASS: {checked} nontrivial CLI outputs match independent NumPy quantization/dense-Q references")


if __name__ == "__main__":
    main()
