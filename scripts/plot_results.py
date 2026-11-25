#!/usr/bin/env python3
"""Wooyoung Kim: timing plots and summary tables from int4_bench results.

Usage:
    python3 scripts/plot_results.py RESULTS/bench_with_reference.csv [--out-dir RESULTS]

Every value is first summarized per seed by the C++ driver (median of the timed runs);
this script then takes the median across data seeds. Outputs (in --out-dir):

    latency_by_mode.png          input-call latency vs K (M = 8, a = 10)
    kernel_scalar_vs_blocked.png kernel-only time, scalar vs blocked (M = 8, a = 10)
    weight_prep.png              weight-preparation time vs K
    tile_edge.png                blocked kernel time by tile edge (only if tuning rows exist)
    summary.md                   error, timing and storage tables

Rotation-error plots are Member 2's deliverable; the error table here is a summary.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated reference palette (dataviz skill), light mode; fixed slot per entity.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
MODE_COLOR = {"fp32": SLOT[0], "direct": SLOT[1], "rotated": SLOT[2]}
MODE_LABEL = {"fp32": "FP32", "direct": "Direct W4A4", "rotated": "Rotated W4A4"}
KERNEL_COLOR = {"blocked": SLOT[0], "scalar": SLOT[1]}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"plot_results: {path} has no rows")
    return rows


def median_over_seeds(rows, key, field):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(float(row[field]))
    return {k: statistics.median(v) for k, v in groups.items()}


def is_main(row) -> bool:
    """FP32 rows and tile-32 blocked rows (the protocol's primary configuration)."""
    return row["kernel"] == "fp32" or (row["kernel"] == "blocked" and row["tile"] == "32")


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelsize=8, labelcolor=INK_2)


def finish(fig, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"plot_results: wrote {path}")


def line_chart(series, title, ylabel, path, note):
    """series: list of (label, color, xs, ys). Log-log axes, legend + end labels."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style(ax, title, "K = N", ylabel)
    for label, color, xs, ys in series:
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    xs_all = sorted({x for _, _, xs, _ in series for x in xs})
    ax.set_xticks(xs_all, [str(x) for x in xs_all])
    ax.set_xlim(xs_all[0] / 1.25, xs_all[-1] * 2.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    # End-of-line labels, pushed apart vertically so near-coincident lines stay legible.
    fig.canvas.draw()
    ends = sorted(((ax.transData.transform((xs[-1], ys[-1]))[1], label, xs[-1], ys[-1])
                   for label, _, xs, ys in series))
    gap_px = 11 * fig.dpi / 72
    placed = []
    for y_px, label, x, y in ends:
        target = max(y_px, placed[-1] + gap_px) if placed else y_px
        placed.append(target)
        ax.annotate(label, (x, y), xytext=(8, (target - y_px) * 72 / fig.dpi), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
    finish(fig, path)


def fmt_us(ns: float) -> str:
    return f"{ns / 1000:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or args.csv.parent
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.csv)
    seeds = sorted({r["data_seed"] for r in rows}, key=int)
    note = f"Median across data seeds {', '.join(seeds)} of per-seed medians; one thread; INT4 rows use the blocked kernel, T = 32."

    protocol = [r for r in rows if r["M"] == "8" and r["a"] == "10"]
    scales = sorted({int(r["K"]) for r in rows})

    # 1. Input-call latency by mode.
    call = median_over_seeds([r for r in protocol if is_main(r)], lambda r: (r["mode"], int(r["K"])),
                             "call_median_ns")
    series = []
    for mode in ("fp32", "direct", "rotated"):
        xs = [k for k in scales if (mode, k) in call]
        if xs:
            series.append((MODE_LABEL[mode], MODE_COLOR[mode], xs, [call[(mode, k)] / 1000 for k in xs]))
    line_chart(series, "Input-call latency (M = 8, a = 10)", "median latency (µs, log)",
               out / "latency_by_mode.png", note)

    # 2. Kernel-only time, scalar vs blocked, on identical packed operands.
    kernel_rows = [r for r in protocol if r["mode"] == "direct" and
                   (r["kernel"] == "scalar" or (r["kernel"] == "blocked" and r["tile"] == "32"))]
    kernel = median_over_seeds(kernel_rows, lambda r: (r["kernel"], int(r["K"])), "kernel_median_ns")
    series = []
    for name, label in (("blocked", "Blocked kernel (T = 32)"), ("scalar", "Scalar kernel")):
        xs = [k for k in scales if (name, k) in kernel]
        if xs:
            series.append((label, KERNEL_COLOR[name], xs, [kernel[(name, k)] / 1000 for k in xs]))
    if len(series) == 2:
        line_chart(series, "INT4 kernel time on identical operands (direct, M = 8, a = 10)",
                   "median kernel time (µs, log)", out / "kernel_scalar_vs_blocked.png",
                   "Kernel-only run(qx, qw), including operand validation and INT32 output allocation; "
                   "median across seeds.")

    # 3. Weight preparation.
    prep = median_over_seeds([r for r in protocol if is_main(r)], lambda r: (r["mode"], int(r["K"])),
                             "prep_median_ns")
    series = []
    for mode in ("fp32", "direct", "rotated"):
        xs = [k for k in scales if (mode, k) in prep]
        if xs:
            series.append((MODE_LABEL[mode], MODE_COLOR[mode], xs, [prep[(mode, k)] / 1000 for k in xs]))
    line_chart(series, "Weight preparation (once per weight)", "median time (µs, log)",
               out / "weight_prep.png", "FP32: validated copy. W4A4: (rotation +) per-column quantization "
               "and packing; median across seeds.")

    # 4. Optional tile-edge tuning (K = N = 512, M = 8, a = 10).
    tuning = [r for r in protocol if r["K"] == "512" and r["kernel"] == "blocked" and r["mode"] == "direct"]
    tiles = sorted({int(r["tile"]) for r in tuning})
    if len(tiles) > 1:
        tile_time = median_over_seeds(tuning, lambda r: int(r["tile"]), "kernel_median_ns")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        style(ax, "Blocked kernel by tile edge (direct, K = N = 512, M = 8)", "median kernel time (µs)", "")
        labels = [f"T = {t}" for t in tiles]
        values = [tile_time[t] / 1000 for t in tiles]
        bars = ax.barh(labels, values, color=SLOT[0], height=0.55)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=8, color=INK_2)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.grid(False, axis="y")
        ax.set_xlim(0, max(values) * 1.2)
        finish(fig, out / "tile_edge.png")

    # Summary tables.
    lines = [f"# Benchmark summary\n", f"Source: `{args.csv.name}`. {note}\n"]
    err = median_over_seeds([r for r in rows if is_main(r)],
                            lambda r: (int(r["K"]), int(r["M"]), float(r["a"]), r["mode"]), "rel_frob_err")
    mae = median_over_seeds([r for r in rows if is_main(r)],
                            lambda r: (int(r["K"]), int(r["M"]), float(r["a"]), r["mode"]), "max_abs_err")
    lat = median_over_seeds([r for r in rows if is_main(r)],
                            lambda r: (int(r["K"]), int(r["M"]), float(r["a"]), r["mode"]), "call_median_ns")
    lines.append("## Output error vs FP64 and input-call latency (all cases)\n")
    lines.append("| K = N | M | a | E_F FP32 | E_F direct | E_F rotated | max abs direct | max abs rotated "
                 "| call µs FP32 | call µs direct | call µs rotated |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in sorted({key[:3] for key in err}):
        k, m, a = case
        lines.append(f"| {k} | {m} | {a:g} | {err[case + ('fp32',)]:.2e} | {err[case + ('direct',)]:.4f} | "
                     f"{err[case + ('rotated',)]:.4f} | {mae[case + ('direct',)]:.3g} | "
                     f"{mae[case + ('rotated',)]:.3g} | {fmt_us(lat[case + ('fp32',)])} | "
                     f"{fmt_us(lat[case + ('direct',)])} | {fmt_us(lat[case + ('rotated',)])} |")

    lines.append("\n## Scalar vs blocked INT4 kernel (M = 8, a = 10, identical operands)\n")
    lines.append("| K = N | mode | scalar kernel µs | blocked kernel µs | scalar / blocked | scalar call µs "
                 "| blocked call µs | validation µs | sums identical |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    kt = median_over_seeds(protocol, lambda r: (int(r["K"]), r["mode"], r["kernel"], r["tile"]), "kernel_median_ns")
    ct = median_over_seeds(protocol, lambda r: (int(r["K"]), r["mode"], r["kernel"], r["tile"]), "call_median_ns")
    vt = median_over_seeds([r for r in protocol if r["mode"] != "fp32"], lambda r: (int(r["K"]), r["mode"]),
                           "validate_median_ns")
    for k in scales:
        for mode in ("direct", "rotated"):
            s, b = (k, mode, "scalar", ""), (k, mode, "blocked", "32")
            if s in kt and b in kt:
                same = all(r["sums_match_scalar"] == "1" for r in protocol if r["K"] == str(k) and r["mode"] == mode)
                lines.append(f"| {k} | {mode} | {fmt_us(kt[s])} | {fmt_us(kt[b])} | {kt[s] / kt[b]:.1f}× | "
                             f"{fmt_us(ct[s])} | {fmt_us(ct[b])} | {fmt_us(vt[(k, mode)])} | {'yes' if same else 'NO'} |")
    if len(tiles) > 1:
        lines.append("\n## Optional tile-edge tuning (K = N = 512, M = 8, a = 10)\n")
        lines.append("| mode | tile | kernel µs | call µs | scratch bytes |")
        lines.append("|---|---:|---:|---:|---:|")
        for mode in ("direct", "rotated"):
            for t in tiles:
                key = (512, mode, "blocked", str(t))
                scratch = next(r["scratch_bytes"] for r in protocol
                               if r["K"] == "512" and r["mode"] == mode and r["tile"] == str(t))
                lines.append(f"| {mode} | {t} | {fmt_us(kt[key])} | {fmt_us(ct[key])} | {scratch} |")

    lines.append("\n## Weight preparation and logical storage (M = 8)\n")
    lines.append("| K = N | mode | prep µs | resident bytes | prep peak bytes | call peak bytes | scratch bytes |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for k in scales:
        for mode in ("fp32", "direct", "rotated"):
            match = [r for r in protocol if r["K"] == str(k) and r["mode"] == mode and is_main(r)]
            if match:
                r0 = match[0]
                lines.append(f"| {k} | {mode} | {fmt_us(prep[(mode, k)])} | {r0['resident_bytes']} | "
                             f"{r0['prep_peak_bytes']} | {r0['call_peak_bytes']} | {r0['scratch_bytes']} |")
    if "np_metrics_agree" in rows[0]:
        agree = sum(r["np_metrics_agree"] == "1" for r in rows)
        lines.append(f"\nNumPy FP64 reference: {agree}/{len(rows)} rows agree with the C++ error metrics.")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"plot_results: wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
