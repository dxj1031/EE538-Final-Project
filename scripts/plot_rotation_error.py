#!/usr/bin/env python3
"""Qinyan Zeng: direct vs rotated error plots from a reference-checked sweep.

One blocked tile-32 row per data seed enters each median. Scalar and tuning rows
are excluded, preventing repeated copies of the same case from biasing results.
Writes four PNG/PDF plots per scale precision, median CSV and a Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def aggregate(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple, dict[int, tuple[float, float]]] = {}
    for row in rows:
        if row["mode"] not in ("direct", "rotated") or row["kernel"] != "blocked" or row["tile"] != "32":
            continue
        if row.get("np_metrics_agree") != "1" or row.get("y_match_scalar") != "1":
            raise ValueError("rotation plots require reference-checked, scalar-matching rows")
        precision = row["scale_precision"]
        if precision not in ("float", "double"):
            raise ValueError("unknown scale precision")
        key = (precision, int(row["K"]), int(row["N"]), int(row["M"]), float(row["a"]), row["mode"])
        seed = int(row["data_seed"])
        errors = (float(row["rel_frob_err"]), float(row["max_abs_err"]))
        if any(not math.isfinite(value) or value < 0 for value in errors):
            raise ValueError("nonfinite or negative error metric")
        by_seed = groups.setdefault(key, {})
        if seed in by_seed:
            raise ValueError(f"duplicate seed in a primary configuration: {key}, {seed}")
        by_seed[seed] = errors
    if not groups:
        raise ValueError("no tile-32 direct/rotated data")
    result = []
    seed_sets: dict[str, set[int]] = {}
    for key, by_seed in sorted(groups.items()):
        precision, k, n, m, a, mode = key
        expected_seeds = seed_sets.setdefault(precision, set(by_seed))
        if set(by_seed) != expected_seeds:
            raise ValueError(f"incomplete seed set in {key}")
        other = key[:-1] + (("rotated" if mode == "direct" else "direct"),)
        if other not in groups or set(groups[other]) != set(by_seed):
            raise ValueError(f"missing paired mode/seeds for {key}")
        result.append(dict(scale_precision=precision, K=k, N=n, M=m, a=a, mode=mode,
                           data_seeds=";".join(str(seed) for seed in sorted(by_seed)),
                           seed_count=len(by_seed),
                           rel_frob_err=statistics.median(value[0] for value in by_seed.values()),
                           max_abs_err=statistics.median(value[1] for value in by_seed.values())))
    return result


def plot(medians: list[dict], out: Path, precision: str, metric: str, sweep: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    rows = [r for r in medians if r["scale_precision"] == precision and
            (r["M"] == 8 if sweep == "a" else r["a"] == 10)]
    scales = sorted({(r["K"], r["N"]) for r in rows})
    if not scales:
        raise ValueError(f"missing {sweep} sweep for {precision}")
    fig, axes = plt.subplots(1, len(scales), figsize=(3.8 * len(scales), 3.7), squeeze=False)
    colors = {"direct": "#df6834", "rotated": "#138a63"}
    labels = {"direct": "Direct W4A4", "rotated": "Rotated W4A4"}
    ylabel = "relative Frobenius error" if metric == "rel_frob_err" else "maximum absolute error"
    xlabel = "outlier factor a" if sweep == "a" else "batch rows M"
    for ax, (k, n) in zip(axes[0], scales):
        panel = [r for r in rows if r["K"] == k and r["N"] == n]
        for mode in ("direct", "rotated"):
            values = sorted((r[sweep], r[metric]) for r in panel if r["mode"] == mode)
            ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", markersize=5,
                    linewidth=1.8, color=colors[mode], label=labels[mode])
        ax.set_xscale("log", base=10 if sweep == "a" else 2)
        ticks = sorted({r[sweep] for r in panel})
        ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
        ax.set_ylim(bottom=0)
        ax.set_title(f"K = {k}, N = {n}", loc="left", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    seeds = rows[0]["data_seeds"].replace(";", ", ")
    fixed = "M = 8" if sweep == "a" else "a = 10"
    fig.suptitle(f"{ylabel.capitalize()} vs {xlabel} ({precision} scales)", x=0.03, ha="left", fontsize=13)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.99, 0.91), ncols=2,
               frameon=False, fontsize=9)
    fig.text(0.03, 0.87, f"Median across data seeds {seeds}; {fixed}; blocked kernel, T = 32", fontsize=9,
             color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.82))
    stem = out / f"rotation_{metric}_vs_{sweep}_{precision}"
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"plot_rotation_error: wrote {stem.name}.png/.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="") as handle:
        medians = aggregate(list(csv.DictReader(handle)))
    out = args.out_dir or args.csv.parent
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rotation_error_medians.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(medians[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(medians)
    for precision in sorted({r["scale_precision"] for r in medians}):
        for metric in ("rel_frob_err", "max_abs_err"):
            for sweep in ("a", "M"):
                plot(medians, out, precision, metric, sweep)
    pairs = {(r["scale_precision"], r["K"], r["N"], r["M"], r["a"]): {} for r in medians}
    for row in medians:
        pairs[(row["scale_precision"], row["K"], row["N"], row["M"], row["a"])][row["mode"]] = row
    lines = ["# Rotation error results", "",
             f"Source: `{args.csv.name}`. Each value is the median across data seeds, using one",
             "blocked tile-32 result per seed. Relative and absolute errors are summarized separately.", "",
             "| scales | K | N | M | a | seeds | relative direct | relative rotated | max abs direct | max abs rotated |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for key, pair in sorted(pairs.items()):
        precision, k, n, m, a = key
        direct, rotated = pair["direct"], pair["rotated"]
        lines.append(f"| {precision} | {k} | {n} | {m} | {a:g} | {direct['data_seeds']} | "
                     f"{direct['rel_frob_err']:.6f} | {rotated['rel_frob_err']:.6f} | "
                     f"{direct['max_abs_err']:.6g} | {rotated['max_abs_err']:.6g} |")
    lines += ["", "At a=100, outlier contributions dominate the FP64 output norm. A lower relative",
              "error alone does not imply a lower maximum absolute error; compare both metrics.", ""]
    (out / "rotation_error_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"plot_rotation_error: wrote medians and summary to {out}")


if __name__ == "__main__":
    main()
