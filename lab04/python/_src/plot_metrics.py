#!/usr/bin/env python3
"""Wykresy S(P), E(P), f(P) z pliku results/bench.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_median_times(csv_path: Path):
    """Grupuje powtórzenia i zwraca medianę czasu dla (version, n, iters, p)."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["version"],
                int(row["n"]),
                int(row["iters"]),
                int(row["p"]),
            )
            buckets[key].append(float(row["time"]))
    medians: dict[tuple, float] = {}
    for key, times in buckets.items():
        medians[key] = float(np.median(times))
    return medians


def metrics_for_curve(p_values: list[int], times: dict[int, float]):
    t1 = times[1]
    s_vals = []
    e_vals = []
    f_vals = []
    p_plot = []
    for p in sorted(p_values):
        if p not in times:
            continue
        s = t1 / times[p]
        e = s / p
        s_vals.append(s)
        e_vals.append(e)
        p_plot.append(p)
        if p > 1:
            f = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
        else:
            f = float("nan")
        f_vals.append(f)
    return p_plot, s_vals, e_vals, f_vals


def plot_version(medians, version: str, out_dir: Path):
    by_n: dict[int, dict[int, float]] = defaultdict(dict)
    iters_by_n: dict[int, int] = {}
    for (ver, n, iters, p), t in medians.items():
        if ver != version:
            continue
        by_n[n][p] = t
        iters_by_n[n] = iters

    if not by_n:
        print(f"Brak danych dla wersji {version}")
        return

    ns_sorted = sorted(by_n.keys())
    all_p = sorted({p for times in by_n.values() for p in times})

    fig_specs = [
        ("speedup", "Przyspieszenie S(P)", "S(P)", True),
        ("efficiency", "Efektywność E(P)", "E(P)", False),
        ("karp_flatt", "Metryka Karpa-Flatta f(P)", "f(P)", False),
    ]

    for slug, title_y, ylabel, draw_ideal in fig_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in ns_sorted:
            times = by_n[n]
            if 1 not in times:
                continue
            p_plot, s_vals, e_vals, f_vals = metrics_for_curve(list(times.keys()), times)
            if slug == "speedup":
                y = s_vals
            elif slug == "efficiency":
                y = e_vals
            else:
                y = f_vals
            iters = iters_by_n[n]
            ax.plot(
                p_plot,
                y,
                marker="o",
                label=f"N = {n}, iters = {iters}",
            )

        if draw_ideal:
            ax.plot(all_p, all_p, "k--", alpha=0.5, label="idealne S = P")

        if slug == "efficiency":
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="idealne E = 1")

        ax.set_xlabel("Liczba procesów P")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_y} — wersja {version}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_p)
        fig.tight_layout()
        path = out_dir / f"{version}_{slug}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Zapisano {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bench.csv"),
        help="plik z wynikami benchmarku",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmark_results"),
        help="katalog na wykresy PNG",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Brak pliku {args.csv}. Uruchom local/bench.sh.")

    args.out.mkdir(parents=True, exist_ok=True)
    medians = load_median_times(args.csv)
    versions = sorted({k[0] for k in medians})
    for ver in versions:
        plot_version(medians, ver, args.out)


if __name__ == "__main__":
    main()
