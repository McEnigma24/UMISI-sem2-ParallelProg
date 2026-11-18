#!/usr/bin/env python3
"""Mini-raport dla przebiegu Langevin (katalog RUN z N*/rep*)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_density(rep_dir: Path, data: dict, meta: dict) -> None:
    counts = data["hist_counts"]
    edges = data["hist_edges"]
    tau = float(data["tau"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    empirical = counts / max(counts.sum(), 1)

    analytic = (
        np.exp(-0.5 * centers * centers / tau)
        / np.sqrt(2.0 * np.pi * tau)
        * width
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, empirical, width=width * 0.95, alpha=0.6, label="GPU histogram", color="steelblue")
    ax.plot(centers, analytic, "r-", lw=2, label=rf"$\mathcal{{N}}(0,{tau})$")
    ax.set_xlabel("x(τ)")
    ax.set_ylabel("Prawdopodobieństwo (bin)")
    ax.legend()
    ax.set_title(
        f"mean={meta.get('mean', 0):.4f} var={meta.get('variance', 0):.4f} "
        f"L1={meta.get('l1_error', 0):.4f}"
    )
    fig.tight_layout()
    fig.savefig(rep_dir / "density.png", dpi=150)
    plt.close(fig)


def plot_mfpt(rep_dir: Path, data: dict, meta: dict) -> None:
    sample = np.asarray(data["exit_times_sample"]).ravel()
    sample = sample[sample > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(sample) > 1:
        ax.hist(sample, bins=60, density=True, alpha=0.7, color="darkorange", edgecolor="0.3")
    ax.axvline(1.0, color="red", linestyle="--", label="analytic ⟨τ⟩=1")
    ax.axvline(meta.get("mean_exit_time", 0), color="blue", linestyle="-", label="estimated mean")
    ax.set_xlabel("exit time τ")
    ax.set_ylabel("gęstość")
    ax.legend()
    ax.set_title(
        f"⟨τ⟩={meta.get('mean_exit_time', 0):.4f}±{meta.get('std_exit_time', 0):.4f}/√N "
        f"censored={meta.get('n_censored', 0)}"
    )
    fig.tight_layout()
    fig.savefig(rep_dir / "exit_times.png", dpi=150)
    plt.close(fig)


def plot_timings(rep_dir: Path, meta: dict) -> None:
    keys = ["t_alloc", "t_kernel_sim", "t_kernel_post", "t_d2h"]
    vals = [float(meta.get(k, 0)) for k in keys]
    labels = ["alloc", "sim", "post", "d2h"]
    other = max(0.0, float(meta.get("t_total", 0)) - sum(vals))
    vals.append(other)
    labels.append("other")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"])
    ax.set_ylabel("czas [s]")
    ax.set_title(f"t_total={meta.get('t_total', 0):.4f}s n={meta.get('n', 0)}")
    fig.tight_layout()
    fig.savefig(rep_dir / "timings.png", dpi=150)
    plt.close(fig)


def plot_time_vs_n(run_dir: Path) -> None:
    points: list[tuple[int, float]] = []
    for meta_path in sorted(run_dir.glob("N*/rep*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        points.append((int(meta["n"]), float(meta["t_total"])))
    if not points:
        return
    by_n: dict[int, list[float]] = {}
    for n, t in points:
        by_n.setdefault(n, []).append(t)
    ns = sorted(by_n)
    means = [float(np.mean(by_n[n])) for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ns, means, "o-", color="steelblue")
    ax.set_xlabel("liczba trajektorii N")
    ax.set_ylabel("t_total [s]")
    ax.set_title("Czas programu vs N")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "time_vs_n.png", dpi=150)
    plt.close(fig)


def analyze_run(run_dir: Path) -> None:
    for rep_dir in sorted(run_dir.glob("N*/rep*")):
        meta_path = rep_dir / "meta.json"
        data_path = rep_dir / "data.npz"
        if not meta_path.is_file() or not data_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = dict(np.load(data_path, allow_pickle=True))
        problem = meta.get("problem", "density")
        if problem == "density":
            plot_density(rep_dir, data, meta)
        else:
            plot_mfpt(rep_dir, data, meta)
        plot_timings(rep_dir, meta)
    plot_time_vs_n(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analyze_run(args.run_dir)
    print(f"Wykresy w {args.run_dir}")


if __name__ == "__main__":
    main()