#!/usr/bin/env python3
"""Mini-raport dla jednego przebiegu TSP (katalog rep z meta.json)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
PHI = (5**0.5 - 1) / 2


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "data.npz", allow_pickle=True)
    return meta, data


def worker_ranks_sorted(ranks: np.ndarray) -> list[int]:
    return sorted(int(r) for r in np.unique(ranks))


def palette(n: int) -> np.ndarray:
    if n <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n)
        return cmap(np.arange(n))[:, :3]
    colors = []
    for i in range(n):
        hue = (i * PHI) % 1.0
        from matplotlib.colors import hsv_to_rgb

        colors.append(hsv_to_rgb([hue, 0.85, 0.95]))
    return np.array(colors)


def plot_hist_task_time(run_dir: Path, times: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(times) < 2:
        edges = np.linspace(0, max(float(times.max()), 1e-9), 21)
    else:
        edges = np.histogram_bin_edges(times, bins="fd")
        nb = len(edges) - 1
        if nb < 10:
            edges = np.linspace(times.min(), times.max(), 11)
        elif nb > 50:
            edges = np.histogram_bin_edges(times, bins=50)
    ax.hist(times, bins=edges, edgecolor="0.3", color="steelblue")
    ax.set_xlabel("Czas zadania [s]")
    ax.set_ylabel("Liczba zadań")
    stats = (
        f"min: {times.min():.6f} s\n"
        f"max: {times.max():.6f} s\n"
        f"mediana: {np.median(times):.6f} s\n"
        f"suma: {times.sum():.6f} s"
    )
    fig.text(0.98, 0.5, stats, va="center", ha="right", fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(run_dir / "hist_task_time.png", dpi=150)
    plt.close(fig)


def plot_nodes_per_task(run_dir: Path, nodes: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(nodes))
    ax.bar(xs, nodes, color="darkorange", width=0.8)
    ax.set_xlabel("ID zadania")
    ax.set_ylabel("Węzły przeszukiwane")
    ax.set_title(f"suma węzłów: {int(nodes.sum())}")
    fig.tight_layout()
    fig.savefig(run_dir / "nodes_per_task.png", dpi=150)
    plt.close(fig)


def count_and_sum_per_worker(ranks: np.ndarray, times: np.ndarray, wr: list[int]):
    n_workers = len(wr)
    m = {r: i for i, r in enumerate(wr)}
    counts = np.zeros(n_workers, dtype=np.int32)
    sums = np.zeros(n_workers, dtype=np.float64)
    for r, t in zip(ranks, times, strict=True):
        i = m[int(r)]
        counts[i] += 1
        sums[i] += float(t)
    return counts, sums


def plot_tasks_per_worker(run_dir: Path, counts: np.ndarray, n_tasks: int):
    n = len(counts)
    labels = [f"proc{i + 1}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), 4))
    ax.bar(labels, counts, color="steelblue")
    ax.axhline(n_tasks / max(n, 1), color="gray", linestyle="--", label="średnia")
    ax.set_ylabel("Liczba zadań (prefiksów)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "tasks_per_worker.png", dpi=150)
    plt.close(fig)


def plot_time_per_worker(run_dir: Path, sums: np.ndarray, colors: np.ndarray, wall: float):
    n = len(sums)
    max_busy = float(sums.max()) if n else 0.0
    overhead = max(0.0, wall - max_busy)

    fig, ax = plt.subplots(figsize=(max(7, (n + 1) * 0.55), 4))
    ax.bar(0, max_busy, color="#666666", label="max worker busy")
    ax.bar(0, overhead, bottom=max_busy, color="#cccccc", label="narzut równoległości")
    xs = np.arange(1, n + 1)
    if n:
        ax.bar(xs, sums, color=colors[:n])
    labels = ["wall"] + [f"proc{i + 1}" for i in range(n)]
    ax.set_xticks(np.arange(0, n + 1), labels)
    ax.axhline(wall, color="red", linestyle="--", linewidth=1.5, label=f"wall {wall:.4f}s")
    ax.set_ylabel("Czas [s]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "time_per_worker.png", dpi=150)
    plt.close(fig)


def plot_tour(run_dir: Path, coords: np.ndarray, tour: np.ndarray, meta: dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=40, zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.text(x * 1.08, y * 1.08, str(i), fontsize=8)

    if tour.size >= 2:
        closed = np.append(tour, tour[0])
        path = coords[closed]
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1.5, alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(
        f"tour cost={meta.get('best_cost', 0):.4f} "
        f"(opt={meta.get('optimal_known', 0):.4f})"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "tour.png", dpi=150)
    plt.close(fig)


def analyze(run_dir: Path) -> dict:
    meta, data = load_run(run_dir)
    times = data["times"].astype(np.float64)
    ranks = data["ranks"].astype(np.int32)
    nodes = data["node_counts"].astype(np.int64)
    coords = data["coords"]
    tour = data["best_tour"]

    wr = worker_ranks_sorted(ranks)
    colors = palette(len(wr))

    plot_hist_task_time(run_dir, times)
    plot_nodes_per_task(run_dir, nodes)
    counts, sums = count_and_sum_per_worker(ranks, times, wr)
    plot_tasks_per_worker(run_dir, counts, len(times))
    wall = float(meta["wall_time"])
    plot_time_per_worker(run_dir, sums, colors, wall)
    plot_tour(run_dir, coords, tour, meta)

    out = {
        **meta,
        "tasks_per_worker": counts.tolist(),
        "time_per_worker": [float(x) for x in sums],
        "total_nodes": int(nodes.sum()),
        "worker_ranks": wr,
        "max_worker_busy": float(sums.max()) if len(sums) else 0.0,
        "parallel_overhead": max(0.0, wall - float(sums.max())) if len(sums) else wall,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="katalog rep z data.npz i meta.json")
    args = parser.parse_args()
    if not (args.run_dir / "data.npz").is_file():
        raise SystemExit(f"Brak data.npz w {args.run_dir}")
    analyze(args.run_dir)
    print(f"Wykresy w {args.run_dir}")


if __name__ == "__main__":
    main()
