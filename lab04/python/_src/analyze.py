#!/usr/bin/env python3
"""Wykresy mini-raportu dla jednego przebiegu (baseline lub sleep)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Patch

PHI = (5**0.5 - 1) / 2


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "data.npz")
    return meta, data["image"], data["ranks"], data["times"], data["slept"]


def worker_ranks_sorted(ranks: np.ndarray) -> list[int]:
    return sorted(int(r) for r in np.unique(ranks))


def rank_to_proc_index(ranks: np.ndarray, wr: list[int]) -> np.ndarray:
    m = {r: i for i, r in enumerate(wr)}
    return np.array([m[int(r)] for r in ranks], dtype=np.int32)


def palette(n: int) -> np.ndarray:
    if n <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n)
        return cmap(np.arange(n))[:, :3]
    colors = []
    for i in range(n):
        hue = (i * PHI) % 1.0
        colors.append(hsv_to_rgb([hue, 0.85, 0.95]))
    return np.array(colors)


def assignment_metrics(ranks: np.ndarray, proc_idx: np.ndarray, n_workers: int) -> dict:
    h = len(ranks)
    p = n_workers
    if h < 2 or p < 1:
        return {
            "B": 0.0,
            "C": 0.0,
            "B_norm": 0.0,
            "C_norm": 0.0,
            "num_segments": 0,
        }

    same_next = proc_idx[:-1] == proc_idx[1:]
    b = float(np.mean(same_next))

    if h > p:
        same_p = proc_idx[:-p] == proc_idx[p:]
        c = float(np.mean(same_p))
    else:
        c = 0.0

    rand = 1.0 / p
    if p > 1:
        b_norm = (b - rand) / (1.0 - rand)
        c_norm = (c - rand) / (1.0 - rand)
    else:
        b_norm = 0.0
        c_norm = 0.0

    segments = 1
    for i in range(1, h):
        if proc_idx[i] != proc_idx[i - 1]:
            segments += 1

    return {
        "B": b,
        "C": c,
        "B_norm": b_norm,
        "C_norm": c_norm,
        "num_segments": segments,
        "n_workers": p,
        "n_lines": h,
    }


def times_for_hist(times: np.ndarray, slept: np.ndarray, sleep_line: int | None):
    mask = np.ones(len(times), dtype=bool)
    if sleep_line is not None and 0 <= sleep_line < len(times):
        if slept[sleep_line] > 0:
            mask[sleep_line] = False
    return times[mask], times


def plot_hist(run_dir: Path, times: np.ndarray, slept: np.ndarray, meta: dict):
    sleep_line = meta.get("sleep_line")
    hist_times, all_times = times_for_hist(times, slept, sleep_line)
    tmin_i = int(np.argmin(all_times))
    tmax_i = int(np.argmax(all_times))

    if len(hist_times) < 2:
        edges = np.linspace(0, max(all_times.max(), 1e-9), 21)
    else:
        edges = np.histogram_bin_edges(hist_times, bins="fd")
        n_bins = len(edges) - 1
        if n_bins < 20:
            edges = np.histogram_bin_edges(hist_times, bins=max(20, n_bins))
        elif n_bins > 100:
            edges = np.histogram_bin_edges(hist_times, bins=100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hist_times, bins=edges, edgecolor="0.3", color="steelblue")
    ax.set_xlabel("Czas linii [s]")
    ax.set_ylabel("Liczba linii")

    diff = float(all_times.max() - all_times.min())
    stats = (
        f"min: {all_times.min():.6f} s (linia {tmin_i})\n"
        f"max: {all_times.max():.6f} s (linia {tmax_i})\n"
        f"max-min: {diff:.6f} s\n"
        f"mediana: {np.median(all_times):.6f} s"
    )
    if sleep_line is not None and meta.get("variant") == "sleep":
        stats += f"\n(sleep linia {sleep_line}: {times[sleep_line]:.6f} s)"

    fig.text(0.98, 0.5, stats, va="center", ha="right", fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(run_dir / "hist_line_time.png", dpi=150)
    plt.close(fig)

    return {
        "time_min": float(all_times.min()),
        "time_max": float(all_times.max()),
        "time_diff": diff,
        "time_min_line": tmin_i,
        "time_max_line": tmax_i,
        "time_median": float(np.median(all_times)),
    }


def build_assignment_rgb(h: int, w: int, proc_idx: np.ndarray, colors: np.ndarray) -> np.ndarray:
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(h):
        rgb[k, :, :] = colors[proc_idx[k]]
    return rgb


def plot_assignment_maps(
    run_dir: Path,
    image: np.ndarray,
    proc_idx: np.ndarray,
    wr: list[int],
    colors: np.ndarray,
    metrics: dict,
):
    h, w = image.shape
    assign = build_assignment_rgb(h, w, proc_idx, colors)
    overlay = assign * (image.astype(np.float32) / 255.0)[:, :, np.newaxis]

    plt.imsave(run_dir / "assignment_map.png", assign, origin="upper")
    plt.imsave(run_dir / "assignment_overlay.png", np.clip(overlay, 0, 1), origin="upper")

    fig, ax = plt.subplots(figsize=(min(10, w / 80), min(8, h / 80)))
    ax.imshow(assign, origin="upper", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    patches = [
        Patch(facecolor=colors[i], label=f"proc{i + 1} (rank {wr[i]})") for i in range(len(wr))
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    box = (
        f"B (sąsiednie): {metrics['B']:.4f}  B_norm: {metrics['B_norm']:.4f}\n"
        f"C (odstęp P): {metrics['C']:.4f}  C_norm: {metrics['C_norm']:.4f}\n"
        f"Segmenty: {metrics['num_segments']}  (blok ~P={metrics['n_workers']}, cykl ~H)"
    )
    fig.text(0.02, 0.02, box, fontsize=8, family="monospace", va="bottom")
    fig.tight_layout()
    fig.savefig(run_dir / "assignment_map_annotated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_julia(run_dir: Path, image: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "julia.png", dpi=150)
    plt.close(fig)


def count_and_sum_per_worker(
    proc_idx: np.ndarray, times: np.ndarray, n_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(n_workers, dtype=np.int32)
    sums = np.zeros(n_workers, dtype=np.float64)
    for i, t in zip(proc_idx, times, strict=True):
        counts[i] += 1
        sums[i] += t
    return counts, sums


def plot_lines_per_worker(run_dir: Path, counts: np.ndarray, h: int):
    n = len(counts)
    labels = [f"proc{i + 1}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), 4))
    ax.bar(labels, counts, color="steelblue")
    ax.axhline(h / n, color="gray", linestyle="--", label=f"średnia {h / n:.1f}")
    ax.set_ylabel("Liczba linii")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "lines_per_worker.png", dpi=150)
    plt.close(fig)


def plot_time_per_worker(
    run_dir: Path,
    sums: np.ndarray,
    colors: np.ndarray,
    wall: float,
):
    n = len(sums)
    max_busy = float(sums.max())
    overhead = max(0.0, wall - max_busy)

    fig, ax = plt.subplots(figsize=(max(7, (n + 1) * 0.55), 4))
    ax.bar(0, max_busy, color="#666666", label="max worker busy")
    ax.bar(0, overhead, bottom=max_busy, color="#cccccc", label="narzut równoległości")
    if max_busy > 0:
        ax.text(0, max_busy / 2, f"{max_busy:.4f}s", ha="center", va="center", fontsize=8)
    if overhead > 0:
        ax.text(
            0,
            max_busy + overhead / 2,
            f"{overhead:.4f}s",
            ha="center",
            va="center",
            fontsize=8,
        )

    xs = np.arange(1, n + 1)
    ax.bar(xs, sums, color=colors[:n])
    labels = ["wall"] + [f"proc{i + 1}" for i in range(n)]
    ax.set_xticks(np.arange(0, n + 1), labels)
    ax.axhline(wall, color="red", linestyle="--", linewidth=1.5, label=f"wall {wall:.4f}s")
    ax.set_ylabel("Czas [s]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "time_per_worker.png", dpi=150)
    plt.close(fig)


def analyze(run_dir: Path) -> dict:
    meta, image, ranks, times, slept = load_run(run_dir)
    h, w = image.shape
    wr = worker_ranks_sorted(ranks)
    n_workers = len(wr)
    proc_idx = rank_to_proc_index(ranks, wr)
    colors = palette(n_workers)

    metrics_assign = assignment_metrics(ranks, proc_idx, n_workers)
    hist_stats = plot_hist(run_dir, times, slept, meta)
    plot_assignment_maps(run_dir, image, proc_idx, wr, colors, metrics_assign)
    plot_julia(run_dir, image)

    counts, sums = count_and_sum_per_worker(proc_idx, times, n_workers)
    plot_lines_per_worker(run_dir, counts, h)
    wall = float(meta["wall_time"])
    plot_time_per_worker(run_dir, sums, colors, wall)

    out = {
        **meta,
        **metrics_assign,
        **hist_stats,
        "lines_per_worker": counts.tolist(),
        "time_per_worker": [float(x) for x in sums],
        "worker_ranks": wr,
        "max_worker_busy": float(sums.max()),
        "parallel_overhead": max(0.0, wall - float(sums.max())),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="katalog z data.npz i meta.json (np. results/runs/.../baseline)",
    )
    args = parser.parse_args()
    if not (args.run_dir / "data.npz").is_file():
        raise SystemExit(f"Brak data.npz w {args.run_dir}")
    analyze(args.run_dir)
    print(f"Wykresy w {args.run_dir}")


if __name__ == "__main__":
    main()
