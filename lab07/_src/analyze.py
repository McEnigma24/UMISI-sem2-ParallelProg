#!/usr/bin/env python3
"""Wykresy dla jednego przebiegu Ising."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "data.npz", allow_pickle=True)
    return meta, data


def plot_lattice(run_dir: Path, lattice: np.ndarray, meta: dict) -> None:
    if lattice.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lattice, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(
        f"L={meta['L']} beta={meta['beta']} sync={meta['sync']} "
        f"t={meta['steps']}  M={meta['magnetization']:.0f}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "lattice.png", dpi=150)
    plt.close(fig)


def plot_snapshots(run_dir: Path, data: np.lib.npyio.NpzFile) -> None:
    snaps = data["snapshots"]
    if snaps.size == 0:
        return
    ts = data["snapshot_ts"]
    snap_list = []
    for s in snaps.ravel():
        if s is None:
            continue
        a = np.asarray(s, dtype=np.int8)
        if a.ndim == 2 and a.size > 0:
            snap_list.append(a)
    if not snap_list:
        return
    snap_list = snap_list[-6:]
    n = len(snap_list)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    for k, (snap, t) in enumerate(zip(snap_list, ts[:n], strict=False)):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.imshow(snap, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"t={t}")
        ax.axis("off")
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    fig.suptitle("Migawki układu")
    fig.tight_layout()
    fig.savefig(run_dir / "snapshots.png", dpi=120)
    plt.close(fig)


def plot_energy_magnetization(run_dir: Path, data: np.lib.npyio.NpzFile, meta: dict) -> None:
    t = data["history_t"]
    if t.size == 0:
        return
    E = data["history_E"]
    M = data["history_M"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, E, "b-")
    ax1.set_ylabel("E")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, M, "r-")
    ax2.set_xlabel("MCS")
    ax2.set_ylabel("M")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"beta={meta['beta']} P={meta['workers']} sync={meta['sync']}")
    fig.tight_layout()
    fig.savefig(run_dir / "energy_magnetization.png", dpi=150)
    plt.close(fig)


def plot_time_breakdown(run_dir: Path, meta: dict) -> None:
    comp = float(meta.get("compute_time_rank0", 0))
    comm = float(meta.get("comm_time_rank0", 0))
    wall = float(meta["wall_time"])
    other = max(0.0, wall - comp - comm)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["compute", "MPI", "pozostałe"]
    vals = [comp, comm, other]
    ax.bar(labels, vals, color=["steelblue", "coral", "gray"])
    ax.set_ylabel("czas [s]")
    ax.set_title(f"wall={wall:.3f}s sync/MCS={meta.get('sync_per_mcs', 0):.1f}")
    fig.tight_layout()
    fig.savefig(run_dir / "time_breakdown.png", dpi=150)
    plt.close(fig)


def analyze(run_dir: Path) -> dict:
    meta, data = load_run(run_dir)
    lat = data["lattice"]
    if lat.ndim == 2:
        plot_lattice(run_dir, lat, meta)
    plot_snapshots(run_dir, data)
    plot_energy_magnetization(run_dir, data, meta)
    plot_time_breakdown(run_dir, meta)
    out = {**meta}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not (args.run_dir / "data.npz").is_file():
        raise SystemExit(f"Brak data.npz w {args.run_dir}")
    analyze(args.run_dir)
    print(f"Wykresy w {args.run_dir}")


if __name__ == "__main__":
    main()