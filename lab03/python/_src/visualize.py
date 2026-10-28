#!/usr/bin/env python3
"""Wizualizacja wyniku membrany: heatmapa, 3D, przekrój vs rozwiązanie analityczne."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

P = 500.0
T = 100.0
A = 50.0


def analytical_solution(x: np.ndarray, y: np.ndarray, a: float, terms: int = 40) -> np.ndarray:
    """Rozwiązanie szeregiem Fouriera dla -lap h = p/T, h=0 na brzegu kwadratu [0,a]^2."""
    h = np.zeros_like(x, dtype=float)
    coeff = 16.0 * P * a**2 / (T * np.pi**4)
    for m in range(1, terms, 2):
        for n in range(1, terms, 2):
            denom = m * n * (m**2 + n**2)
            h += coeff * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / a) / denom
    return h


def grid_coords(h: np.ndarray, a: float):
    n = h.shape[0]
    da = a / n
    a_eff = (n - 1) * da
    xs = np.arange(n) * da
    ys = np.arange(n) * da
    return xs, ys, da, a_eff


def load_or_raise(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"Brak pliku {path}")
    return np.load(path)


def plot_heatmap(h: np.ndarray, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(h, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("indeks x")
    ax.set_ylabel("indeks y")
    fig.colorbar(im, ax=ax, label="h")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_surface3d(h: np.ndarray, xs: np.ndarray, ys: np.ndarray, out: Path, title: str):
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, h, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("h")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cross_section(h: np.ndarray, xs: np.ndarray, a_eff: float, out: Path, title: str):
    n = h.shape[0]
    mid = n // 2
    y_mid = xs[mid]
    h_line = h[mid, :]
    x_phys = xs
    h_an = analytical_solution(x_phys, np.full_like(x_phys, y_mid), a_eff)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_phys, h_line, label="numeryczne (Jacobi)", linewidth=2)
    ax.plot(x_phys, h_an, "--", label="analityczne (szereg Fouriera)", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("h(x, y = const)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return float(np.max(np.abs(h_line - h_an)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loops",
        type=Path,
        default=Path("results/h_loops.npy"),
        help="wynik main.py",
    )
    parser.add_argument(
        "--vec",
        type=Path,
        default=Path("results/h_vec.npy"),
        help="wynik main_vec.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/correctness_visualizations"),
        help="katalog na PNG",
    )
    parser.add_argument(
        "--loops-p1",
        type=Path,
        default=None,
        help="opcjonalnie h z main.py przy P=1 (weryfikacja MPI)",
    )
    parser.add_argument(
        "--loops-p4",
        type=Path,
        default=None,
        help="opcjonalnie h z main.py przy P=4",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    h_loops = load_or_raise(args.loops)
    h_vec = load_or_raise(args.vec)

    diff_impl = float(np.max(np.abs(h_loops - h_vec)))
    print(f"max |h_loops - h_vec| = {diff_impl:.6e}")

    if args.loops_p1 and args.loops_p4:
        h_p1 = load_or_raise(args.loops_p1)
        h_p4 = load_or_raise(args.loops_p4)
        diff_mpi = float(np.max(np.abs(h_p1 - h_p4)))
        print(f"max |h P=1 - h P=4| (loops) = {diff_mpi:.6e}")

    for name, h in [("loops", h_loops), ("vec", h_vec)]:
        xs, ys, da, a_eff = grid_coords(h, A)
        print(f"{name}: shape={h.shape}, da={da:.6f}, a_eff={a_eff:.6f}")

        plot_heatmap(h, args.out / f"heatmap_{name}.png", f"Membrana h — {name}")
        plot_surface3d(
            h,
            xs,
            ys,
            args.out / f"surface3d_{name}.png",
            f"Powierzchnia h — {name}",
        )
        err = plot_cross_section(
            h,
            xs,
            a_eff,
            args.out / f"cross_section_{name}.png",
            f"Przekrój przez środek — {name}",
        )
        print(f"{name}: max błąd vs analityczne (przekrój) = {err:.6e}")


if __name__ == "__main__":
    main()
