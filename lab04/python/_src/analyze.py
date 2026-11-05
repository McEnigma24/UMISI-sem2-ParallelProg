#!/usr/bin/env python3
"""Analiza wyników Julia + MPIPoolExecutor (p. 3–6 z lab04)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAGS = ("baseline", "sleep")


def load_lines(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_run_info(run_path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not run_path.is_file():
        return info
    for line in run_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def worker_ranks(rows: list[dict]) -> list[int]:
    return sorted({int(r["rank"]) for r in rows})


def lines_per_worker(rows: list[dict]) -> dict[int, int]:
    c: dict[int, int] = defaultdict(int)
    for r in rows:
        c[int(r["rank"])] += 1
    return dict(c)


def time_per_worker(rows: list[dict]) -> dict[int, float]:
    s: dict[int, float] = defaultdict(float)
    for r in rows:
        s[int(r["rank"])] += float(r["time"])
    return dict(s)


def line_time_stats(rows: list[dict]) -> tuple[float, float, float, int, int]:
    times = [(int(r["line"]), float(r["time"])) for r in rows]
    tvals = [t for _, t in times]
    tmin = min(tvals)
    tmax = max(tvals)
    lmin = min(line for line, t in times if t == tmin)
    lmax = min(line for line, t in times if t == tmax)
    return tmin, tmax, tmax - tmin, lmin, lmax


def predict_block_rank(line: int, height: int, ranks: list[int]) -> int:
    w = len(ranks)
    chunk = (height + w - 1) // w
    idx = min(line // chunk, w - 1)
    return ranks[idx]


def predict_cyclic_rank(line: int, ranks: list[int]) -> int:
    w = len(ranks)
    return ranks[line % w]


def assignment_hypothesis(rows: list[dict], height: int, mode: str) -> tuple[float, list[tuple[int, int, int]]]:
    ranks = worker_ranks(rows)
    samples: list[tuple[int, int, int]] = []
    ok = 0
    for r in rows:
        line = int(r["line"])
        actual = int(r["rank"])
        if mode == "block":
            expected = predict_block_rank(line, height, ranks)
        else:
            expected = predict_cyclic_rank(line, ranks)
        if len(samples) < 30:
            samples.append((line, actual, expected))
        if actual == expected:
            ok += 1
    pct = 100.0 * ok / len(rows) if rows else 0.0
    return pct, samples


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline()
        if not magic.startswith(b"P5"):
            raise ValueError(f"Nieobsługiwany format PGM: {magic!r}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        if maxval > 255:
            raise ValueError("PGM maxval > 255")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(h, w)


def plot_all(
    tag: str,
    rows: list[dict],
    total_time: float,
    height: int,
    results_dir: Path,
    plots_dir: Path,
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    lines = np.array([int(r["line"]) for r in rows])
    times = np.array([float(r["time"]) for r in rows])
    ranks = np.array([int(r["rank"]) for r in rows])
    wr = worker_ranks(rows)
    lpc = lines_per_worker(rows)
    tpc = time_per_worker(rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(lines, times, c=ranks, cmap="tab10", s=8, alpha=0.7)
    ax.set_xlabel("Numer linii")
    ax.set_ylabel("Czas linii [s]")
    ax.set_title(f"Czas obliczeń linii ({tag}) — kolor = rank workera")
    plt.colorbar(sc, ax=ax, label="rank")
    fig.tight_layout()
    fig.savefig(plots_dir / f"line_time_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(lines, ranks, s=6, alpha=0.5)
    ax.set_xlabel("Numer linii")
    ax.set_ylabel("Rank workera")
    ax.set_title(f"Przydział linii do workerów ({tag})")
    ax.set_yticks(wr)
    fig.tight_layout()
    fig.savefig(plots_dir / f"line_rank_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [str(r) for r in wr]
    ys = [lpc.get(r, 0) for r in wr]
    ax.bar(xs, ys)
    ax.set_xlabel("Rank workera")
    ax.set_ylabel("Liczba linii")
    ax.set_title(f"Linie na workera ({tag})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"lines_per_worker_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ys2 = [tpc.get(r, 0) for r in wr]
    ax.bar(xs, ys2)
    ax.axhline(total_time, color="red", linestyle="--", label=f"czas programu = {total_time:.3f} s")
    ax.set_xlabel("Rank workera")
    ax.set_ylabel("Suma czasów linii [s]")
    ax.set_title(f"Suma czasów workera vs czas całkowity ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"time_per_worker_{tag}.png", dpi=150)
    plt.close(fig)

    pgm = results_dir / f"julia_{tag}.pgm"
    if pgm.is_file():
        img = read_pgm(pgm)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, cmap="gray", origin="upper")
        ax.set_title(f"Zbiór Julii ({tag})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / f"julia_{tag}.png", dpi=150)
        plt.close(fig)


def analyze_tag(
    tag: str,
    out_lines: list[str],
    results_dir: Path,
    plots_dir: Path,
) -> dict:
    csv_path = results_dir / f"lines_{tag}.csv"
    run_path = results_dir / f"run_{tag}.txt"
    rows = load_lines(csv_path)
    info = load_run_info(run_path)
    height = int(info.get("height", len(rows)))
    total_time = float(info.get("total_time", 0))

    out_lines.append(f"\n{'=' * 60}")
    out_lines.append(f"TAG: {tag}")
    out_lines.append(f"{'=' * 60}")
    out_lines.append(f"Plik: {csv_path}, linii: {len(rows)}, workers: {info.get('workers', '?')}")
    out_lines.append(f"Czas całkowity programu (MPI.Wtime): {total_time:.6f} s")

    tmin, tmax, tdiff, lmin, lmax = line_time_stats(rows)
    out_lines.append("\n--- p.3 Czasy linii ---")
    out_lines.append(f"Min czas: {tmin:.6f} s (linia {lmin})")
    out_lines.append(f"Max czas: {tmax:.6f} s (linia {lmax})")
    out_lines.append(f"Różnica max-min: {tdiff:.6f} s")

    pct_b, samp_b = assignment_hypothesis(rows, height, "block")
    pct_c, samp_c = assignment_hypothesis(rows, height, "cyclic")
    out_lines.append("\n--- p.4 Przydział linii ---")
    out_lines.append(f"Zgodność z podziałem blokowym: {pct_b:.2f}%")
    out_lines.append(f"Zgodność z podziałem cyklicznym: {pct_c:.2f}%")
    out_lines.append("Pierwsze przypisania (linia, rank_faktyczny, rank_oczekiwany blok/cykl):")
    for i in range(min(10, len(samp_b))):
        lb, ab, eb = samp_b[i]
        _, ac, ec = samp_c[i]
        out_lines.append(f"  linia {lb}: rank={ab} (blok->{eb}, cykl->{ec})")
    if pct_b < 50 and pct_c < 50:
        out_lines.append(
            "Wniosek: przydział nie jest ani czysto blokowy, ani cykliczny — "
            "typowy dynamiczny pool (wolny worker bierze kolejną linię)."
        )

    lpc = lines_per_worker(rows)
    out_lines.append("\n--- p.4b Liczba linii na workera ---")
    for r in sorted(lpc):
        out_lines.append(f"  rank {r}: {lpc[r]} linii")

    tpc = time_per_worker(rows)
    sums = list(tpc.values())
    out_lines.append("\n--- p.5 Suma czasów na workera ---")
    for r in sorted(tpc):
        out_lines.append(f"  rank {r}: {tpc[r]:.6f} s")
    if sums:
        out_lines.append(
            f"Min/max suma: {min(sums):.6f} / {max(sums):.6f} s, "
            f"std={np.std(sums):.6f} s"
        )
    sum_all = sum(sums)
    out_lines.append(f"Suma wszystkich czasów linii: {sum_all:.6f} s")
    if total_time > 0:
        eff = sum_all / total_time
        out_lines.append(
            f"Suma czasów / czas programu = {eff:.3f} "
            f"(efektywnie ~{eff:.1f} równoległych workerów)"
        )
    out_lines.append(
        "Uwaga: suma czasów workerów >> czas programu, bo linie liczą się równolegle; "
        "czas programu ≈ długość ścieżki krytycznej w harmonogramie dynamicznym."
    )

    plot_all(tag, rows, total_time, height, results_dir, plots_dir)

    return {
        "tag": tag,
        "total_time": total_time,
        "pct_block": pct_b,
        "pct_cyclic": pct_c,
        "lines_per_worker": lpc,
        "time_per_worker": tpc,
        "tdiff": tdiff,
    }


def compare_baseline_sleep(stats: dict, out_lines: list[str]):
    if "baseline" not in stats or "sleep" not in stats:
        return
    b = stats["baseline"]
    s = stats["sleep"]
    out_lines.append(f"\n{'=' * 60}")
    out_lines.append("--- p.6 Baseline vs sleep ---")
    out_lines.append(f"Czas programu: {b['total_time']:.3f} s -> {s['total_time']:.3f} s")
    out_lines.append(
        f"Zgodność blok: {b['pct_block']:.1f}% -> {s['pct_block']:.1f}%; "
        f"cykl: {b['pct_cyclic']:.1f}% -> {s['pct_cyclic']:.1f}%"
    )
    out_lines.append("Linie na workera (baseline -> sleep):")
    all_r = sorted(set(b["lines_per_worker"]) | set(s["lines_per_worker"]))
    for r in all_r:
        out_lines.append(
            f"  rank {r}: {b['lines_per_worker'].get(r, 0)} -> {s['lines_per_worker'].get(r, 0)}"
        )
    out_lines.append("Suma czasów workera (baseline -> sleep):")
    for r in all_r:
        out_lines.append(
            f"  rank {r}: {b['time_per_worker'].get(r, 0):.3f} -> {s['time_per_worker'].get(r, 0):.3f} s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results_dir = args.results_dir
    plots_dir = results_dir / "plots"

    out_lines: list[str] = ["Analiza lab04 — Julia MPIPoolExecutor"]
    stats: dict[str, dict] = {}
    for tag in TAGS:
        csv_path = results_dir / f"lines_{tag}.csv"
        if not csv_path.is_file():
            print(f"Brak {csv_path}, pomijam tag {tag}", file=sys.stderr)
            continue
        stats[tag] = analyze_tag(tag, out_lines, results_dir, plots_dir)

    compare_baseline_sleep(stats, out_lines)

    summary_path = results_dir / "summary.txt"
    text = "\n".join(out_lines) + "\n"
    summary_path.write_text(text)
    print(text)
    print(f"Zapisano {summary_path} i wykresy w {plots_dir}/")


if __name__ == "__main__":
    main()
