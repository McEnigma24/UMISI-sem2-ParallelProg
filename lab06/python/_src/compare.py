"""Agregacja przebiegów Langevin — wykresy czasu, throughput i błędu vs N."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r"_(density|mfpt)_(cupy|pycuda|cpu)_")


def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_run_name(name: str) -> tuple[str, str] | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_records(results_root: Path) -> list[dict]:
    records: list[dict] = []
    for meta_path in results_root.glob("*/*/rep*/meta.json"):
        parts = meta_path.parts
        try:
            idx = parts.index("results")
        except ValueError:
            continue
        rest = parts[idx + 1 :]
        if len(rest) < 4:
            continue
        run_name = rest[1]
        parsed = parse_run_name(run_name)
        if not parsed:
            continue
        problem, backend = parsed
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_meta_path = meta_path.parent.parent.parent / "run_meta.json"
        dt = None
        if run_meta_path.is_file():
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            dt = float(run_meta.get("dt", 0))
        err = meta.get("l1_error") if problem == "density" else meta.get("mean_error")
        n_steps_total = meta.get("n_steps_total")
        if n_steps_total is None and problem == "mfpt" and dt:
            mean_t = meta.get("mean_exit_time")
            n_ex = int(meta.get("n_exited", 0))
            if mean_t and n_ex > 0:
                n_steps_total = mean_t * n_ex / dt
        t_total = float(meta["t_total"])
        throughput = None
        if n_steps_total is not None and t_total > 0:
            throughput = float(n_steps_total) / t_total
        records.append(
            {
                "run_name": run_name,
                "problem": problem,
                "backend": backend,
                "n": int(meta["n"]),
                "t_total": t_total,
                "t_kernel_sim": float(meta.get("t_kernel_sim", 0)),
                "error": float(err) if err is not None else None,
                "throughput": throughput,
                "dt": dt,
                "mean_exit": meta.get("mean_exit_time"),
                "std_exit": meta.get("std_exit_time"),
                "n_exited": int(meta.get("n_exited", 0)),
            }
        )
    return records


def aggregate_mean(rows: list[dict], key: str) -> dict[tuple, dict]:
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        k = (r["problem"], r["backend"], r["n"])
        v = r.get(key)
        if v is not None:
            bucket[k].append(float(v))
    out: dict[tuple, dict] = {}
    for k, vals in bucket.items():
        out[k] = {"n": k[2], "mean": float(np.mean(vals))}
    return out


def plot_time_vs_n(rows: list[dict], problem: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, color in [("cupy", "#1f77b4"), ("pycuda", "#ff7f0e"), ("cpu", "#2ca02c")]:
        sub = [r for r in rows if r["problem"] == problem and r["backend"] == backend]
        if not sub:
            continue
        agg = aggregate_mean(sub, "t_total")
        agg_sim = aggregate_mean(sub, "t_kernel_sim")
        ns = sorted({k[2] for k in agg})
        ax.loglog(ns, [agg[(problem, backend, n)]["mean"] for n in ns], "o-", color=color, label=f"{backend} total")
        ax.loglog(
            ns,
            [agg_sim[(problem, backend, n)]["mean"] for n in ns],
            "s--",
            color=color,
            alpha=0.7,
            label=f"{backend} sim",
        )
    ax.set_xlabel("N trajektorii")
    ax.set_ylabel("czas [s]")
    ax.set_title(f"Czas vs N — {problem}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_throughput_vs_n(rows: list[dict], problem: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, color in [("cupy", "#1f77b4"), ("pycuda", "#ff7f0e"), ("cpu", "#2ca02c")]:
        sub = [r for r in rows if r["problem"] == problem and r["backend"] == backend]
        sub = [r for r in sub if r.get("throughput")]
        if not sub:
            continue
        agg = aggregate_mean(sub, "throughput")
        ns = sorted({k[2] for k in agg})
        ax.loglog(ns, [agg[(problem, backend, n)]["mean"] for n in ns], "o-", color=color, label=backend)
    ax.set_xlabel("N trajektorii")
    ax.set_ylabel("kroki/s (N·n_steps / t_total)")
    ax.set_title(f"Throughput vs N — {problem}")
    h, lab = ax.get_legend_handles_labels()
    if lab:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density_error_vs_n(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    first_ns: list[int] = []
    first_scale = 1.0
    for backend, color in [("cupy", "#1f77b4"), ("pycuda", "#ff7f0e"), ("cpu", "#2ca02c")]:
        sub = [r for r in rows if r["problem"] == "density" and r["backend"] == backend]
        sub = [r for r in sub if r.get("error") is not None]
        if not sub:
            continue
        agg = aggregate_mean(sub, "error")
        ns = sorted({k[2] for k in agg})
        ax.loglog(ns, [agg[("density", backend, n)]["mean"] for n in ns], "o-", color=color, label=backend)
        if not first_ns:
            first_ns = ns
            first_scale = agg[("density", backend, ns[0])]["mean"]
    if first_ns:
        ref = 1.0 / np.sqrt(np.array(first_ns, dtype=float))
        ax.loglog(first_ns, ref / ref[0] * first_scale, "k:", label="∝1/√N (skala)")
    ax.set_xlabel("N trajektorii")
    ax.set_ylabel("L1: |histogram − N(0,τ)|")
    ax.set_title("Density: błąd histogramu vs N")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _sem_stderr(rows: list[dict]) -> dict[tuple, float]:
    """Średni błąd standardowy estymatora ⟨τ⟩ (SEM) per (problem, backend, n)."""
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        std = r.get("std_exit")
        n_ex = r.get("n_exited", 0)
        if std is None or n_ex < 1:
            continue
        sem = float(std) / np.sqrt(n_ex)
        bucket[(r["problem"], r["backend"], r["n"])].append(sem)
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def plot_mfpt_mean_vs_n(rows: list[dict], out_path: Path) -> None:
    """Szacowane ⟨τ⟩ w funkcji N (nie |⟨τ⟩−1|): widać limit numeryczny i teorię."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ref_rows = [r for r in rows if r["problem"] == "mfpt" and r.get("mean_exit") is not None]
    sem_map = _sem_stderr(ref_rows)

    ys_all = [float(r["mean_exit"]) for r in ref_rows]
    ymax_data = max(ys_all)
    ymin_data = min(ys_all)
    ax.set_xscale("log")
    ax.set_xlabel("N trajektorii")
    ax.set_ylabel("szacowane ⟨τ⟩")
    ax.set_title("MFPT: średni czas wyjścia vs N")
    # Dolna granica musi obejmować teorię ⟨τ⟩=1 (dane są ~1.02–1.04, inaczej linia znika).
    ax.set_ylim(0.99, ymax_data + 0.012)
    ax.grid(True, which="both", alpha=0.3)

    large = [r for r in ref_rows if r["n"] >= 1_000_000 and r["backend"] in ("cupy", "pycuda")]
    limit = float(np.mean([r["mean_exit"] for r in large])) if large else None

    ax.axhline(
        1.0,
        color="crimson",
        linestyle="--",
        linewidth=2.0,
        zorder=1,
        label="analityczne ⟨τ⟩=1 (SDE ciągłe)",
    )
    if limit is not None:
        ax.axhline(
            limit,
            color="0.45",
            linestyle=":",
            linewidth=2.0,
            zorder=1,
            label=rf"limit numeryczny ≈ {limit:.4f} (bias Δt)",
        )

    def series_for(backend: str) -> tuple[list[int], list[float], list[float]] | None:
        sub = [r for r in ref_rows if r["backend"] == backend]
        if not sub:
            return None
        agg = aggregate_mean(sub, "mean_exit")
        ns = sorted({k[2] for k in agg})
        ys = [agg[("mfpt", backend, n)]["mean"] for n in ns]
        yerr = [sem_map.get(("mfpt", backend, n), 0.0) for n in ns]
        return ns, ys, yerr

    cupy = series_for("cupy")
    pycuda = series_for("pycuda")
    gpu_same = False
    if cupy and pycuda:
        ns1, ys1, _ = cupy
        ns2, ys2, _ = pycuda
        gpu_same = ns1 == ns2 and all(abs(a - b) < 1e-12 for a, b in zip(ys1, ys2, strict=True))

    series_plan: list[tuple[str, str, str, str]] = []
    if gpu_same and cupy:
        series_plan.append(("cupy / pycuda", "#1f77b4", "o", "cupy"))
    else:
        if cupy:
            series_plan.append(("cupy", "#1f77b4", "o", "cupy"))
        if pycuda:
            series_plan.append(("pycuda", "#ff7f0e", "s", "pycuda"))
    series_plan.append(("cpu", "#2ca02c", "^", "cpu"))

    for label, color, marker, key in series_plan:
        data = series_for(key)
        if not data:
            continue
        ns, ys, yerr = data
        ax.errorbar(
            ns,
            ys,
            yerr=yerr,
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            label=label,
            linewidth=1.8,
            markersize=6,
            zorder=3,
        )

    ax.text(
        1.008,
        1.0,
        "  teoria: 1.0",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        color="crimson",
        fontsize=9,
        clip_on=False,
    )

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "run_name",
        "problem",
        "backend",
        "n",
        "t_total",
        "t_kernel_sim",
        "throughput",
        "error",
        "mean_exit",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["problem"], x["backend"], x["n"])):
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    root = find_project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "results" / "local",
    )
    args = parser.parse_args()
    compare_dir = root / "results" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(args.results)
    if not rows:
        print(f"Brak rekordów w {args.results}")
        return

    for problem in ("density", "mfpt"):
        plot_time_vs_n(rows, problem, compare_dir / f"{problem}_time_vs_n.png")
        plot_throughput_vs_n(rows, problem, compare_dir / f"{problem}_throughput_vs_n.png")
    plot_density_error_vs_n(rows, compare_dir / "density_error_vs_n.png")
    plot_mfpt_mean_vs_n(rows, compare_dir / "mfpt_error_vs_n.png")

    write_summary_csv(rows, compare_dir / "summary.csv")
    print(f"Porównanie w {compare_dir} ({len(rows)} punktów)")


if __name__ == "__main__":
    main()