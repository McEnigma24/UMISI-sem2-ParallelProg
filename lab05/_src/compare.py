"""Agregacja meta.json i wykresy S, E, f (local / grid)."""

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

RUN_RE = re.compile(r"_N(\d+)_d(\d+)_(\d+)w$")


def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_run_dir_name(name: str) -> tuple[int, int, int] | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_records(results_root: Path) -> list[dict]:
    records: list[dict] = []
    for meta_path in results_root.glob("*/*/*/meta.json"):
        parts = meta_path.parts
        try:
            idx = parts.index("results")
        except ValueError:
            continue
        rest = parts[idx + 1 :]
        if len(rest) != 4:
            continue
        platform, run_name, rep_name = rest[0], rest[1], rest[2]
        if platform not in ("local", "grid"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n = int(meta.get("n", 0))
        d = int(meta.get("depth", 0))
        workers = int(meta.get("workers", 0))
        if n <= 0 or d <= 0:
            parsed = parse_run_dir_name(run_name)
            if parsed:
                n, d, workers = parsed
        records.append(
            {
                "platform": meta.get("platform", platform),
                "n": n,
                "depth": d,
                "workers": workers,
                "wall_time": float(meta["wall_time"]),
                "correct": bool(meta.get("correct", False)),
                "total_nodes": int(meta.get("total_nodes", 0)) if "total_nodes" in meta else None,
                "run": run_name,
                "rep": rep_name,
            }
        )

    # nodes z metrics.json jeśli brak w meta
    for rec in records:
        if rec["total_nodes"] is not None:
            continue
        metrics_path = (
            results_root
            / rec["platform"]
            / rec["run"]
            / rec["rep"]
            / "metrics.json"
        )
        if metrics_path.is_file():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            rec["total_nodes"] = int(m.get("total_nodes", 0))
    return records


def aggregate(records: list[dict]) -> dict[tuple, dict]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    nodes_buckets: dict[tuple, list[int]] = defaultdict(list)
    for r in records:
        key = (r["platform"], r["n"], r["depth"], r["workers"])
        buckets[key].append(r["wall_time"])
        if r.get("total_nodes") is not None:
            nodes_buckets[key].append(int(r["total_nodes"]))
    out: dict[tuple, dict] = {}
    for key, times in buckets.items():
        arr = np.asarray(times, dtype=np.float64)
        item = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }
        if key in nodes_buckets and nodes_buckets[key]:
            item["nodes_mean"] = float(np.mean(nodes_buckets[key]))
        out[key] = item
    return out


def serial_reference(agg: dict, platform: str, n: int, d: int) -> float | None:
    key1 = (platform, n, d, 1)
    if key1 in agg:
        return agg[key1]["mean"]
    # fallback: minimalny czas przy 1 workerze z dowolnego wpisu
    times = [v["mean"] for k, v in agg.items() if k[0] == platform and k[1] == n and k[2] == d and k[3] == 1]
    return times[0] if times else None


def karp_flatt(speedup: float, p: int) -> float:
    if p <= 1 or speedup <= 0:
        return float("nan")
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def metric_tables(
    agg: dict, platform: str, n: int
) -> list[dict]:
    rows = []
    for (plat, n_i, d, p), stats in sorted(agg.items()):
        if plat != platform or n_i != n:
            continue
        t1 = serial_reference(agg, platform, n, d)
        if t1 is None or t1 <= 0:
            continue
        tp = stats["mean"]
        s = t1 / tp
        e = s / p if p > 0 else float("nan")
        f = karp_flatt(s, p)
        rows.append(
            {
                "platform": platform,
                "n": n,
                "depth": d,
                "workers": p,
                "wall_mean": tp,
                "t_serial_ref": t1,
                "speedup": s,
                "efficiency": e,
                "serial_fraction": f,
                "nodes_mean": stats.get("nodes_mean"),
            }
        )
    return rows


def plot_metric_vs_p(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_d[r["depth"]].append(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in sorted(by_d):
        pts = sorted(by_d[d], key=lambda x: x["workers"])
        xs = [p["workers"] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, "o-", label=f"d={d}")
    ax.set_xlabel("liczba procesów (workerów)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_vs_d(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    by_p: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["workers"] <= 1:
            continue
        by_p[r["workers"]].append(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in sorted(by_p):
        pts = sorted(by_p[p], key=lambda x: x["depth"])
        xs = [t["depth"] for t in pts]
        ys = [t[metric] for t in pts]
        ax.plot(xs, ys, "o-", label=f"p={p}")
    ax.set_xlabel("głębokość d (prefiks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nodes_vs_d(rows: list[dict], out_path: Path) -> None:
    by_p: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("nodes_mean") is None:
            continue
        by_p[r["workers"]].append(r)
    if not by_p:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in sorted(by_p):
        pts = sorted(by_p[p], key=lambda x: x["depth"])
        xs = [t["depth"] for t in pts]
        ys = [t["nodes_mean"] for t in pts]
        ax.plot(xs, ys, "o-", label=f"p={p}")
    ax.set_xlabel("głębokość d")
    ax.set_ylabel("średnia łączna liczba węzłów B&B")
    ax.set_title("Praca (węzły) vs d — lokalne obcinanie gałęzi")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def process_platform(agg: dict, platform: str, compare_dir: Path) -> None:
    ns = sorted({k[1] for k in agg if k[0] == platform})
    if not ns:
        return
    plat_dir = compare_dir / platform
    plat_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for n in ns:
        rows = metric_tables(agg, platform, n)
        all_rows.extend(rows)
        if not rows:
            continue
        plot_metric_vs_p(
            rows,
            "speedup",
            "przyspieszenie S",
            f"[{platform}] S vs p (N={n})",
            plat_dir / f"speedup_vs_p_N{n}.png",
        )
        plot_metric_vs_p(
            rows,
            "efficiency",
            "efektywność E",
            f"[{platform}] E vs p (N={n})",
            plat_dir / f"efficiency_vs_p_N{n}.png",
        )
        plot_metric_vs_p(
            rows,
            "serial_fraction",
            "serial fraction f",
            f"[{platform}] f vs p (N={n})",
            plat_dir / f"serial_fraction_vs_p_N{n}.png",
        )
        plot_metric_vs_d(
            rows,
            "speedup",
            "przyspieszenie S",
            f"[{platform}] S vs d (N={n})",
            plat_dir / f"speedup_vs_d_N{n}.png",
        )
        plot_metric_vs_d(
            rows,
            "efficiency",
            "efektywność E",
            f"[{platform}] E vs d (N={n})",
            plat_dir / f"efficiency_vs_d_N{n}.png",
        )
        plot_metric_vs_d(
            rows,
            "serial_fraction",
            "serial fraction f",
            f"[{platform}] f vs d (N={n})",
            plat_dir / f"serial_fraction_vs_d_N{n}.png",
        )
        plot_nodes_vs_d(rows, plat_dir / f"nodes_vs_d_N{n}.png")

    write_summary_csv(all_rows, plat_dir / "summary.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Wykresy S, E, f dla TSP B&B")
    p.add_argument("--results-root", type=Path, default=None)
    args = p.parse_args()
    root = args.results_root or (find_project_root() / "results")
    compare_dir = root / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(root)
    if not records:
        print(f"Brak meta.json pod {root}")
        return

    agg = aggregate(records)
    for platform in ("local", "grid"):
        process_platform(agg, platform, compare_dir)

    write_summary_csv(
        sorted(
            [
                row
                for plat in ("local", "grid")
                for n in sorted({k[1] for k in agg if k[0] == plat})
                for row in metric_tables(agg, plat, n)
            ],
            key=lambda r: (r["platform"], r["n"], r["depth"], r["workers"]),
        ),
        compare_dir / "summary_all.csv",
    )
    print(f"Zapisano wykresy w {compare_dir}")


if __name__ == "__main__":
    main()
