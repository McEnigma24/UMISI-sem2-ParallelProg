"""Zbiera meta.json z results/ i rysuje porównanie local vs grid (wall_time)."""

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
from matplotlib.cm import Greens, Purples

RUN_RE = re.compile(r"_(\d+)x(\d+)_(\d+)w$")


def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def megapixels(width: int, height: int) -> float:
    return width * height / 1e6


def parse_run_dir_name(name: str) -> tuple[int, int, int] | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _record_from_meta(meta_path: Path, platform: str, run_name: str) -> dict | None:
    parts = meta_path.parts
    variant = meta_path.parent.name
    if variant not in ("baseline", "sleep"):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    w = int(meta.get("width", 0))
    h = int(meta.get("height", 0))
    workers = int(meta.get("workers", 0))
    if w <= 0 or h <= 0:
        parsed = parse_run_dir_name(run_name)
        if parsed:
            w, h, workers = parsed
    return {
        "platform": meta.get("platform", platform),
        "variant": meta.get("variant", variant),
        "width": w,
        "height": h,
        "workers": workers,
        "wall_time": float(meta["wall_time"]),
    }


def load_records(results_root: Path) -> list[dict]:
    records: list[dict] = []
    patterns = (
        "*/*/*/meta.json",
        "map/*/*/*/meta.json",
        "submit/*/*/*/meta.json",
    )
    seen: set[str] = set()
    for pattern in patterns:
        for meta_path in results_root.glob(pattern):
            key = str(meta_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            parts = meta_path.parts
            try:
                idx = parts.index("results")
            except ValueError:
                continue
            rest = parts[idx + 1 :]
            if len(rest) == 4:
                platform, run_name = rest[0], rest[1]
                if platform not in ("local", "grid"):
                    continue
            elif len(rest) == 5 and rest[0] in ("map", "submit"):
                platform, run_name = rest[1], rest[2]
            else:
                continue
            rec = _record_from_meta(meta_path, platform, run_name)
            if rec:
                records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict[tuple, dict]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["platform"], r["variant"], r["width"], r["height"], r["workers"])
        buckets[key].append(r["wall_time"])
    out: dict[tuple, dict] = {}
    for key, times in buckets.items():
        arr = np.asarray(times, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }
    return out


def unique_sizes(agg: dict) -> list[tuple[int, int, float]]:
    sizes = {(k[2], k[3]) for k in agg}
    items = [(w, h, megapixels(w, h)) for w, h in sizes]
    items.sort(key=lambda x: (x[2], x[0], x[1]))
    return items


def worker_list(agg: dict, **filters) -> list[int]:
    ws = set()
    for key in agg:
        platform, variant, w, h, workers = key
        if filters.get("platform") and platform != filters["platform"]:
            continue
        if filters.get("variant") and variant != filters["variant"]:
            continue
        ws.add(workers)
    return sorted(ws)


def cmap_shade(cmap, workers: list[int], n: int) -> np.ndarray:
    if n not in workers:
        return np.array([cmap(0.65)])
    lo, hi = 0.35, 0.9
    t = (workers.index(n) + 1) / max(len(workers), 1)
    return cmap(lo + (hi - lo) * t)


def series_points(
    agg: dict,
    platform: str,
    variant: str,
    workers: int,
    sizes: list[tuple[int, int, float]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[str]]:
    xs, ys, ylo, yhi = [], [], [], []
    labels = []
    for w, h, mpx in sizes:
        key = (platform, variant, w, h, workers)
        if key not in agg:
            continue
        xs.append(mpx)
        ys.append(agg[key]["mean"])
        ylo.append(agg[key]["min"])
        yhi.append(agg[key]["max"])
        labels.append(f"{w}x{h}")
    return (
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(ylo),
        np.asarray(yhi),
    ), labels


def plot_local_vs_grid(agg: dict, out_path: Path, common_only: bool) -> None:
    sizes_all = unique_sizes(agg)
    ws_local = set(worker_list(agg, platform="local"))
    ws_grid = set(worker_list(agg, platform="grid"))
    if common_only:
        workers = sorted(ws_local & ws_grid)
        size_set_local = {(k[2], k[3]) for k in agg if k[0] == "local"}
        size_set_grid = {(k[2], k[3]) for k in agg if k[0] == "grid"}
        common_sizes = size_set_local & size_set_grid
        sizes = [(w, h, m) for w, h, m in sizes_all if (w, h) in common_sizes]
    else:
        workers = sorted(ws_local | ws_grid)
        sizes = sizes_all

    if not sizes or not workers:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    suffix = "wspólna siatka" if common_only else "wszystkie pomiary"
    for ax, variant in zip(axes, ("baseline", "sleep")):
        for n in workers:
            c_loc = cmap_shade(Greens, workers, n)
            c_grid = cmap_shade(Purples, workers, n)
            (xs_l, ys_l, lo_l, hi_l), _ = series_points(agg, "local", variant, n, sizes)
            (xs_g, ys_g, lo_g, hi_g), labels = series_points(
                agg, "grid", variant, n, sizes
            )
            if len(xs_l):
                ax.plot(xs_l, ys_l, color=c_loc, ls="-", lw=2, label=f"local {n}w")
                ax.fill_between(xs_l, lo_l, hi_l, color=c_loc, alpha=0.12)
            if len(xs_g):
                ax.plot(xs_g, ys_g, color=c_grid, ls="--", lw=2, label=f"grid {n}w")
                ax.fill_between(xs_g, lo_g, hi_g, color=c_grid, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("rozmiar [Mpx]")
        ax.set_ylabel("wall_time [s]")
        ax.set_title(f"{variant} ({suffix})")
        ax.grid(True, which="both", alpha=0.3)
        if labels:
            tick_x = [m for _, _, m in sizes]
            tick_l = [f"{w}x{h}" for w, h, _ in sizes]
            ax.set_xticks(tick_x)
            ax.set_xticklabels(tick_l, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(agg: dict, out_path: Path) -> None:
    rows = []
    for key, stats in sorted(agg.items()):
        platform, variant, w, h, workers = key
        rows.append(
            {
                "platform": platform,
                "variant": variant,
                "width": w,
                "height": h,
                "workers": workers,
                "megapixels": megapixels(w, h),
                "wall_mean": stats["mean"],
                "wall_min": stats["min"],
                "wall_max": stats["max"],
                "n_runs": stats["n"],
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Wykresy local vs grid (wall_time)")
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
    write_summary_csv(agg, compare_dir / "summary.csv")
    plot_local_vs_grid(agg, compare_dir / "local_vs_grid.png", common_only=False)
    plot_local_vs_grid(agg, compare_dir / "local_vs_grid_common.png", common_only=True)
    print(f"Zapisano wykresy i summary.csv w {compare_dir}")


if __name__ == "__main__":
    main()
