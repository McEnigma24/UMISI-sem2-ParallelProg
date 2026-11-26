"""Agregacja meta.json — wykresy S, E, f dla Ising Metropolis."""

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

RUN_RE = re.compile(r"_L(\d+)_(border|smart)_(\d+)p$")


def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_run_dir_name(name: str) -> tuple[int, str, int] | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


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
        L = int(meta.get("L", 0))
        sync = str(meta.get("sync", ""))
        workers = int(meta.get("workers", 0))
        if L <= 0 or workers <= 0:
            parsed = parse_run_dir_name(run_name)
            if parsed:
                L, sync, workers = parsed
        records.append(
            {
                "platform": meta.get("platform", platform),
                "L": L,
                "sync": sync,
                "workers": workers,
                "wall_time": float(meta["wall_time"]),
                "sync_per_mcs": float(meta.get("sync_per_mcs", 0)),
                "run": run_name,
                "rep": rep_name,
            }
        )
    return records


def aggregate(records: list[dict]) -> dict[tuple, dict]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    sync_buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["platform"], r["L"], r["sync"], r["workers"])
        buckets[key].append(r["wall_time"])
        sync_buckets[key].append(r["sync_per_mcs"])
    out: dict[tuple, dict] = {}
    for key, times in buckets.items():
        arr = np.asarray(times, dtype=np.float64)
        item = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }
        if key in sync_buckets:
            item["sync_per_mcs_mean"] = float(np.mean(sync_buckets[key]))
        out[key] = item
    return out


def serial_reference(
    agg: dict, platform: str, L: int, sync: str
) -> tuple[float | None, int, str]:
    key1 = (platform, L, sync, 1)
    if key1 in agg:
        return agg[key1]["mean"], 1, platform
    for plat in ("local", "grid"):
        if plat == platform:
            continue
        key = (plat, L, sync, 1)
        if key in agg:
            return agg[key]["mean"], 1, plat
    return None, 0, ""


def karp_flatt(speedup: float, p: int) -> float:
    if p <= 1 or speedup <= 0:
        return float("nan")
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def metric_tables(agg: dict, platform: str, L: int) -> list[dict]:
    rows = []
    for (plat, L_i, sync, p), stats in sorted(agg.items()):
        if plat != platform or L_i != L:
            continue
        t1, p_ref, ref_source = serial_reference(agg, platform, L, sync)
        if t1 is None or t1 <= 0 or p_ref != 1:
            continue
        tp = stats["mean"]
        s = t1 / tp
        e = s / p if p > 0 else float("nan")
        f = karp_flatt(s, p)
        rows.append(
            {
                "platform": platform,
                "L": L,
                "sync": sync,
                "workers": p,
                "wall_mean": tp,
                "t_serial_ref": t1,
                "serial_ref_source": ref_source,
                "speedup": s,
                "efficiency": e,
                "serial_fraction": f,
                "sync_per_mcs_mean": stats.get("sync_per_mcs_mean"),
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
    by_sync: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sync[r["sync"]].append(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    workers = sorted({int(r["workers"]) for r in rows})
    if metric == "speedup" and workers:
        p_ideal = np.asarray(workers, dtype=np.float64)
        ax.plot(p_ideal, p_ideal, "k--", linewidth=1.8, label="idealne S = p", zorder=1)
    if metric == "efficiency" and workers:
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1.8, label="idealne E = 1", zorder=1)
    for sync in sorted(by_sync):
        pts = sorted(by_sync[sync], key=lambda x: x["workers"])
        xs = [p["workers"] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, "o-", label=sync)
    ax.set_xlabel("liczba procesów P")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wall_vs_p(agg: dict, platform: str, L: int, out_path: Path) -> None:
    by_sync: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (plat, L_i, sync, p), stats in agg.items():
        if plat != platform or L_i != L:
            continue
        by_sync[sync].append((p, stats["mean"]))
    if not by_sync:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for sync in sorted(by_sync):
        pts = sorted(by_sync[sync], key=lambda x: x[0])
        ax.plot([x[0] for x in pts], [x[1] for x in pts], "o-", label=sync)
    ax.set_xlabel("P")
    ax.set_ylabel("wall_time [s]")
    ax.set_title(f"[{platform}] wall_time vs P (L={L})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sync_vs_p(agg: dict, platform: str, L: int, out_path: Path) -> None:
    by_sync: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (plat, L_i, sync, p), stats in agg.items():
        if plat != platform or L_i != L:
            continue
        if "sync_per_mcs_mean" in stats:
            by_sync[sync].append((p, stats["sync_per_mcs_mean"]))
    if not by_sync:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for sync in sorted(by_sync):
        pts = sorted(by_sync[sync], key=lambda x: x[0])
        ax.plot([x[0] for x in pts], [x[1] for x in pts], "o-", label=sync)
    ax.set_xlabel("P")
    ax.set_ylabel("synchronizacje / MCS")
    ax.set_title(f"[{platform}] sync/MCS vs P (L={L})")
    ax.grid(True, alpha=0.3)
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


def write_tex_table(rows: list[dict], out_path: Path, platform: str, L: int) -> None:
    subset = [r for r in rows if r["platform"] == platform and r["L"] == L]
    if not subset:
        return
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"sync & $P$ & $T$ [s] & $S$ & $E$ & $f$ \\",
        r"\midrule",
    ]
    for r in sorted(subset, key=lambda x: (x["sync"], x["workers"])):
        fval = r["serial_fraction"]
        fstr = f"{fval:.3f}" if np.isfinite(fval) else "---"
        lines.append(
            f"{r['sync']} & {r['workers']} & {r['wall_mean']:.4f} & "
            f"{r['speedup']:.3f} & {r['efficiency']:.3f} & {fstr} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_platform(agg: dict, platform: str, compare_dir: Path) -> None:
    Ls = sorted({k[1] for k in agg if k[0] == platform})
    if not Ls:
        return
    plat_dir = compare_dir / platform
    plat_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = compare_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for L in Ls:
        rows = metric_tables(agg, platform, L)
        all_rows.extend(rows)
        plot_wall_vs_p(agg, platform, L, plat_dir / f"wall_time_vs_p_L{L}.png")
        plot_sync_vs_p(agg, platform, L, plat_dir / f"sync_per_mcs_vs_p_L{L}.png")
        if not rows:
            print(f"[compare] {platform} L={L}: brak S,E,f (dodaj run z P=1)")
            continue
        for metric, ylab, fname in (
            ("speedup", "przyspieszenie S", "speedup_vs_p"),
            ("efficiency", "efektywność E", "efficiency_vs_p"),
            ("serial_fraction", "serial fraction f", "serial_fraction_vs_p"),
        ):
            plot_metric_vs_p(
                rows,
                metric,
                ylab,
                f"[{platform}] {metric} vs P (L={L})",
                plat_dir / f"{fname}_L{L}.png",
            )
        write_tex_table(rows, tex_dir / f"metrics_{platform}_L{L}.tex", platform, L)

    write_summary_csv(all_rows, plat_dir / "summary.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Wykresy S, E, f — Ising")
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

    all_rows = sorted(
        [
            row
            for plat in ("local", "grid")
            for L in sorted({k[1] for k in agg if k[0] == plat})
            for row in metric_tables(agg, plat, L)
        ],
        key=lambda r: (r["platform"], r["L"], r["sync"], r["workers"]),
    )
    write_summary_csv(all_rows, compare_dir / "summary_all.csv")
    print(f"Zapisano wykresy w {compare_dir}")


if __name__ == "__main__":
    main()