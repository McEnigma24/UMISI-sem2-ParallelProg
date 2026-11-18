"""Langevin equation — density and mean first passage time (GPU)."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from backends import make_backend


def parse_n_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if "e" in part.lower():
            out.append(int(float(part)))
        else:
            out.append(int(part))
    return out


def l1_hist_error(counts: np.ndarray, edges: np.ndarray, tau: float) -> float:
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    empirical = counts / max(counts.sum(), 1)
    analytic = (
        np.exp(-0.5 * centers * centers / tau)
        / np.sqrt(2.0 * np.pi * tau)
        * width
    )
    return float(np.abs(empirical - analytic).sum())


def parse_args():
    p = argparse.ArgumentParser(description="Langevin SDE on GPU")
    p.add_argument("--problem", choices=("density", "mfpt"), required=True)
    p.add_argument("--backend", choices=("cupy", "pycuda", "cpu"), default="cupy")
    p.add_argument("--n-list", type=str, default="1e5")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--bins", type=int, default=200)
    p.add_argument("--max-time", type=float, default=20.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    n_list = parse_n_list(args.n_list)
    backend = make_backend(args.backend)
    backend.compile()
    if args.backend != "cpu":
        backend.warmup()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "problem": args.problem,
        "backend": args.backend,
        "device": backend.device_name(),
        "t_compile": backend.t_compile,
        "dt": args.dt,
        "tau": args.tau,
        "bins": args.bins,
        "max_time": args.max_time,
        "n_list": n_list,
        "repeats": args.repeats,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": "local",
    }
    (args.out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2),
        encoding="utf-8",
    )

    runner = backend.run_density if args.problem == "density" else backend.run_mfpt
    extra = (
        {"tau": args.tau, "bins": args.bins}
        if args.problem == "density"
        else {"max_time": args.max_time}
    )

    for n in n_list:
        n_dir = args.out_dir / f"N{int(n)}"
        n_dir.mkdir(parents=True, exist_ok=True)
        for rep in range(1, args.repeats + 1):
            rep_dir = n_dir / f"rep{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed + rep * 17 + int(n)
            if args.problem == "density":
                results, times = runner(
                    int(n), args.dt, extra["tau"], extra["bins"], seed
                )
            else:
                results, times = runner(
                    int(n), args.dt, extra["max_time"], seed
                )

            meta = {
                "problem": args.problem,
                "backend": args.backend,
                "n": int(n),
                "rep": rep,
                "seed": seed,
                **times,
            }
            if args.problem == "density":
                meta["mean"] = results["mean"]
                meta["variance"] = results["variance"]
                meta["l1_error"] = l1_hist_error(
                    results["hist_counts"],
                    results["hist_edges"],
                    results["tau"],
                )
                meta["var_error"] = abs(results["variance"] - results["analytic_var"])
            else:
                meta["mean_exit_time"] = results["mean_exit_time"]
                meta["std_exit_time"] = results["std_exit_time"]
                meta["n_exited"] = results["n_exited"]
                meta["n_censored"] = results["n_censored"]
                meta["mean_error"] = abs(
                    results["mean_exit_time"] - results["analytic_mean"]
                )

            (rep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            np.savez_compressed(rep_dir / "data.npz", **results)

            if not args.quiet:
                print(
                    f"n={n} rep={rep} t_total={times['t_total']:.4f}s meta={meta}",
                    flush=True,
                )
            else:
                line = (
                    f"n={n} rep={rep} wall={times['t_total']:.4f}s"
                    f" sim={times['t_kernel_sim']:.4f}s"
                )
                if args.problem == "density":
                    line += f" L1={meta['l1_error']:.4f}"
                else:
                    line += f" mean={results['mean_exit_time']:.4f}"
                print(line, flush=True)

    print(f"Zapisano przebieg: {args.out_dir}")


if __name__ == "__main__":
    main()