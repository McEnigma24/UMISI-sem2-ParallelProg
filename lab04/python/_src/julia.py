"""Zbiór Julii — MPIPoolExecutor (mpi4py.futures)."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

C_JULIA = complex(0, 0.65)
X0, X1 = -2.0, +2.0
Y0, Y1 = -1.5, +1.5


def julia(x: float, y: float) -> int:
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + C_JULIA
        n -= 1
    return n


def julia_line(
    k: int,
    w: int,
    h: int,
    sleep_line: int,
    sleep_time: float,
    quiet: bool = True,
):
    rank = MPI.COMM_WORLD.Get_rank()
    t0 = time.perf_counter()
    slept = 0.0

    if sleep_line >= 0 and k == sleep_line and sleep_time > 0:
        time.sleep(sleep_time)
        slept = sleep_time

    dx = (X1 - X0) / w
    dy = (Y1 - Y0) / h
    line = np.empty(w, dtype=np.uint8)
    y = Y1 - k * dy
    for j in range(w):
        line[j] = julia(X0 + j * dx, y)

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"line={k} rank={rank} time={elapsed:.6f}", flush=True)
    return (k, rank, elapsed, slept, line)


def max_worker_time_from_dir(baseline_dir: Path, factor: float = 1.0) -> float:
    path = baseline_dir / "data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Brak {path}")
    data = np.load(path)
    ranks = data["ranks"].astype(np.int64)
    times = data["times"].astype(np.float64)
    acc: dict[int, float] = {}
    for r, t in zip(ranks, times, strict=True):
        acc[int(r)] = acc.get(int(r), 0.0) + float(t)
    if not acc:
        return 0.0
    return max(acc.values()) * factor


def parse_args():
    p = argparse.ArgumentParser(description="Julia set — MPIPoolExecutor")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="katalog na data.npz i meta.json",
    )
    p.add_argument("--variant", default="baseline", help="baseline | sleep")
    p.add_argument("--sleep-line", type=int, default=1)
    p.add_argument("--sleep-time", type=float, default=None)
    p.add_argument(
        "--sleep-from",
        type=Path,
        default=None,
        help="katalog baseline z data.npz — sleep = max busy worker * factor",
    )
    p.add_argument("--sleep-factor", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument(
        "--platform",
        choices=("local", "grid"),
        default="local",
        help="źródło pomiaru (zapis w meta.json)",
    )
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="domyślnie bez printów linii (--no-quiet aby włączyć)",
    )
    return p.parse_args()


def main():
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        return

    args = parse_args()
    w, h = args.width, args.height
    sleep_line = -1
    sleep_time = 0.0
    if args.variant == "sleep":
        sleep_line = args.sleep_line
        if args.sleep_from is not None:
            sleep_time = max_worker_time_from_dir(args.sleep_from, args.sleep_factor)
        elif args.sleep_time is not None:
            sleep_time = args.sleep_time
        else:
            sleep_time = 0.0
        print(
            f"Sleep: linia {sleep_line}, czas {sleep_time:.6f} s",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    workers = comm.Get_size() - 1
    image = np.empty((h, w), dtype=np.uint8)
    ranks = np.empty(h, dtype=np.int32)
    times = np.empty(h, dtype=np.float64)
    slept = np.zeros(h, dtype=np.float64)

    t0 = MPI.Wtime()
    with MPIPoolExecutor() as executor:
        for k, rank, elapsed, sl, line in executor.map(
            julia_line,
            range(h),
            repeat(w),
            repeat(h),
            repeat(sleep_line),
            repeat(sleep_time),
            repeat(args.quiet),
            chunksize=args.chunksize,
        ):
            image[k] = line
            ranks[k] = rank
            times[k] = elapsed
            slept[k] = sl
    wall_time = MPI.Wtime() - t0

    np.savez_compressed(
        args.out_dir / "data.npz",
        image=image,
        ranks=ranks,
        times=times,
        slept=slept,
    )

    meta = {
        "platform": args.platform,
        "variant": args.variant,
        "width": w,
        "height": h,
        "workers": workers,
        "mpi_size": comm.Get_size(),
        "wall_time": wall_time,
        "chunksize": args.chunksize,
        "sleep_line": sleep_line if args.variant == "sleep" else None,
        "sleep_time": sleep_time if args.variant == "sleep" else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"variant={args.variant} workers={workers} wall_time={wall_time:.6f} s")
    print(f"Zapisano: {args.out_dir / 'data.npz'}, {args.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
