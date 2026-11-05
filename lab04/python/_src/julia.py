"""Zbiór Julii — MPIPoolExecutor (mpi4py.futures)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

# Wymiary widzą też workery (import modułu przy starcie mpiexec).
_w = int(os.environ.get("JULIA_WIDTH", "1280"))
_h = int(os.environ.get("JULIA_HEIGHT", "960"))
x0, x1 = -2.0, +2.0
y0, y1 = -1.5, +1.5
dx = (x1 - x0) / _w
dy = (y1 - y0) / _h
c = complex(0, 0.65)


def julia(x: float, y: float) -> int:
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n


def julia_line(k: int):
    rank = MPI.COMM_WORLD.Get_rank()
    t0 = time.perf_counter()
    slept = 0.0

    sleep_line = int(os.environ.get("JULIA_SLEEP_LINE", "-1"))
    sleep_time = float(os.environ.get("JULIA_SLEEP_TIME", "0"))
    if sleep_line >= 0 and k == sleep_line and sleep_time > 0:
        time.sleep(sleep_time)
        slept = sleep_time

    line = bytearray(_w)
    y = y1 - k * dy
    for j in range(_w):
        x = x0 + j * dx
        line[j] = julia(x, y)

    elapsed = time.perf_counter() - t0
    print(f"line={k} rank={rank} time={elapsed:.6f}", flush=True)
    return (k, rank, elapsed, slept, bytes(line))


def max_worker_time_from_csv(csv_path: Path, factor: float = 1.0) -> float:
    by_rank: dict[int, float] = defaultdict(float)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_rank[int(row["rank"])] += float(row["time"])
    if not by_rank:
        return 0.0
    return max(by_rank.values()) * factor


def parse_args():
    p = argparse.ArgumentParser(description="Julia set — MPIPoolExecutor")
    p.add_argument("--width", type=int, default=_w)
    p.add_argument("--height", type=int, default=_h)
    p.add_argument("--tag", default="baseline", help="baseline | sleep")
    p.add_argument("--sleep-line", type=int, default=1)
    p.add_argument("--sleep-time", type=float, default=None)
    p.add_argument(
        "--sleep-from",
        type=Path,
        default=None,
        help="CSV z baseline — sleep = max suma czasów workera * factor",
    )
    p.add_argument("--sleep-factor", type=float, default=1.0)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    return p.parse_args()


def main():
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        return

    args = parse_args()
    if args.width != _w or args.height != _h:
        print(
            f"Uwaga: workery używają JULIA_WIDTH/HEIGHT={_w}x{_h} z env; "
            f"CLI podaje {args.width}x{args.height}.",
            file=sys.stderr,
        )

    sleep_time = args.sleep_time
    if args.tag == "sleep":
        if args.sleep_from is not None:
            sleep_time = max_worker_time_from_csv(args.sleep_from, args.sleep_factor)
        if sleep_time is None:
            sleep_time = 0.0
        os.environ["JULIA_SLEEP_LINE"] = str(args.sleep_line)
        os.environ["JULIA_SLEEP_TIME"] = str(sleep_time)
        print(
            f"Sleep: linia {args.sleep_line}, czas {sleep_time:.6f} s "
            f"(env ustawione tylko na masterze — dla workerów export w pipeline)",
            flush=True,
        )

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    workers = comm.Get_size() - 1
    image: list[bytes | None] = [None] * _h
    rows: list[tuple[int, int, float, float]] = []

    pool_env = {}
    if args.tag == "sleep" and sleep_time is not None:
        pool_env["JULIA_SLEEP_LINE"] = str(args.sleep_line)
        pool_env["JULIA_SLEEP_TIME"] = str(sleep_time)

    t0 = MPI.Wtime()
    executor_kwargs = {"env": pool_env} if pool_env else {}
    with MPIPoolExecutor(**executor_kwargs) as executor:
        for k, rank, elapsed, slept, line_bytes in executor.map(julia_line, range(_h)):
            rows.append((k, rank, elapsed, slept))
            image[k] = line_bytes
    total_time = MPI.Wtime() - t0

    tag = args.tag
    csv_path = results_dir / f"lines_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["line", "rank", "time", "slept"])
        wtr.writerows(rows)

    pgm_path = results_dir / f"julia_{tag}.pgm"
    with pgm_path.open("wb") as f:
        f.write(f"P5\n{_w} {_h}\n255\n".encode())
        for line in image:
            assert line is not None
            f.write(line)

    run_path = results_dir / f"run_{tag}.txt"
    with run_path.open("w") as f:
        f.write(f"tag={tag}\n")
        f.write(f"width={_w}\n")
        f.write(f"height={_h}\n")
        f.write(f"mpi_size={comm.Get_size()}\n")
        f.write(f"workers={workers}\n")
        f.write(f"total_time={total_time:.6f}\n")
        if args.tag == "sleep":
            f.write(f"sleep_line={args.sleep_line}\n")
            f.write(f"sleep_time={sleep_time}\n")

    print(f"tag={tag} workers={workers} total_time={total_time:.6f} s")
    print(f"Zapisano: {csv_path}, {pgm_path}, {run_path}")


if __name__ == "__main__":
    main()
