"""TSP — równoległy branch and bound (MPIPoolExecutor)."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


def cities_on_circle(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * math.pi, size=n))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def min_outgoing(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf
        out[i] = float(row.min())
    return out

def optimal_circular_tour(coords: np.ndarray, dist: np.ndarray) -> tuple[float, list[int]]:
    # Miasta na okręgu w kolejności kątów z cities_on_circle.
    n = len(coords)
    order = list(range(n))
    cost = 0.0
    for i in range(n):
        cost += float(dist[order[i], order[(i + 1) % n]])
    return cost, order

def generate_prefixes(n: int, depth: int) -> list[list[int]]:
    """Prefiksy długości depth, pierwsze miasto 0."""
    if depth < 1:
        depth = 1
    if depth > n:
        depth = n
    if depth == 1:
        return [[0]]

    out: list[list[int]] = []

    def rec(path: list[int]) -> None:
        if len(path) == depth:
            out.append(path.copy())
            return
        used = set(path)
        for c in range(1, n):
            if c not in used:
                path.append(c)
                rec(path)
                path.pop()

    rec([0])
    return out

def path_cost(dist: np.ndarray, path: list[int]) -> float:
    s = 0.0
    for i in range(len(path) - 1):
        s += float(dist[path[i], path[i + 1]])
    return s

def lower_bound(
    dist: np.ndarray,
    min_out: np.ndarray,
    path: list[int],
    cost: float,
) -> float:
    n = dist.shape[0]
    cur = path[-1]
    unvisited = [u for u in range(n) if u not in path]
    if not unvisited:
        return cost + float(dist[cur, 0])
    to_uv = min(float(dist[cur, u]) for u in unvisited)
    tail = sum(float(min_out[u]) for u in unvisited)
    return cost + to_uv + tail

def solve_subtree(task_id: int, prefix: list[int], dist: np.ndarray, min_out: np.ndarray):
    rank = MPI.COMM_WORLD.Get_rank()
    n = dist.shape[0]
    t0 = time.perf_counter()
    nodes = 0

    best_cost = math.inf
    best_tour: list[int] | None = None

    stack: list[tuple[list[int], float]] = [(prefix, path_cost(dist, prefix))]
    while stack:
        path, cost = stack.pop()
        nodes += 1
        lb = lower_bound(dist, min_out, path, cost)
        if lb >= best_cost:
            continue
        if len(path) == n:
            total = cost + float(dist[path[-1], 0])
            if total < best_cost:
                best_cost = total
                best_tour = path.copy()
            continue

        cur = path[-1]
        used = set(path)
        children = [
            (float(dist[cur, c]), c) for c in range(n) if c not in used
        ]
        children.sort(key=lambda x: x[0])
        for _, c in reversed(children):
            stack.append((path + [c], cost + float(dist[cur, c])))

    elapsed = time.perf_counter() - t0
    if best_tour is None:
        best_cost = math.inf
    return task_id, rank, elapsed, nodes, best_cost, best_tour


def parse_args():
    p = argparse.ArgumentParser(description="TSP branch and bound — MPIPoolExecutor")
    p.add_argument("--n", type=int, default=12, help="liczba miast")
    p.add_argument("--depth", type=int, default=2, help="głębokość prefiksu d (włącznie z miastem 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--platform",
        choices=("local", "grid"),
        default="local",
    )
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()

def main():
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        return

    args = parse_args()
    n, d = args.n, args.depth
    if n < 2:
        raise SystemExit("n >= 2")
    if d < 1 or d > n:
        raise SystemExit("1 <= depth <= n")

    t0 = MPI.Wtime()

    coords = cities_on_circle(n, args.seed)
    dist = distance_matrix(coords)
    min_out = min_outgoing(dist)
    opt_cost, opt_tour = optimal_circular_tour(coords, dist)

    prefixes = generate_prefixes(n, d)
    task_ids = list(range(len(prefixes)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = comm.Get_size() - 1

    if not args.quiet:
        print(f"N={n} d={d} zadań={len(prefixes)} workers={workers}", flush=True)

    results: list[tuple] = []
    with MPIPoolExecutor() as executor:
        for row in executor.map(
            solve_subtree,
            task_ids,
            prefixes,
            repeat(dist),
            repeat(min_out),
            chunksize=args.chunksize,
        ):
            results.append(row)
            if not args.quiet:
                tid, rank, elapsed, nodes, bc, _ = row
                print(
                    f"task={tid} rank={rank} time={elapsed:.6f} nodes={nodes} cost={bc}",
                    flush=True,
                )

    wall_time = MPI.Wtime() - t0

    best_global = math.inf
    best_tour: list[int] | None = None
    # Najlepszy wynik spośród workerów.
    for _, _, _, _, bc, tour in results:
        if tour is None:
            continue
        if bc < best_global or (bc == best_global and tour < (best_tour or [])):
            best_global = bc
            best_tour = tour

    correct = False
    if best_tour is not None and math.isfinite(best_global):
        correct = abs(best_global - opt_cost) < 1e-6 * max(1.0, opt_cost)

    n_tasks = len(results)
    ranks = np.array([r[1] for r in results], dtype=np.int32)
    times = np.array([r[2] for r in results], dtype=np.float64)
    node_counts = np.array([r[3] for r in results], dtype=np.int64)
    costs = np.array([r[4] for r in results], dtype=np.float64)
    prefix_lens = np.array([len(p) for p in prefixes], dtype=np.int32)

    np.savez_compressed(
        args.out_dir / "data.npz",
        coords=coords,
        dist=dist,
        prefixes=np.array([str(p) for p in prefixes], dtype=object),
        ranks=ranks,
        times=times,
        node_counts=node_counts,
        costs=costs,
        prefix_lens=prefix_lens,
        best_tour=np.array(best_tour if best_tour else [], dtype=np.int32),
    )

    meta = {
        "platform": args.platform,
        "n": n,
        "depth": d,
        "seed": args.seed,
        "workers": workers,
        "mpi_size": comm.Get_size(),
        "n_tasks": n_tasks,
        "wall_time": wall_time,
        "chunksize": args.chunksize,
        "best_cost": best_global,
        "optimal_known": opt_cost,
        "correct": correct,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(
        f"N={n} d={d} workers={workers} wall_time={wall_time:.6f} s "
        f"best={best_global:.6f} opt={opt_cost:.6f} correct={correct}"
    )
    print(f"Zapisano: {args.out_dir / 'data.npz'}, {args.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
