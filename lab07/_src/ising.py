"""Równoległy Metropolis 2D Ising — dekompozycja pozioma, mpi4py."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import (
    MODE_BORDER,
    MODE_SMART,
    find_next_border,
    flip_one,
    local_energy_magnetization,
    owner_stale_at,
    run_chunk,
    warmup_jit,
)

SYNC_MODES = {"border": MODE_BORDER, "smart": MODE_SMART}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D Ising Metropolis (MPI)")
    p.add_argument("--L", type=int, required=True, help="bok siatki L×L")
    p.add_argument("--steps", type=int, required=True, help="liczba MCS (t)")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--sync", choices=("border", "smart"), default="border")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--platform", choices=("local", "grid"), default="local")
    p.add_argument(
        "--init",
        choices=("random", "up"),
        default="random",
    )
    p.add_argument(
        "--measure-every",
        type=int,
        default=0,
        help="co ile MCS zapisać E,M (0=tylko koniec)",
    )
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def build_exp_table(beta: float, max_de: int = 8) -> np.ndarray:
    tab = np.zeros(max_de + 1, dtype=np.float64)
    for d in range(1, max_de + 1):
        tab[d] = np.exp(-beta * d)
    return tab


def init_lattice(h: int, L: int, init: str, rng: np.random.Generator) -> np.ndarray:
    spins = np.ones((h + 2, L), dtype=np.int8)
    if init == "random":
        spins[1 : h + 1, :] = rng.choice(np.array([-1, 1], dtype=np.int8), size=(h, L))
    else:
        spins[1 : h + 1, :] = 1
    return spins


def exchange_halos(spins: np.ndarray, comm: MPI.Intracomm, rank: int, size: int) -> float:
    """Wymiana wierszy halo; zwraca czas komunikacji (sekundy)."""
    t0 = time.perf_counter()
    h = spins.shape[0] - 2
    L = spins.shape[1]
    up = (rank - 1) % size
    down = (rank + 1) % size
    recv_top = np.empty(L, dtype=np.int8)
    recv_bot = np.empty(L, dtype=np.int8)
    comm.Sendrecv(
        spins[h, :].copy(),
        dest=down,
        sendtag=10,
        recvbuf=recv_top,
        source=up,
        recvtag=10,
    )
    comm.Sendrecv(
        spins[1, :].copy(),
        dest=up,
        sendtag=11,
        recvbuf=recv_bot,
        source=down,
        recvtag=11,
    )
    spins[0, :] = recv_top
    spins[h + 1, :] = recv_bot
    return time.perf_counter() - t0


def gather_lattice(spins: np.ndarray, comm: MPI.Intracomm, rank: int, size: int, L: int, h: int) -> np.ndarray | None:
    local = spins[1 : h + 1, :].copy()
    if rank == 0:
        full = np.empty((L, L), dtype=np.int8)
        full[0:h, :] = local
        for src in range(1, size):
            buf = np.empty((h, L), dtype=np.int8)
            comm.Recv(buf, source=src, tag=100)
            r0 = src * h
            full[r0 : r0 + h, :] = buf
        return full
    comm.Send(local, dest=0, tag=100)
    return None


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    L = args.L
    if L % size != 0:
        if rank == 0:
            raise SystemExit(f"L={L} musi być podzielne przez P={size}")
        comm.Abort(1)
    h = L // size
    if h < 2:
        if rank == 0:
            raise SystemExit(f"Każdy pas musi mieć h>=2, mam h={h}")
        comm.Abort(1)

    row0 = rank * h
    row1 = row0 + h
    mode = SYNC_MODES[args.sync]
    n_flip = L * L

    warmup_jit(L=min(8, L), h=min(4, h))
    comm.Barrier()

    init_rng = np.random.default_rng(args.seed + 1)
    spins = init_lattice(h, L, args.init, init_rng)

    stale_top = np.zeros(L, dtype=np.bool_)
    stale_bot = np.zeros(L, dtype=np.bool_)
    exp_tab = build_exp_table(args.beta)

    # początkowa wymiana halo
    comm_time_0 = exchange_halos(spins, comm, rank, size)
    stale_top[:] = False
    stale_bot[:] = False

    measure_every = args.measure_every or args.steps
    snap_every = args.snapshot_every
    history_E: list[float] = []
    history_M: list[float] = []
    history_t: list[int] = []
    snapshots: list[np.ndarray] = []
    snap_ts: list[int] = []

    compute_time = 0.0
    comm_time = comm_time_0
    n_sync = 1
    total_accepted = 0
    total_attempted_local = 0

    if rank == 0 and not args.quiet:
        print(
            f"L={L} P={size} h={h} steps={args.steps} beta={args.beta} sync={args.sync}",
            flush=True,
        )

    wall_t0 = MPI.Wtime()

    for step in range(1, args.steps + 1):
        idx_rng = np.random.default_rng(np.random.SeedSequence(args.seed + step).generate_state(1)[0])
        rows = idx_rng.integers(0, L, size=n_flip, dtype=np.int32)
        cols = idx_rng.integers(0, L, size=n_flip, dtype=np.int32)
        acc_child = np.random.SeedSequence(args.seed + 1_000_000 + step).spawn(size)[rank]
        uacc = np.random.default_rng(acc_child).random(n_flip).astype(np.float64)

        i = 0
        while i < n_flip:
            k_border = find_next_border(rows, i, n_flip, h)

            t_comp0 = time.perf_counter()
            acc, att = run_chunk(
                spins,
                rows,
                cols,
                uacc,
                i,
                k_border,
                row0,
                row1,
                mode,
                stale_top,
                stale_bot,
                exp_tab,
                args.beta,
                L,
            )
            compute_time += time.perf_counter() - t_comp0
            total_accepted += acc
            total_attempted_local += att

            if k_border >= n_flip:
                break

            need_sync = (
                1
                if mode == MODE_BORDER
                else owner_stale_at(rows, cols, k_border, row0, row1, stale_top, stale_bot)
            )
            need_sync = comm.allreduce(need_sync, op=MPI.MAX)

            if need_sync:
                dt = exchange_halos(spins, comm, rank, size)
                comm_time += dt
                n_sync += 1
                stale_top[:] = False
                stale_bot[:] = False

            t_comp0 = time.perf_counter()
            acc1, att1 = flip_one(
                spins,
                rows,
                cols,
                uacc,
                k_border,
                row0,
                row1,
                mode,
                stale_top,
                stale_bot,
                exp_tab,
                args.beta,
                L,
            )
            compute_time += time.perf_counter() - t_comp0
            total_accepted += acc1
            total_attempted_local += att1
            i = k_border + 1

        if step % measure_every == 0 or step == args.steps:
            e_loc, m_loc = local_energy_magnetization(spins, L)
            e_g = comm.allreduce(e_loc, op=MPI.SUM)
            m_g = comm.allreduce(m_loc, op=MPI.SUM)
            if rank == 0:
                history_E.append(float(e_g))
                history_M.append(float(m_g))
                history_t.append(step)

        if snap_every and step % snap_every == 0:
            if rank == 0:
                snap = gather_lattice(spins, comm, rank, size, L, h)
                if snap is not None:
                    snapshots.append(snap)
                    snap_ts.append(step)
            else:
                gather_lattice(spins, comm, rank, size, L, h)
        elif not snap_every and step == args.steps:
            pass

    wall_time = MPI.Wtime() - wall_t0

    e_loc, m_loc = local_energy_magnetization(spins, L)
    energy = comm.allreduce(e_loc, op=MPI.SUM)
    magnetization = comm.allreduce(m_loc, op=MPI.SUM)
    acc_global = comm.allreduce(total_accepted, op=MPI.SUM)
    att_global = comm.allreduce(total_attempted_local, op=MPI.SUM)
    sync_global = comm.allreduce(n_sync, op=MPI.SUM)
    # suma sync_global na wszystkich rankach to nie to samo co liczba sync — każdy rank liczy to samo
    n_sync_effective = n_sync

    if rank == 0:
        lattice = gather_lattice(spins, comm, rank, size, L, h)
    else:
        gather_lattice(spins, comm, rank, size, L, h)
        lattice = None

    if rank != 0:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = size
    accept_rate = acc_global / max(att_global, 1)

    np.savez_compressed(
        args.out_dir / "data.npz",
        lattice=lattice if lattice is not None else np.array([], dtype=np.int8),
        history_t=np.array(history_t, dtype=np.int32),
        history_E=np.array(history_E, dtype=np.float64),
        history_M=np.array(history_M, dtype=np.float64),
        snapshot_ts=np.array(snap_ts, dtype=np.int32),
        snapshots=(
            np.stack(snapshots, axis=0).astype(np.int8)
            if snapshots
            else np.zeros((0, L, L), dtype=np.int8)
        ),
        rows=rows,
        cols=cols,
    )

    meta = {
        "platform": args.platform,
        "L": L,
        "steps": args.steps,
        "beta": args.beta,
        "sync": args.sync,
        "seed": args.seed,
        "init": args.init,
        "workers": workers,
        "mpi_size": size,
        "h": h,
        "wall_time": wall_time,
        "compute_time_rank0": compute_time,
        "comm_time_rank0": comm_time,
        "n_sync": n_sync_effective,
        "sync_per_mcs": n_sync_effective / args.steps,
        "energy": float(energy),
        "magnetization": float(magnetization),
        "accept_rate": float(accept_rate),
        "attempts_local_sum": int(att_global),
        "accepted_sum": int(acc_global),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"L={L} P={workers} sync={args.sync} wall={wall_time:.4f}s "
        f"E={energy:.2f} M={magnetization:.0f} sync/MCS={meta['sync_per_mcs']:.1f}"
    )
    print(f"Zapisano: {args.out_dir}")


if __name__ == "__main__":
    main()