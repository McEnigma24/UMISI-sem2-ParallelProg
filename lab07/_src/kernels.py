"""Numba CPU kernels: Metropolis flips (lockstep batches)."""

from __future__ import annotations

import numba
import numpy as np

MODE_BORDER = 0
MODE_SMART = 1


@numba.njit(cache=True)
def neighbor_sum(spins: np.ndarray, i: int, j: int, L: int) -> int:
    s = 0
    s += spins[i - 1, j]
    s += spins[i + 1, j]
    s += spins[i, (j - 1) % L]
    s += spins[i, (j + 1) % L]
    return s


@numba.njit(cache=True)
def local_energy_magnetization(spins: np.ndarray, L: int) -> tuple[float, float]:
    h = spins.shape[0] - 2
    e = 0.0
    m = 0.0
    for i in range(1, h + 1):
        for j in range(L):
            s = spins[i, j]
            m += s
            e -= s * spins[i, (j + 1) % L]
            e -= s * spins[i + 1, j]
    return e, m


@numba.njit(cache=True)
def find_next_border(rows: np.ndarray, start: int, n: int, h: int) -> int:
    """Pierwszy indeks >= start, którego spin lezy na poziomej granicy poddomeny."""
    for k in range(start, n):
        gr = int(rows[k])
        owner = gr // h
        lr = gr - owner * h + 1
        if lr == 1 or lr == h:
            return k
    return n


@numba.njit(cache=True)
def owner_stale_at(
    rows: np.ndarray,
    cols: np.ndarray,
    k: int,
    row0: int,
    row1: int,
    stale_top: np.ndarray,
    stale_bot: np.ndarray,
) -> int:
    gr = int(rows[k])
    if gr < row0 or gr >= row1:
        return 0
    lr = gr - row0 + 1
    gc = int(cols[k])
    if lr == 1 and stale_bot[gc]:
        return 1
    if lr == row1 - row0 and stale_top[gc]:
        return 1
    return 0


@numba.njit(cache=True)
def run_chunk(
    spins: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    uacc: np.ndarray,
    start: int,
    end: int,
    row0: int,
    row1: int,
    mode: int,
    stale_top: np.ndarray,
    stale_bot: np.ndarray,
    exp_tab: np.ndarray,
    beta: float,
    L: int,
) -> tuple[int, int]:
    """Próby Metropolisa dla indeksów [start, end) — bez spinów brzegowych poddomen."""
    accepted = 0
    attempted = 0
    h = spins.shape[0] - 2
    for k in range(start, end):
        gr = int(rows[k])
        gc = int(cols[k])
        if gr < row0 or gr >= row1:
            continue
        lr = gr - row0 + 1
        if lr == 1 or lr == h:
            continue
        attempted += 1
        s = spins[lr, gc]
        nn = neighbor_sum(spins, lr, gc, L)
        de = 2 * s * nn
        if de <= 0:
            accept = True
        else:
            di = int(de)
            if di >= len(exp_tab):
                pacc = np.exp(-beta * de)
            else:
                pacc = exp_tab[di]
            accept = uacc[k] < pacc
        if accept:
            spins[lr, gc] = -s
    return accepted, attempted


@numba.njit(cache=True)
def flip_one(
    spins: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    uacc: np.ndarray,
    k: int,
    row0: int,
    row1: int,
    mode: int,
    stale_top: np.ndarray,
    stale_bot: np.ndarray,
    exp_tab: np.ndarray,
    beta: float,
    L: int,
) -> tuple[int, int]:
    """Pojedyncza próba na indeksie k (spin brzegowy lub wewnętrzny)."""
    gr = int(rows[k])
    gc = int(cols[k])
    if gr < row0 or gr >= row1:
        return 0, 0
    h = spins.shape[0] - 2
    lr = gr - row0 + 1
    attempted = 1
    s = spins[lr, gc]
    nn = neighbor_sum(spins, lr, gc, L)
    de = 2 * s * nn
    if de <= 0:
        accept = True
    else:
        di = int(de)
        if di >= len(exp_tab):
            pacc = np.exp(-beta * de)
        else:
            pacc = exp_tab[di]
        accept = uacc[k] < pacc
    if accept:
        spins[lr, gc] = -s
        if mode == MODE_SMART:
            if lr == 1:
                stale_bot[gc] = True
            if lr == h:
                stale_top[gc] = True
        return 1, attempted
    return 0, attempted


def warmup_jit(L: int = 8, h: int = 4) -> None:
    spins = np.ones((h + 2, L), dtype=np.int8)
    rows = np.arange(L * L, dtype=np.int32) % h
    cols = np.arange(L * L, dtype=np.int32) % L
    uacc = np.random.random(L * L)
    stale = np.zeros(L, dtype=np.bool_)
    exp_tab = np.array([0.0, np.exp(-1.0)], dtype=np.float64)
    find_next_border(rows, 0, len(rows), h)
    run_chunk(spins, rows, cols, uacc, 0, 4, 0, h, MODE_BORDER, stale, stale, exp_tab, 1.0, L)
    flip_one(spins, rows, cols, uacc, 0, 0, h, MODE_BORDER, stale, stale, exp_tab, 1.0, L)
    local_energy_magnetization(spins, L)