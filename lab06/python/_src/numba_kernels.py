"""Langevin CUDA kernels — Numba @cuda.jit (bez pliku .cu).

RNG: SplitMix64 + Box–Muller z (seed, traj, krok) — inny niż curand Philox w .cu,
ten sam schemat numeryczny Euler–Maruyama.
"""

from __future__ import annotations

import math

from numba import cuda, float32, float64, uint32, uint64

BLOCK = 256
HIST_SHARED_MAX = 512


@cuda.jit(device=True)
def _clip_int(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@cuda.jit(device=True)
def _splitmix64(x):
    z = uint64(x) + uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@cuda.jit(device=True)
def _u01(seed, traj, stream):
    h = (
        uint64(seed)
        ^ (uint64(traj) * uint64(0xD1342543DE82EF95))
        ^ (uint64(stream) * uint64(0x94D049BB133111EB))
    )
    z = _splitmix64(h)
    return float32(z >> 40) * float32(1.0 / 16777216.0)


@cuda.jit(device=True)
def _normal11(seed, traj, counter):
    u1 = _u01(seed, traj, counter * 2)
    u2 = _u01(seed, traj, counter * 2 + 1)
    u1f = float(u1)
    u2f = float(u2)
    if u1f < 1.0e-12:
        u1f = 1.0e-12
    r = math.sqrt(-2.0 * math.log(u1f))
    return float32(r * math.cos(6.283185307179586 * u2f))


@cuda.jit
def density_kernel(x_out, n, n_steps, sqrt_dt, seed):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for traj in range(tid, n, stride):
        x = float32(0.0)
        for s in range(n_steps):
            xi = _normal11(seed, traj, s)
            x += sqrt_dt * xi
        x_out[traj] = x


@cuda.jit
def histogram_kernel(x, hist, n, bins, xmin, xmax):
    shist = cuda.shared.array(HIST_SHARED_MAX, dtype=uint32)
    tid = cuda.threadIdx.x
    if tid < bins:
        shist[tid] = 0
    cuda.syncthreads()

    gid = cuda.grid(1)
    stride = cuda.gridsize(1)
    inv_w = float32(bins) / (xmax - xmin)

    for i in range(gid, n, stride):
        v = x[i]
        if v < xmin or v >= xmax:
            continue
        bin_idx = int((v - xmin) * inv_w)
        bin_idx = _clip_int(bin_idx, 0, bins - 1)
        cuda.atomic.add(shist, bin_idx, 1)

    cuda.syncthreads()

    if tid < bins:
        val = shist[tid]
        if val > 0:
            cuda.atomic.add(hist, tid, val)


@cuda.jit
def mfpt_kernel(exit_t, n, dt, sqrt_dt, max_steps, seed):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for traj in range(tid, n, stride):
        x = float32(0.0)
        t = float32(0.0)
        steps = 0
        counter = 0
        while steps < max_steps:
            xi = _normal11(seed, traj, counter)
            counter += 1
            x += sqrt_dt * xi
            t += dt
            steps += 1
            if x > float32(1.0) or x < float32(-1.0):
                exit_t[traj] = t
                break
        if steps >= max_steps:
            exit_t[traj] = -t


@cuda.jit
def sum_reduce_kernel(inp, n, sum_out, count_out):
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    gid = cuda.grid(1)
    stride = cuda.gridsize(1)

    local_sum = float64(0.0)
    local_count = float64(0.0)
    for i in range(gid, n, stride):
        v = inp[i]
        if v > float32(0.0):
            local_sum += float64(v)
            local_count += float64(1.0)

    shmem_sum = cuda.shared.array(BLOCK, dtype=float64)
    shmem_count = cuda.shared.array(BLOCK, dtype=float64)
    shmem_sum[tid] = local_sum
    shmem_count[tid] = local_count
    cuda.syncthreads()

    s = nthreads // 2
    while s > 0:
        if tid < s:
            shmem_sum[tid] += shmem_sum[tid + s]
            shmem_count[tid] += shmem_count[tid + s]
        cuda.syncthreads()
        s >>= 1

    if tid == 0:
        cuda.atomic.add(sum_out, 0, shmem_sum[0])
        cuda.atomic.add(count_out, 0, int(shmem_count[0]))


def warmup_compile() -> None:
    """Pierwsze uruchomienie = JIT; wywołanie na małych danych."""
    import numpy as np
    from numba import cuda as nb_cuda

    if not nb_cuda.is_available():
        raise RuntimeError("Numba CUDA niedostępna (brak GPU / sterownika)")

    n = 128
    x = nb_cuda.device_array(n, dtype=np.float32)
    h = nb_cuda.device_array(50, dtype=np.uint32)
    ex = nb_cuda.device_array(n, dtype=np.float32)
    s = nb_cuda.device_array(1, dtype=np.float64)
    c = nb_cuda.device_array(1, dtype=np.uint32)
    g, b = 1, BLOCK
    seed = np.uint64(0)
    density_kernel[g, b](x, n, 10, np.float32(0.0316), seed)
    histogram_kernel[g, b](x, h, n, 50, np.float32(-5.0), np.float32(5.0))
    mfpt_kernel[g, b](ex, n, np.float32(1e-3), np.float32(0.0316), 100, seed)
    sum_reduce_kernel[g, b](ex, n, s, c)
    nb_cuda.synchronize()