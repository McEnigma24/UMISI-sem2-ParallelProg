"""Backend cupy_naive: ndarray + cp.random + cp.histogram (bez langevin.cu).

Density: losowanie macierzy (batch × n_steps) + suma wzdłuż osi — bez własnych kerneli.
MFPT: pętla kroków na tablicach GPU (maska active), jak wektorowy CPU.

Uwaga: density zużywa O(batch·n_steps) VRAM; przy bardzo dużym N batching ogranicza szczyt.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from backends import LangevinBackend

# Szacunek: 2 × batch × n_steps × 4 B; batch 2^18 przy n_steps≈1000 ≈ 1 GB na tensorze xi
_DENSITY_BATCH_TRAJ = 262_144


def _ensure_cupy_cuda_path() -> None:
    for candidate in ("/usr/lib/cuda", "/usr/local/cuda"):
        if Path(candidate).is_dir():
            os.environ.setdefault("CUDA_PATH", candidate)
            break


class CupyNaiveBackend(LangevinBackend):
    name = "cupy_naive"

    def __init__(self) -> None:
        import cupy as cp

        _ensure_cupy_cuda_path()
        self.cp = cp
        self._compiled = False

    def compile(self) -> None:
        self.t_compile = 0.0
        self._compiled = True

    def device_name(self) -> str:
        cp = self.cp
        with cp.cuda.Device(0):
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            return name

    def _event_s(self, start, end) -> float:
        end.synchronize()
        return float(self.cp.cuda.get_elapsed_time(start, end)) / 1000.0

    def run_density(
        self, n: int, dt: float, tau: float, bins: int, seed: int
    ) -> tuple[dict, dict]:
        cp = self.cp
        self.compile()
        n_steps = max(1, int(round(tau / dt)))
        sqrt_dt = cp.float32(np.sqrt(dt))
        rng = cp.random.default_rng(seed)
        wall0 = time.perf_counter()
        times: dict[str, float] = {}

        x_out = cp.empty(n, dtype=cp.float32)
        times["t_alloc"] = 0.0

        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        batch = min(_DENSITY_BATCH_TRAJ, n)
        for start in range(0, n, batch):
            end = min(n, start + batch)
            bn = end - start
            xi = rng.standard_normal((bn, n_steps), dtype=cp.float32)
            x_out[start:end] = cp.sum(xi * sqrt_dt, axis=1, dtype=cp.float32)
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_s(ev0, ev1)

        xmin, xmax = -5.0 * np.sqrt(tau), 5.0 * np.sqrt(tau)

        ev2, ev3 = cp.cuda.Event(), cp.cuda.Event()
        ev2.record()
        hist_h, edges_h = cp.histogram(
            x_out, bins=bins, range=(float(xmin), float(xmax))
        )
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_s(ev2, ev3)
        times["t_histogram"] = times["t_kernel_post"]
        times["t_sum_reduce"] = 0.0

        ev4, ev5 = cp.cuda.Event(), cp.cuda.Event()
        ev4.record()
        hist_np = cp.asnumpy(hist_h)
        edges_np = cp.asnumpy(edges_h)
        sample = cp.asnumpy(x_out[:: max(1, n // 100_000)])
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_s(ev4, ev5)

        results = {
            "hist_counts": hist_np.astype(np.int64),
            "hist_edges": edges_np,
            "mean": float(sample.mean()),
            "variance": float(sample.var(ddof=0)),
            "n_steps": n_steps,
            "tau": tau,
            "analytic_var": tau,
        }
        times["t_total"] = time.perf_counter() - wall0
        times["n_trajectories"] = n
        times["n_steps_total"] = n * n_steps
        return results, times

    def run_mfpt(
        self, n: int, dt: float, max_time: float, seed: int
    ) -> tuple[dict, dict]:
        cp = self.cp
        self.compile()
        max_steps = max(1, int(round(max_time / dt)))
        sqrt_dt = cp.float32(np.sqrt(dt))
        dt_f = cp.float32(dt)
        rng = cp.random.default_rng(seed)
        wall0 = time.perf_counter()
        times: dict[str, float] = {}

        x = cp.zeros(n, dtype=cp.float32)
        t_arr = cp.zeros(n, dtype=cp.float32)
        exit_t = cp.zeros(n, dtype=cp.float32)
        active = cp.ones(n, dtype=cp.bool_)

        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        for _ in range(max_steps):
            if not bool(cp.any(active)):
                break
            xi = rng.standard_normal(n, dtype=cp.float32)
            x = cp.where(active, x + sqrt_dt * xi, x)
            t_arr = cp.where(active, t_arr + dt_f, t_arr)
            hit = active & ((x > 1.0) | (x < -1.0))
            exit_t = cp.where(hit, t_arr, exit_t)
            active = active & ~hit
        exit_t = cp.where(active, -t_arr, exit_t)
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_s(ev0, ev1)

        ev2, ev3 = cp.cuda.Event(), cp.cuda.Event()
        ev2.record()
        pos = exit_t > 0
        total_sum = float(cp.sum(exit_t[pos]))
        n_exited = int(cp.sum(pos))
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_s(ev2, ev3)
        times["t_histogram"] = 0.0
        times["t_sum_reduce"] = times["t_kernel_post"]

        ev4, ev5 = cp.cuda.Event(), cp.cuda.Event()
        ev4.record()
        exit_h = cp.asnumpy(exit_t)
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_s(ev4, ev5)
        times["t_alloc"] = 0.0

        exited = exit_h[exit_h > 0]
        n_censored = n - n_exited
        mean_exit = total_sum / max(n_exited, 1)
        std_exit = float(exited.std(ddof=1)) if len(exited) > 1 else 0.0

        results = {
            "mean_exit_time": mean_exit,
            "std_exit_time": std_exit,
            "n_exited": n_exited,
            "n_censored": n_censored,
            "max_steps": max_steps,
            "analytic_mean": 1.0,
            "exit_times_sample": exited[:: max(1, len(exited) // 50_000)],
        }
        times["t_total"] = time.perf_counter() - wall0
        times["n_trajectories"] = n
        times["n_steps_total"] = int(total_sum / dt)
        return results, times