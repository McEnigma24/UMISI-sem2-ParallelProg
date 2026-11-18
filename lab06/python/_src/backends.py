"""GPU backends (CuPy, PyCUDA) and optional NumPy CPU reference."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

KERNEL_PATH = Path(__file__).resolve().parent / "kernels" / "langevin.cu"
BLOCK = 256
NVCC_INCLUDES = ("-I/usr/include", "-I/usr/lib/cuda/include")


def _read_cuda_source() -> str:
    return KERNEL_PATH.read_text(encoding="utf-8")


def launch_grid(n: int) -> tuple[int, int]:
    grid = min(max(1, (n + BLOCK - 1) // BLOCK), 65535)
    return grid, BLOCK


class LangevinBackend(ABC):
    name: str
    t_compile: float = 0.0

    @abstractmethod
    def compile(self) -> None:
        ...

    @abstractmethod
    def device_name(self) -> str:
        ...

    @abstractmethod
    def run_density(
        self, n: int, dt: float, tau: float, bins: int, seed: int
    ) -> tuple[dict, dict]:
        ...

    @abstractmethod
    def run_mfpt(
        self, n: int, dt: float, max_time: float, seed: int
    ) -> tuple[dict, dict]:
        ...

    def warmup(self) -> None:
        self.run_density(1024, 1e-3, 0.01, 50, 0)
        self.run_mfpt(1024, 1e-3, 1.0, 1)


class CupyBackend(LangevinBackend):
    name = "cupy"

    def __init__(self) -> None:
        import cupy as cp

        self.cp = cp
        self._mod = None
        self._compiled = False

    def compile(self) -> None:
        if self._compiled:
            return
        import os

        cp = self.cp
        for candidate in ("/usr/lib/cuda", "/usr/local/cuda"):
            if Path(candidate).is_dir():
                os.environ.setdefault("CUDA_PATH", candidate)
                break
        t0 = time.perf_counter()
        opts = tuple(NVCC_INCLUDES)
        self._mod = cp.RawModule(code=_read_cuda_source(), options=opts)
        self.t_compile = time.perf_counter() - t0
        self._compiled = True

    def _kernels(self):
        return {
            "density": self._mod.get_function("density_kernel"),
            "histogram": self._mod.get_function("histogram_kernel"),
            "mfpt": self._mod.get_function("mfpt_kernel"),
            "sum_reduce": self._mod.get_function("sum_reduce_kernel"),
        }

    def device_name(self) -> str:
        cp = self.cp
        with cp.cuda.Device(0):
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            return name

    def _event_ms(self, start, end) -> float:
        return float(self.cp.cuda.get_elapsed_time(start, end))

    def run_density(
        self, n: int, dt: float, tau: float, bins: int, seed: int
    ) -> tuple[dict, dict]:
        cp = self.cp
        self.compile()
        k = self._kernels()
        n_steps = max(1, int(round(tau / dt)))
        sqrt_dt = float(np.sqrt(dt))
        grid, block = launch_grid(n)
        times: dict[str, float] = {}
        wall0 = time.perf_counter()

        t0 = cp.cuda.Event()
        t1 = cp.cuda.Event()
        t0.record()
        x_out = cp.empty(n, dtype=cp.float32)
        t_alloc = time.perf_counter() - wall0

        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        k["density"](
            (grid,),
            (block,),
            (
                x_out,
                np.int32(n),
                np.int32(n_steps),
                cp.float32(sqrt_dt),
                np.uint64(seed),
            ),
        )
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_ms(ev0, ev1) / 1000.0

        xmin, xmax = -5.0 * np.sqrt(tau), 5.0 * np.sqrt(tau)
        hist = cp.zeros(bins, dtype=cp.uint32)
        shmem = bins * 4

        ev2, ev3 = cp.cuda.Event(), cp.cuda.Event()
        ev2.record()
        k["histogram"](
            (grid,),
            (block,),
            (
                x_out,
                np.int32(n),
                hist,
                np.int32(bins),
                cp.float32(xmin),
                cp.float32(xmax),
            ),
            shared_mem=shmem,
        )
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_ms(ev2, ev3) / 1000.0

        ev4, ev5 = cp.cuda.Event(), cp.cuda.Event()
        ev4.record()
        hist_h = cp.asnumpy(hist)
        sample = cp.asnumpy(x_out[:: max(1, n // 100_000)])
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_ms(ev4, ev5) / 1000.0

        mean = float(sample.mean())
        var = float(sample.var(ddof=0))
        edges = np.linspace(xmin, xmax, bins + 1)

        results = {
            "hist_counts": hist_h.astype(np.int64),
            "hist_edges": edges,
            "mean": mean,
            "variance": var,
            "n_steps": n_steps,
            "tau": tau,
            "analytic_var": tau,
        }
        times["t_alloc"] = t_alloc
        times["t_total"] = time.perf_counter() - wall0
        times["n_trajectories"] = n
        times["n_steps_total"] = n * n_steps
        return results, times

    def run_mfpt(
        self, n: int, dt: float, max_time: float, seed: int
    ) -> tuple[dict, dict]:
        cp = self.cp
        self.compile()
        k = self._kernels()
        max_steps = max(1, int(round(max_time / dt)))
        sqrt_dt = float(np.sqrt(dt))
        grid, block = launch_grid(n)
        wall0 = time.perf_counter()
        times: dict[str, float] = {}

        t_alloc = time.perf_counter() - wall0
        exit_t = cp.empty(n, dtype=cp.float32)
        sum_out = cp.zeros(1, dtype=cp.float64)
        cnt_out = cp.zeros(1, dtype=cp.uint32)

        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        k["mfpt"](
            (grid,),
            (block,),
            (
                exit_t,
                np.int32(n),
                cp.float32(dt),
                cp.float32(sqrt_dt),
                np.int32(max_steps),
                np.uint64(seed),
            ),
        )
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_ms(ev0, ev1) / 1000.0

        shmem = 2 * BLOCK * 8
        sum_out[:] = 0
        cnt_out[:] = 0
        ev2, ev3 = cp.cuda.Event(), cp.cuda.Event()
        ev2.record()
        k["sum_reduce"](
            (grid,),
            (block,),
            (exit_t, np.int32(n), sum_out, cnt_out),
            shared_mem=shmem,
        )
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_ms(ev2, ev3) / 1000.0

        ev4, ev5 = cp.cuda.Event(), cp.cuda.Event()
        ev4.record()
        total_sum = float(sum_out[0])
        n_exited = int(cnt_out[0])
        exit_h = cp.asnumpy(exit_t)
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_ms(ev4, ev5) / 1000.0

        exited = exit_h[exit_h > 0]
        n_censored = int((exit_h <= 0).sum())
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
        times["t_alloc"] = t_alloc
        times["t_total"] = time.perf_counter() - wall0
        times["n_trajectories"] = n
        times["n_steps_total"] = int(total_sum / dt)
        return results, times


class PycudaBackend(LangevinBackend):
    name = "pycuda"

    def __init__(self) -> None:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.cuda = cuda
        self.SourceModule = SourceModule
        self._mod = None
        self._compiled = False

    def compile(self) -> None:
        if self._compiled:
            return
        t0 = time.perf_counter()
        opts = ["-ccbin", "g++-13", *NVCC_INCLUDES]
        self._mod = self.SourceModule(
            _read_cuda_source(),
            no_extern_c=True,
            options=opts,
        )
        self.t_compile = time.perf_counter() - t0
        self._compiled = True

    def _kernels(self):
        return {
            "density": self._mod.get_function("density_kernel"),
            "histogram": self._mod.get_function("histogram_kernel"),
            "mfpt": self._mod.get_function("mfpt_kernel"),
            "sum_reduce": self._mod.get_function("sum_reduce_kernel"),
        }

    def device_name(self) -> str:
        dev = self.cuda.Device(0)
        return dev.name().decode() if isinstance(dev.name(), bytes) else str(dev.name())

    def _event_ms(self, start, end) -> float:
        end.synchronize()
        return float(start.time_till(end))

    def run_density(
        self, n: int, dt: float, tau: float, bins: int, seed: int
    ) -> tuple[dict, dict]:
        cuda = self.cuda
        self.compile()
        k = self._kernels()
        n_steps = max(1, int(round(tau / dt)))
        sqrt_dt = float(np.sqrt(dt))
        grid, block = launch_grid(n)
        times: dict[str, float] = {}
        wall0 = time.perf_counter()

        x_out = cuda.mem_alloc(n * 4)
        times["t_alloc"] = time.perf_counter() - wall0

        ev0, ev1 = cuda.Event(), cuda.Event()
        ev0.record()
        k["density"](
            x_out,
            np.int32(n),
            np.int32(n_steps),
            np.float32(sqrt_dt),
            np.uint64(seed),
            block=(block, 1, 1),
            grid=(grid, 1),
        )
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_ms(ev0, ev1) / 1000.0

        xmin, xmax = -5.0 * np.sqrt(tau), 5.0 * np.sqrt(tau)
        hist = cuda.mem_alloc(bins * 4)
        cuda.memset_d32(hist, 0, bins)

        ev2, ev3 = cuda.Event(), cuda.Event()
        ev2.record()
        k["histogram"](
            x_out,
            np.int32(n),
            hist,
            np.int32(bins),
            np.float32(xmin),
            np.float32(xmax),
            block=(block, 1, 1),
            grid=(grid, 1),
            shared=bins * 4,
        )
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_ms(ev2, ev3) / 1000.0

        ev4, ev5 = cuda.Event(), cuda.Event()
        ev4.record()
        hist_h = np.empty(bins, dtype=np.uint32)
        cuda.memcpy_dtoh(hist_h, hist)
        x_sample = np.empty(min(n, 100_000), dtype=np.float32)
        stride = max(1, n // len(x_sample))
        host_buf = np.empty(n, dtype=np.float32)
        cuda.memcpy_dtoh(host_buf, x_out)
        x_sample = host_buf[::stride]
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_ms(ev4, ev5) / 1000.0

        mean = float(x_sample.mean())
        var = float(x_sample.var(ddof=0))
        edges = np.linspace(xmin, xmax, bins + 1)

        results = {
            "hist_counts": hist_h.astype(np.int64),
            "hist_edges": edges,
            "mean": mean,
            "variance": var,
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
        cuda = self.cuda
        self.compile()
        k = self._kernels()
        max_steps = max(1, int(round(max_time / dt)))
        sqrt_dt = float(np.sqrt(dt))
        grid, block = launch_grid(n)
        wall0 = time.perf_counter()
        times: dict[str, float] = {}

        exit_t = cuda.mem_alloc(n * 4)
        sum_out = cuda.mem_alloc(8)
        cnt_out = cuda.mem_alloc(4)
        cuda.memset_d32(sum_out, 0, 2)
        cuda.memset_d32(cnt_out, 0, 1)
        times["t_alloc"] = time.perf_counter() - wall0

        ev0, ev1 = cuda.Event(), cuda.Event()
        ev0.record()
        k["mfpt"](
            exit_t,
            np.int32(n),
            np.float32(dt),
            np.float32(sqrt_dt),
            np.int32(max_steps),
            np.uint64(seed),
            block=(block, 1, 1),
            grid=(grid, 1),
        )
        ev1.record()
        ev1.synchronize()
        times["t_kernel_sim"] = self._event_ms(ev0, ev1) / 1000.0

        shmem = 2 * BLOCK * 8
        ev2, ev3 = cuda.Event(), cuda.Event()
        ev2.record()
        k["sum_reduce"](
            exit_t,
            np.int32(n),
            sum_out,
            cnt_out,
            block=(block, 1, 1),
            grid=(grid, 1),
            shared=shmem,
        )
        ev3.record()
        ev3.synchronize()
        times["t_kernel_post"] = self._event_ms(ev2, ev3) / 1000.0

        ev4, ev5 = cuda.Event(), cuda.Event()
        ev4.record()
        total_sum = np.empty(1, dtype=np.float64)
        n_exited_arr = np.empty(1, dtype=np.uint32)
        cuda.memcpy_dtoh(total_sum, sum_out)
        cuda.memcpy_dtoh(n_exited_arr, cnt_out)
        exit_h = np.empty(n, dtype=np.float32)
        cuda.memcpy_dtoh(exit_h, exit_t)
        ev5.record()
        ev5.synchronize()
        times["t_d2h"] = self._event_ms(ev4, ev5) / 1000.0

        n_exited = int(n_exited_arr[0])
        exited = exit_h[exit_h > 0]
        n_censored = int((exit_h <= 0).sum())
        mean_exit = float(total_sum[0]) / max(n_exited, 1)
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
        times["n_steps_total"] = int(float(total_sum[0]) / dt)
        return results, times


class NumpyBackend(LangevinBackend):
    name = "cpu"

    def compile(self) -> None:
        self.t_compile = 0.0

    def device_name(self) -> str:
        return "CPU NumPy"

    def run_density(
        self, n: int, dt: float, tau: float, bins: int, seed: int
    ) -> tuple[dict, dict]:
        wall0 = time.perf_counter()
        n_steps = max(1, int(round(tau / dt)))
        sqrt_dt = float(np.sqrt(dt))
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        x = np.zeros(n, dtype=np.float64)
        for _ in range(n_steps):
            x += sqrt_dt * rng.standard_normal(n)
        t_sim = time.perf_counter() - t0

        xmin, xmax = -5.0 * np.sqrt(tau), 5.0 * np.sqrt(tau)
        t1 = time.perf_counter()
        hist, edges = np.histogram(x, bins=bins, range=(xmin, xmax))
        t_post = time.perf_counter() - t1

        results = {
            "hist_counts": hist.astype(np.int64),
            "hist_edges": edges,
            "mean": float(x.mean()),
            "variance": float(x.var(ddof=0)),
            "n_steps": n_steps,
            "tau": tau,
            "analytic_var": tau,
        }
        times = {
            "t_alloc": 0.0,
            "t_kernel_sim": t_sim,
            "t_kernel_post": t_post,
            "t_d2h": 0.0,
            "t_total": time.perf_counter() - wall0,
            "n_trajectories": n,
            "n_steps_total": n * n_steps,
        }
        return results, times

    def run_mfpt(
        self, n: int, dt: float, max_time: float, seed: int
    ) -> tuple[dict, dict]:
        wall0 = time.perf_counter()
        max_steps = max(1, int(round(max_time / dt)))
        sqrt_dt = float(np.sqrt(dt))
        rng = np.random.default_rng(seed)
        exit_times = np.empty(n, dtype=np.float64)

        t0 = time.perf_counter()
        x = np.zeros(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        t_arr = np.zeros(n, dtype=np.float64)
        for _ in range(max_steps):
            if not np.any(active):
                break
            k = int(active.sum())
            x[active] += sqrt_dt * rng.standard_normal(k)
            t_arr[active] += dt
            hit = active & (np.abs(x) > 1.0)
            exit_times[hit] = t_arr[hit]
            active &= ~hit
        still = active
        exit_times[still] = -t_arr[still]
        t_sim = time.perf_counter() - t0

        exited = exit_times[exit_times > 0]
        n_exited = len(exited)
        n_censored = n - n_exited
        mean_exit = float(exited.mean()) if n_exited else 0.0
        std_exit = float(exited.std(ddof=1)) if n_exited > 1 else 0.0

        results = {
            "mean_exit_time": mean_exit,
            "std_exit_time": std_exit,
            "n_exited": n_exited,
            "n_censored": n_censored,
            "max_steps": max_steps,
            "analytic_mean": 1.0,
            "exit_times_sample": exited[:: max(1, len(exited) // 50_000)],
        }
        times = {
            "t_alloc": 0.0,
            "t_kernel_sim": t_sim,
            "t_kernel_post": 0.0,
            "t_d2h": 0.0,
            "t_total": time.perf_counter() - wall0,
            "n_trajectories": n,
            "n_steps_total": int(float(exited.sum()) / dt),
        }
        return results, times


def make_backend(name: str) -> LangevinBackend:
    name = name.lower()
    if name == "cupy":
        return CupyBackend()
    if name == "pycuda":
        return PycudaBackend()
    if name == "cpu":
        return NumpyBackend()
    raise ValueError(f"Unknown backend: {name}")