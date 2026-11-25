# Lab06 — równanie Langevina na GPU (CuPy + PyCUDA)

Symulacja wielu trajektorii równania Langevina (schemat Euler–Maruyama) na GPU:

1. **density** — rozkład \(p(x,\tau=1)\) (histogram położeń w chwili \(\tau\)).
2. **mfpt** — średni pierwszy czas wyjścia z przedziału \([-1,1]\).

Wspólne kernele CUDA C: [`_src/kernels/langevin.cu`](_src/kernels/langevin.cu).  
Backendy: **`cupy_kernel`** / `pycuda` (wspólny [`langevin.cu`](_src/kernels/langevin.cu)), **`cupy_naive`** ([`cupy_naive.py`](_src/cupy_naive.py): `cp.random` + `cp.histogram`), `numba`, `cpu`. Aliasy CLI: `cupy`→`cupy_kernel`, `cupy_native`→`cupy_naive`.

### Wymagania (WSL2 + NVIDIA)

```bash
uv sync
```

Przy CuPy na WSL2 skrypty ustawiają `LD_LIBRARY_PATH=/usr/lib/wsl/lib` i `CUDA_PATH=/usr/lib/cuda` (patrz [`scripts/pipeline.sh`](scripts/pipeline.sh)).

### Lokalnie

```bash
# jeden przebieg (domyślnie density, cupy_kernel, N do 1e6)
bash local/run.sh [problem] [backend] "N_LIST"

# pełny sweep (density + mfpt × cupy_kernel, pycuda, numba, cupy_naive, cpu)
bash local/sweep.sh
```

**Wyniki:** `results/local/<data>_<problem>_<backend>_dt<dt>/`  
Struktura: `N<trajektorii>/rep<i>/` z `meta.json`, `data.npz`, wykresami PNG; w katalogu przebiegu też `time_vs_n.png`, `run_meta.json`, `log.txt`.

Po sweeie: `results/compare/` (`*_time_vs_n.png`, `summary.csv`) — `uv run python _src/compare.py`.

### Pojedynczy run

```bash
export PYTHONPATH=_src
bash scripts/pipeline.sh density cupy_kernel 1e-3 "1e3 1e4 1e5" 1 42
bash scripts/pipeline.sh density cupy_naive 1e-3 "1e5 1e6" 1 42
```

Kod: `_src/main.py` (punkt wejścia), `_src/analyze.py`, `_src/compare.py`.  
Raport: [REPORT.md](REPORT.md).