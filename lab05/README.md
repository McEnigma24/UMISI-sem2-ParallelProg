# Lab05 — TSP, równoległy branch and bound (MPIPoolExecutor)

### Lokalnie

```bash
bash local/run.sh [workers] [N] [d] [repeats] [seed]   # domyślnie 4, 12, 2, 1, 42
bash local/sweep.sh                                    # N ze SWEEP_N (domyślnie 12)
```

**Wyniki:** `results/local/<data>_N<N>_d<D>_<P>w/rep<i>/` + `log.txt`  
Po sweeie: `results/compare/local/` (S, E, f) i `summary_all.csv`.

### PLGrid (Ares, partycja `plgrid`)

```bash
bash grid/run.sh      # jeden job — edytuj N, WORKERS, D_LIST
bash grid/sweep.sh
```

**Wyniki:** `results/grid/...`  
Po pobraniu: `uv run python _src/compare.py`

Kod: `_src/tsp.py`, `_src/analyze.py`, `_src/compare.py`, `scripts/pipeline.sh`.  
Raport: [REPORT.md](REPORT.md).
