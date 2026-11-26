# Lab07 — model Isinga, równoległy Metropolis (MPI + Numba)

### Lokalnie

```bash
bash local/run.sh [P] [L] [steps] [beta] [sync] [repeats] [seed]   # domyślnie 4 256 1000 1 border
bash local/visualize.sh                                            # beta=1 i beta=4, L=256, t=10000
bash local/sweep.sh                                                # skalowanie (kalibracja STEPS w sweep)
```

**Wyniki:** `results/local/<data>_L<L>_<sync>_<P>p/rep<i>/` + `log.txt`  
Po sweeie: `results/compare/local/` (S, E, f) i `summary_all.csv`.

### PLGrid (Ares, partycja `plgrid`)

```bash
bash grid/run.sh
bash grid/sweep.sh
```

Kod: `_src/ising.py`, `_src/kernels.py`, `_src/analyze.py`, `_src/compare.py`, `scripts/pipeline.sh`.  
Raport PDF: `bash report/build.sh` → `report/report.pdf`.