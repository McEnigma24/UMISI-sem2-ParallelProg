# Lab04 — zbiór Julii (MPIPoolExecutor)

## Uruchomienie

**Lokalnie (WSL):**

```bash
cd lab04/python
bash local/run.sh          # domyślnie 8 workerów (9 procesów MPI)
bash local/run.sh 4        # 4 workery
```

Wyjście: `output/out`, wyniki: `results/` (CSV, PGM, `summary.txt`, wykresy w `results/plots/`).

**PLGrid (Ares):**

```bash
cd lab04/python
bash grid/run.sh           # sbatch → kolejka plgrid, 16 workerów + master
```

Job: [`grid/job`](grid/job) — `--partition=plgrid`, `--ntasks=17`, ten sam pipeline co lokalnie (`scripts/pipeline.sh grid`).

## Pliki

| Plik | Opis |
|------|------|
| [`_src/julia.py`](_src/julia.py) | Obliczenia + `MPIPoolExecutor`, pomiar czasu linii |
| [`_src/analyze.py`](_src/analyze.py) | Analiza zadań 3–6, wykresy |
| [`scripts/pipeline.sh`](scripts/pipeline.sh) | baseline → sleep → analyze |
| [`REPORT.md`](REPORT.md) | Odpowiedzi na polecenia + PCAM |

Wywołanie zgodne z treścią ćwiczenia:

```bash
mpiexec -n <workers+1> python -m mpi4py.futures _src/julia.py --tag baseline
```

(lokalnie przez `uv run mpiexec --oversubscribe ...`).
