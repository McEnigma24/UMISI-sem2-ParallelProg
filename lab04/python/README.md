# Lab04 — Julia + MPIPoolExecutor

### Lokalnie

```bash
bash local/run.sh [workers] [width] [height]   # domyślnie 8, 1280, 960
bash local/sweep.sh                             # tablice WORKERS i SIZES w pliku
```

### PLGrid (Ares, partycja `plgrid`)

```bash
bash grid/run.sh      # jeden job — edytuj WORKERS i SIZES_ITEM w pliku
bash grid/sweep.sh    # pętla workerów × rozmiarów, wiele sbatch
bash grid/status.sh   # status ostatnich jobów (jeśli używasz)
```

Zgłoszone job ID: `grid/submitted_jobs.txt`, ostatni pojedynczy: `grid/.latest_job_started`.

Wyniki jak lokalnie: `results/runs/<data>_<W>x<H>_<N>w/{baseline,sleep}/` + `log.txt`.  
Slurm stdout: `output/out_<jobid>.txt`.

Kod: `_src/julia.py`, `_src/analyze.py`, `scripts/pipeline.sh`.  
Raport (PCAM): [REPORT.md](REPORT.md).
