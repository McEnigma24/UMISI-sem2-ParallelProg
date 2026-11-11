# Lab04 — Julia + MPIPoolExecutor

### Lokalnie

```bash
bash local/run.sh [workers] [width] [height]   # domyślnie 8, 1280, 960
bash local/sweep.sh                             # WORKERS × SIZES w pliku
```

**Wyniki:** `results/local/<data>_<W>x<H>_<N>w/{baseline,sleep}/` + `log.txt`  
Na końcu sweepu: `results/compare/` (local vs grid).

### PLGrid (Ares, partycja `plgrid`)

```bash
bash grid/run.sh      # jeden job — edytuj WORKERS i SIZES_ITEM
bash grid/sweep.sh
```

**Wyniki:** `results/grid/...`  
Po pobraniu: `uv run python _src/compare.py`

Przy największym rozmiarze sweep ma też warianty: **szerszy** (więcej W), **dłuższy** (więcej H), **mniejszy 4:3** (mniej Mpx, podobne proporcje).

Kod: `_src/julia.py`, `_src/analyze.py`, `_src/compare.py`, `scripts/pipeline.sh`.  
Raport: [REPORT.md](REPORT.md).
