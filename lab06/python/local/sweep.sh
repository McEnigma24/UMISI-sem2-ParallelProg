#!/bin/bash
source "_common_" 2>/dev/null

PROBLEMS=(density mfpt)
BACKENDS=(cupy_kernel pycuda numba)
N_LIST="1e3 1e4 1e5 3e5 1e6 3e6 1e7 3e7 1e8"
N_LIST_CPU="1e3 1e4 1e5 3e5 1e6"
# cupy_naive MFPT: pętla cp.any co krok — nie ciągniemy do 1e8
N_LIST_NAIVE_MFPT="1e3 1e4 1e5 3e5 1e6"
DT="1e-3"
REPEATS=1
SEED=42

mkdir -p results/local results/compare output

for problem in "${PROBLEMS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        echo "######## sweep problem=${problem} backend=${backend} ########"
        bash scripts/pipeline.sh "$problem" "$backend" "$DT" "$N_LIST" "$REPEATS" "$SEED"
    done
    echo "######## sweep problem=${problem} backend=cupy_naive ########"
    if [[ "$problem" == "density" ]]; then
        bash scripts/pipeline.sh "$problem" "cupy_naive" "$DT" "$N_LIST" "$REPEATS" "$SEED"
    else
        bash scripts/pipeline.sh "$problem" "cupy_naive" "$DT" "$N_LIST_NAIVE_MFPT" "$REPEATS" "$SEED"
    fi
    echo "######## sweep problem=${problem} backend=cpu ########"
    bash scripts/pipeline.sh "$problem" "cpu" "$DT" "$N_LIST_CPU" "$REPEATS" "$SEED"
done

uv run python _src/compare.py