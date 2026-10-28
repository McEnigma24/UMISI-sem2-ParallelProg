#!/bin/bash
source "_common_" 2>/dev/null

mkdir -p results results/benchmark_results
CSV="results/bench.csv"
echo "version,n,iters,p,time" > "$CSV"

PROCS=(1 2 3 4 6 8 12)
REPEATS=3

# Te same N i iters dla obu wersji — porównywalne krzywe S(P), E(P), f(P).
BENCH_NS=(50 100 200 300 400 500 600 700 800 900 1000)
BENCH_ITERS=100

run_bench() {
    local script=$1
    local version=$2
    local n=$3
    local iters=$4
    for p in "${PROCS[@]}"; do
        for ((r = 1; r <= REPEATS; r++)); do
            line=$(uv run mpiexec --oversubscribe -n "$p" python "$script" \
                --n "$n" --iters "$iters" 2>/dev/null | tail -1)
            echo "$line" >> "$CSV"
            echo "  $version n=$n iters=$iters P=$p run=$r -> $line"
        done
    done
}

for n in "${BENCH_NS[@]}"; do
    echo "=== N=$n, iters=$BENCH_ITERS ==="
    run_bench "_src/main.py" "loops" "$n" "$BENCH_ITERS"
    run_bench "_src/main_vec.py" "vec" "$n" "$BENCH_ITERS"
done

echo "Done. CSV: $CSV"
echo "Wykresy: uv run python _src/plot_metrics.py"
