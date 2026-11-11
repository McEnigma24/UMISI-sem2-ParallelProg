# Lab05 — TSP, równoległy branch and bound

## Opis problemu

**TSP:** dane jest N miast i macierz odległości; szukamy cyklu Hamiltona o minimalnej długości.  
Miasta losujemy na okręgu jednostkowym (`--seed`); znane optimum to obwód wielokąta w kolejności kątów.

**Branch and bound:** przeszukiwanie drzewa permutacji z dolnym oszacowaniem kosztu; gałęzie z `lb >= best` są obcinane.  
W wersji równoległej **obcinanie opiera się wyłącznie na lokalnym `best`** w danym poddrzewie (brak globalnego `best` między workerami).

## Równoległość (MPIPoolExecutor)

1. Rank 0 generuje wszystkie prefiksy trasy zaczynające się w mieście 0, długości **d** (np. N=4, d=2: `[0,1]`, `[0,2]`, `[0,3]`).
2. Każdy prefiks to zadanie `solve_subtree` — dalsze B&B aż do pełnych tras.
3. `executor.map(..., chunksize=1)` — dynamiczny przydział zadań.
4. Czas **`wall_time`** mierzony od startu `main()` na ranku 0 (włącznie z generowaniem prefiksów).

Wykresy i CSV: `results/compare/{local,grid}/`.

## Metryki (dwa eksperymenty)

Dla ustalonego N i platformy; odniesienie **T(1)** — ten sam program z **1 workerem** i tym samym **d**.

| Symbol | Wzór |
|--------|------|
| Przyspieszenie S | T(1) / T(p) |
| Efektywność E | S / p |
| Serial fraction f (Karp-Flatt) | (1/S − 1/p) / (1 − 1/p), p > 1 |

**Eksperyment 1:** stałe d, rośnie liczba procesów p → `speedup_vs_p_N*.png`, `efficiency_vs_p_N*.png`, `serial_fraction_vs_p_N*.png` (osobne krzywe dla każdego d).

**Eksperyment 2:** stałe p, rośnie d → `*_vs_d_N*.png`.

Dodatkowo `nodes_vs_d_N*.png` — łączna liczba odwiedzonych węzłów B&B (pokazuje koszt braku globalnego pruning).

## PCAM

- **P — podział:** prefiksy głębokości d (N−1 permutowanych pozostałych miast w prefiksie); poddrzewo = dalsze permutacje + domknięcie cyklu.
- **C — komunikacja:** zadania i wyniki master ↔ worker; brak komunikacji między workerami, brak współdzielonego `best`.
- **A — agregacja:** jedno zadanie = całe poddrzewo od prefiksu; na masterze redukcja `min(cost)`.
- **M — mapowanie:** dynamiczny pool (`MPIPoolExecutor`); większe d → więcej, mniejszych zadań, lepsze balansowanie kosztem serial generacji prefiksów i słabszego globalnego pruning.

## Wnioski (lokalnie, N=12, seed=42)

Pełny sweep: `bash local/sweep.sh` → wykresy w `results/compare/local/`.

1. **d=1** (jedno zadanie): S **maleje** przy rosnącym p (nadmiar procesów, brak podziału); f blisko 1 — dominuje część sekwencyjna i narzut MPI.
2. **d=2…5**: większe **d** → więcej zadań; przy d=4…5 uzyskano S do ok. **5,7×** (11 workerów) przy E ~0,52–0,99 dla p=2…4.
3. **nodes_mean** rośnie z **d** (ten sam problem, więcej odwiedzonych węzłów przez lokalne B&B), ale przy większym **d** łatwiej uzyskać dobrą efektywność, bo lepsze balansowanie kosztem większej pracy całkowitej.
4. Wszystkie przebiegi: `correct=True` względem optimum na okręgu.

Na grid: `bash grid/run.sh` lub `grid/sweep.sh` (domyślnie N=14).
