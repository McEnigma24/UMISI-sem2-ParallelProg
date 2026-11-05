# Lab04 — raport: Julia + MPIPoolExecutor

Środowisko pomiarów lokalnych: **8 workerów**, **9 procesów MPI** (`NP = workers + 1`), obraz **1280×960**, uruchomienie: `bash local/run.sh 8`.  
Pełne liczby: [`results/summary.txt`](results/summary.txt), wykresy: [`results/plots/`](results/plots/).

---

## 1. Działający program (MPIPoolExecutor)

Program [`_src/julia.py`](_src/julia.py) bazuje na przykładzie z dokumentacji mpi4py.futures. Master (rank 0) rozdaje linie obrazu funkcją `executor.map(julia_line, range(h))`. Workery (ranki 1…8) wykonują `julia_line(k)`.

Uruchomienie:

```bash
mpiexec -n 9 python -m mpi4py.futures _src/julia.py --tag baseline
```

Wynik: `results/julia_baseline.pgm`, logi linii w stdout (`line=… rank=… time=…`).

---

## 2. Rozszerzenie `julia_line`

Dla każdej linii wypisywane są: **czas obliczeń**, **rank MPI** workera, **numer linii** (oraz zapis do `results/lines_<tag>.csv`).

---

## 3. Czy czasy zależą od numeru linii?

**Tak.** (baseline, lokalnie)

| | Wartość |
|---|--------|
| Min | 0,000357 s (linia 2) |
| Max | 0,006202 s (linia 644) |
| **Różnica max − min** | **0,005845 s** |

Linie blisko środka obrazu (dużo iteracji Mandelbrota/Julii) trwają dłużej niż linie przy krawędziach — widać to na wykresie `results/plots/line_time_baseline.png`.

---

## 4. Przydział linii — blokowy czy cykliczny?

| Hipoteza | Zgodność z danymi (baseline) |
|----------|------------------------------|
| Podział **blokowy** | **12,19%** |
| Podział **cykliczny** | **12,71%** |

**Wniosek:** nie da się uznać przydziału za blokowy ani cykliczny. `MPIPoolExecutor` przydziela zadania **dynamicznie** — wolny worker dostaje następną linię z kolejki. Przy starcie krótko widać „falę” przypisań (linie 0–7 po jednej na rank 1–8), potem kolejność zależy od tego, kto wcześniej skończył (linia 8 może iść na rank 3, a nie na rank 1).

**Liczba linii na workera (baseline):** ok. 119–124 (przy 960 liniach i 8 workerach; średnio 120).

---

## 5. Suma czasów workerów vs czas programu

**Baseline**

| Rank | Suma czasów linii [s] |
|------|------------------------|
| 1 | 0,275 |
| 2 | 0,277 |
| 3 | 0,288 |
| 4 | 0,283 |
| 5 | 0,282 |
| 6 | 0,285 |
| 7 | 0,278 |
| 8 | 0,283 |

- **Czas całkowity programu** (MPI.Wtime na masterze): **0,395 s**
- **Suma wszystkich czasów linii** (po workerach): **2,25 s** — to suma pracy, nie wall-clock; linie liczyły się równolegle.
- Stosunek suma/czas programu ≈ **5,7** — odpowiada efektywnemu wykorzystaniu ~6 workerów w tym przebiegu.
- Sumy po workerach są do siebie podobne (std ≈ 0,004 s), bo dynamiczny pool wyrównuje obciążenie mimo różnych czasów pojedynczych linii.

---

## 6. Eksperyment ze `sleep` na linii 1

- **Sleep:** linia **1**, czas **≈ 0,288 s** (= maks. suma czasów workera z baseline × 1,0).
- **Czas programu:** 0,395 s → **0,405 s** (+≈ 0,01 s; ścieżka krytyczna wydłużona głównie przez jedną „długą” linię).

| | baseline | sleep |
|---|----------|-------|
| Linie rank **2** | 115 | **39** |
| Suma czasów rank **2** | 0,277 s | **0,360 s** |
| Max czas linii | 0,006 s (linia 644) | **0,289 s (linia 1)** |

Worker, który dostał linię 1, spędził ~0,29 s w `sleep` i mniej czasu na kolejne linie — pozostali wzięli więcej zadań (np. rank 4: 121 → 134 linii). Ponowna analiza p. 4–5 potwierdza dynamiczny charakter przydziału; rozkład linii staje się mniej równomierny (rank 2 wyraźnie odstaje).

---

## 7. Analiza PCAM

| Składnik | Zastosowanie w tym zadaniu |
|----------|----------------------------|
| **P — Partitioning** | Jednostką zadania jest **scanline** (960 zadań). Możliwy mniejszy podział: piksel — więcej równoległości, większy narzut komunikacji. |
| **C — Communication** | Linie są **niezależne**; master wysyła indeks `k`, worker zwraca wiersz bajtów. Brak komunikacji między workerami. |
| **A — Agglomeration** | Grupowanie w **linię** (1280 pikseli) zmniejsza liczbę komunikatów względem zadania per-piksel. `executor.map` bez dużego `chunksize` = jedna linia na task. |
| **M — Mapping** | **Dynamiczny** master–worker (`MPIPoolExecutor`) zamiast statycznego blok/cykl — uzasadnione, bo czas linii zależy od `k` (patrz wykres `line_time_*`). Statyczny podział dałby nierównowagę (długie linie w środku obrazu). |

---

## PLGrid

Po wgraniu projektu na Ares: `bash grid/run.sh` (partycja **plgrid**, 16 workerów). Wyniki w `output/out_<jobid>.txt` oraz w `results/` na scratch/katalogu joba.
