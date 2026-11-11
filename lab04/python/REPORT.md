# Lab04 — metodologia PCAM

Wyniki pomiarów (wykresy, `metrics.json`, `data.npz`) są w katalogach  
`results/runs/<data>_<W>x<H>_<N>w/baseline/` oraz `.../sleep/` — jeden mini-raport na przebieg, bez osobnego opisu liczb w tym pliku.

Uruchomienie lokalne: `bash local/run.sh [workers] [width] [height]`, sweep: `bash local/sweep.sh`.  
PLGrid: `bash grid/run.sh` (jeden job) lub `bash grid/sweep.sh` (macierz workerów × rozmiarów) — konfiguracja w tablicach w tych skryptach.

---

## P — Partitioning (podział)

- **Domena:** obraz płaszczyzny zespolonej dla zbioru Julii o wymiarach W×H.
- **Zadanie elementarne:** jedna **linia** (scanline) — indeks `k ∈ {0, …, H−1}`, 1280 pikseli na linię w domyślnej konfiguracji.
- **Alternatywy:** piksel (W×H zadań, ogromny narzut komunikacji) albo **blok linii** (większy `chunksize` w `executor.map` — mniej komunikatów, gorsza równowaga przy nierównych czasach linii).

---

## C — Communication (komunikacja)

- Linie są **niezależne** — brak wymiany danych między workerami.
- Master wysyła do workera argumenty zadania `(k, w, h, sleep_line, sleep_time)` i odbiera wynik (wiersz `uint8` + metadane czasu).
- W `MPIPoolExecutor` komunikacja to kolejka zadań i zwrot wyników; workery **nie** synchronizują się między sobą.

---

## A — Agglomeration (agregacja)

- Naturalna agregacja: **cała linia** jako jedno zadanie — łączy W obliczeń pikselowych w jednej wysyłce MPI/futures.
- Parametr **`--chunksize`**: grupowanie kilku linii w jednym zadaniu zmniejsza liczbę round-tripów do mastera kosztem mniejszej granularności load balancingu.
- Zapis wyniku po stronie mastera: prealokowane tablice `numpy` `(H, W)` — bez `append` na listach, gotowe pod dalszą analizę i ewentualny zapis do pliku.

---

## M — Mapping (mapowanie)

- **Dynamiczny** model master–worker: `MPIPoolExecutor.map` — wolny worker pobiera kolejną linię.
- **Uzasadnienie:** czas linii zależy od `k` (środek fraktala vs brzeg) — statyczny **podział blokowy** lub **cykliczny** prowadziłby do nierównowagi. W mini-raporcie mierzymy to indeksami `B`, `C` i `B_norm`, `C_norm` (zgodność z wzorcem blokowym/cyklicznym względem losowego przydziału).
- **Sleep na jednej linii:** symuluje długie zadanie — pool przekierowuje pracę do pozostałych workerów; widać to na mapie przydziału i wykresie liczby linii na proces.
