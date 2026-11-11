# Lab04 — metodologia PCAM

Wyniki pomiarów (wykresy, `metrics.json`, `data.npz`):

- `results/local/` i `results/grid/` — `<data>_<W>x<H>_<N>w/baseline/` i `.../sleep/`
- porównanie platform: `results/compare/` (`uv run python _src/compare.py`)

Stare katalogi `results/map/` i `results/submit/` można zignorować lub usunąć.

Uruchomienie: `bash local/run.sh`, sweep `bash local/sweep.sh`; PLGrid: `bash grid/run.sh` / `bash grid/sweep.sh`.

Na największym progu rozmiaru (lokalnie ok. 2560×1920, na gridzie 6400×4800) dodatkowo mierzymy obrazy **szersze**, **dłuższe** i **mniejsze przy tym samym 4:3** — inna liczba zadań (H) vs dłuższa linia (W).

---

## P — Partitioning (podział)

- **Domena:** obraz płaszczyzny zespolonej dla zbioru Julii o wymiarach W×H.
- **Zadanie elementarne:** jedna **linia** (scanline) — indeks `k ∈ {0, …, H−1}`.
- **Alternatywy:** piksel (W×H zadań, ogromny narzut komunikacji) albo **blok linii** (większy `chunksize` w `executor.map`).

---

## C — Communication (komunikacja)

- Linie są **niezależne** — brak wymiany danych między workerami.
- Master wysyła do workera argumenty zadania `(k, w, h, sleep_line, sleep_time)` i odbiera wynik (wiersz `uint8` + metadane czasu).
- W `MPIPoolExecutor` komunikacja to kolejka zadań i zwrot wyników; workery **nie** synchronizują się między sobą.

---

## A — Agglomeration (agregacja)

- Naturalna agregacja: **cała linia** jako jedno zadanie — łączy W obliczeń pikselowych w jednej wysyłce MPI/futures.
- Parametr **`--chunksize`**: grupowanie kilku linii w jednym zadaniu.
- Zapis wyniku po stronie mastera: prealokowane tablice `numpy` `(H, W)`.

---

## M — Mapping (mapowanie)

- **Dynamiczny** model master–worker: `MPIPoolExecutor.map` — wolny worker pobiera kolejną linię.
- **Uzasadnienie:** czas linii zależy od `k` — statyczny podział blokowy/cykliczny prowadziłby do nierównowagi (`B`, `C`, `B_norm`, `C_norm` w mini-raporcie).
- **Sleep na jednej linii:** symuluje długie zadanie — widać to na mapie przydziału i wykresie liczby linii na proces.
