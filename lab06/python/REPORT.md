# Lab06 — równanie Langevina na GPU

## Opis problemu

**Langevin (bez siły zewnętrznej):** \(\mathrm{d}x/\mathrm{d}t = \xi(t)\), biały szum gaussowski.  
Dyskretyzacja Euler–Maruyama:

\[
x(t+\Delta t) = x(t) + \sqrt{\Delta t}\,\xi_t,\quad \xi_t \sim \mathcal{N}(0,1).
\]

**Zadanie 2 — gęstość:** symulujemy \(N\) niezależnych trajektorii od \(x(0)=0\) do \(\tau=1\), histogram \(x(\tau)\).  
Teoretycznie \(p(x,\tau)=\mathcal{N}(0,\tau)\) (z \(D=\tfrac12\) w skali Itô odpowiada wariancja \(\tau\)).

**Zadanie 3 — MFPT:** integrujemy aż \(|x|>1\), zapisujemy czas wyjścia; średnia po trajektoriach.  
Dla \(x_0=0\) i absorbing boundaries \(\pm1\): \(\langle\tau\rangle = 1\) (analitycznie).  
Przy dyskretnym kroku czasowym i sprawdzaniu granicy po kroku występuje **bias rzędu \(\mathcal{O}(\sqrt{\Delta t})\)** (literatura o SDE / Mannella) — oczekujemy \(\langle\tau\rangle\) nieco powyżej 1 przy \(\Delta t=10^{-3}\).

## Mapowanie na GPU (PCAM)

| | |
|---|---|
| **P — podział** | Jedna trajektoria = jeden wątek (grid-stride po \(N\)); każdy wątek ma własny strumień Philox (`curand_init(seed, traj_id, …)`). |
| **C — komunikacja** | Brak między wątkami w fazie symulacji. Histogram / redukcja: atomiki w shared memory, potem atomik do globalnej tablicy (wykład CUDA). |
| **A — agregacja** | Histogram binów (gęstość) lub suma czasów wyjścia + licznik (MFPT). |
| **M — mapowanie** | Trajektoria \(\leftrightarrow\) wątek CUDA; bloki 256 wątków, siatka \(\min(\lceil N/256\rceil, 65535)\). |

**Nierówna długość trajektorii (MFPT):** wątki w warpach kończą w różnych momentach → dywergencja (niewykorzystane PE w warpach). To typowy koszt „jedna trajektoria na wątek”.

## Implementacja

- Kernele: [`_src/kernels/langevin.cu`](_src/kernels/langevin.cu) — `density_kernel`, `histogram_kernel`, `mfpt_kernel`, `sum_reduce_kernel`.
- **CuPy:** `RawModule` + zdarzenia CUDA; **PyCUDA:** `SourceModule` (`-ccbin g++-13`).
- Metryki czasu: `t_kernel_sim`, `t_kernel_post`, `t_d2h`, `t_total` (wall clock dla danego \(N\)).
- Poprawność: L1 dystans histogramu vs \(\mathcal{N}(0,\tau)\); dla MFPT \|\(\langle\tau\rangle - 1\|\).

## Wyniki (lokalnie, RTX 3060, \(\Delta t=10^{-3}\), \(\tau=1\))

Pełny sweep: `bash local/sweep.sh` → 6 przebiegów, `results/compare/summary.csv` (50 punktów).

| Problem | Backend | \(N=10^8\) | \(t_\text{total}\) | Jakość |
|---------|---------|-----------|-------------------|--------|
| density | cupy | tak | 0.96 s | L1 = 0.0008 |
| density | pycuda | tak | 1.13 s | L1 = 0.0008 |
| density | cpu | nie (do \(10^6\)) | 8.17 s @ \(10^6\) | L1 = 0.0075 |
| mfpt | cupy | tak | 4.91 s | \(\langle\tau\rangle = 1.0374\) |
| mfpt | pycuda | tak | 4.95 s | \(\langle\tau\rangle = 1.0374\) |
| mfpt | cpu | nie (do \(10^6\)) | 311 s @ \(10^6\) | \(\langle\tau\rangle = 1.0367\) |

Wykresy: `results/compare/density_time_vs_n.png`, `mfpt_time_vs_n.png`, `*_error_vs_n.png`.

### Wnioski wstępne

1. **Skalowanie czasu:** po przejściu regionu stałego narzutu \(t_\text{total}\) rośnie w przybliżeniu liniowo z \(N\) (gęstość: \(N \times \tau/\Delta t\) kroków na trajektorię).
2. **CuPy vs PyCUDA:** ten sam kod CUDA C — różnice to głównie kompilacja cache, alokacje i pomiar; wyniki numeryczne (L1, \(\langle\tau\rangle\)) pokrywają się.
3. **Poprawność gęstości:** błąd L1 maleje jak \(1/\sqrt{N}\) (por. `*_error_vs_n.png`).
4. **MFPT:** średnia zgodna z teorią w granicy \(\Delta t\to 0\); przy \(10^{-3}\) widoczny dodatni bias \(\sqrt{\Delta t}\).
5. **CPU (NumPy):** referencja tylko dla małych \(N\) (sweep do \(10^6\)); MFPT jest wolny (pętla po trajektoriach).

## Uruchomienie

```bash
bash local/run.sh density cupy "1e3 1e4 1e5 1e6 1e7"
uv run python _src/compare.py
```