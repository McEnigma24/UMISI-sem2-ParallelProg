# Lab07 — równoległy Metropolis, model Isinga 2D

## Algorytm
Siatka $L\times L$, Hamiltonian z $J=1$, $\beta$ steruje temperaturą. Jeden krok = 1 MCS ($L^2$ prób).
Dekompozycja pozioma na $P$ pasów; synchronizacja halo przy spinach brzegowych (`border`) lub leniwie (`smart`, Barkema).

Metryki: $S=T(1)/T(P)$, $E=S/P$, $f$ (Karp-Flatt).

PDF: `bash report/build.sh` → `report/report.pdf`.

## PCAM
Patrz sekcja w `report/report.tex`.

## Uruchomienie
`bash local/visualize.sh` ( $\beta\in\{1,4\}$ ), `bash local/sweep.sh` (skalowanie).