#!/usr/bin/env python
import math
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()


def sieve_basic(limit: int) -> list[int]:
    """Klasyczne sito Eratostenesa na [2, limit]."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = b"\x00" * ((limit - p * p) // p + 1)
    return [i for i in range(2, limit + 1) if is_prime[i]]


def segment_sieve(lo: int, hi: int, divisors: list[int]) -> list[int]:
    """Sito segmentowe na [lo, hi] z podzielnikami z przedziału B."""
    if lo > hi:
        return []
    seg_len = hi - lo + 1
    is_prime = bytearray(b"\x01") * seg_len
    for p in divisors:
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start > hi:
            continue
        offset = start - lo
        is_prime[offset::p] = b"\x00" * ((hi - start) // p + 1)
    return [lo + i for i in range(seg_len) if is_prime[i]]


def partition_c(sqrt_n: int, n: int, rank: int, size: int) -> tuple[int, int]:
    """Dekompozycja domenowa C = [sqrt_n+1, n]."""
    c_start = sqrt_n + 1
    if c_start > n:
        return c_start, c_start - 1
    c_len = n - sqrt_n
    chunk = (c_len + size - 1) // size
    lo = c_start + rank * chunk
    hi = min(lo + chunk - 1, n)
    return lo, hi


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    comm.Barrier()
    t0 = MPI.Wtime()

    sqrt_n = math.isqrt(n)

    # Krok 1: każdy proces sam liczy podzielniki z B (replikacja)
    divisors = sieve_basic(sqrt_n)

    # Krok 2–3: własny podprzedział C i sito segmentowe
    lo, hi = partition_c(sqrt_n, n, MPI_rank, MPI_size)
    local_primes = segment_sieve(lo, hi, divisors)

    # Krok 4: proces 0 zbiera wyniki
    all_parts = comm.gather(local_primes, root=0)

    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    if MPI_rank == 0:
        if n < 2:
            primes = []
        else:
            primes = divisors + [p for part in all_parts for p in part]

        expected = sieve_basic(n)
        ok = primes == expected

        print(f"n = {n}, procesów = {MPI_size}, sqrt(n) = {sqrt_n}")
        print(f"Czas: {elapsed:.6f} s")
        print(f"Liczba liczb pierwszych: {len(primes)} (oczekiwane: {len(expected)})")
        print(f"Weryfikacja: {'OK' if ok else 'MISMATCH'}")

        for r in range(MPI_size):
            r_lo, r_hi = partition_c(sqrt_n, n, r, MPI_size)
            count = len(all_parts[r]) if all_parts else 0
            print(f"  rank {r}: [{r_lo}, {r_hi}] -> {count} liczb pierwszych")

        if len(primes) <= 100:
            print(f"Wynik: {primes}")
        elif primes:
            print(f"Pierwsze 10: {primes[:10]}")
            print(f"Ostatnie 10: {primes[-10:]}")
