#!/usr/bin/env python
"""
Równoległe sito Eratostenesa — wariant z pamięcią współdzieloną (MPI Win.Allocate_shared).

Kroki 1–3 jak w message passing: replikacja sita na B, dekompozycja C, sito segmentowe.
Zamiast gather: każdy proces zapisuje flagi pierwszości swojego podprzedziału C do
wspólnej tablicy w oknie shared memory; rank 0 uzupełnia też przedział B.
"""
import math
import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()


def sieve_basic(limit: int) -> list[int]:
    if limit < 2:
        return []
    flags = sieve_flags(limit)
    return [i for i in range(2, limit + 1) if flags[i]]


def sieve_flags(limit: int) -> bytearray:
    if limit < 2:
        return bytearray()
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = b"\x00" * ((limit - p * p) // p + 1)
    return is_prime


def segment_sieve_flags(lo: int, hi: int, divisors: list[int]) -> bytearray:
    if lo > hi:
        return bytearray()
    seg_len = hi - lo + 1
    is_prime = bytearray(b"\x01") * seg_len
    for p in divisors:
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start > hi:
            continue
        offset = start - lo
        is_prime[offset::p] = b"\x00" * ((hi - start) // p + 1)
    return is_prime


def partition_c(sqrt_n: int, n: int, rank: int, size: int) -> tuple[int, int]:
    c_start = sqrt_n + 1
    if c_start > n:
        return c_start, c_start - 1
    c_len = n - sqrt_n
    chunk = (c_len + size - 1) // size
    lo = c_start + rank * chunk
    hi = min(lo + chunk - 1, n)
    return lo, hi


def primes_from_flags(flags: np.ndarray, n: int) -> list[int]:
    if n < 2:
        return []
    return [i for i in range(2, n + 1) if flags[i]]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    comm.Barrier()
    t0 = MPI.Wtime()

    sqrt_n = math.isqrt(n)
    divisors = sieve_basic(sqrt_n)

    dtype = np.uint8
    itemsize = np.dtype(dtype).itemsize
    nbytes = (n + 1) * itemsize if MPI_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

    buf_ptr, _ = win.Shared_query(0)
    flags = np.ndarray(buffer=buf_ptr, dtype=dtype, shape=(n + 1,))

    win.Fence()

    if n >= 2:
        if MPI_rank == 0:
            b_flags = sieve_flags(sqrt_n)
            flags[0] = 0
            flags[1] = 0
            flags[2 : sqrt_n + 1] = np.frombuffer(
                b_flags[2 : sqrt_n + 1], dtype=dtype, count=sqrt_n - 1
            )

        lo, hi = partition_c(sqrt_n, n, MPI_rank, MPI_size)
        if lo <= hi:
            seg = segment_sieve_flags(lo, hi, divisors)
            flags[lo : hi + 1] = np.frombuffer(seg, dtype=dtype, count=hi - lo + 1)

    win.Fence()

    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    if MPI_rank == 0:
        primes = primes_from_flags(flags, n)
        expected = sieve_basic(n)
        ok = primes == expected

        print(f"[shared memory] n = {n}, procesów = {MPI_size}, sqrt(n) = {sqrt_n}")
        print(f"Czas: {elapsed:.6f} s")
        print(f"Liczba liczb pierwszych: {len(primes)} (oczekiwane: {len(expected)})")
        print(f"Weryfikacja: {'OK' if ok else 'MISMATCH'}")

        for r in range(MPI_size):
            r_lo, r_hi = partition_c(sqrt_n, n, r, MPI_size)
            if r_lo > r_hi:
                count = 0
            else:
                count = int(np.sum(flags[r_lo : r_hi + 1]))
            print(f"  rank {r}: [{r_lo}, {r_hi}] -> {count} liczb pierwszych")

        if len(primes) <= 100:
            print(f"Wynik: {primes}")
        elif primes:
            print(f"Pierwsze 10: {primes[:10]}")
            print(f"Ostatnie 10: {primes[-10:]}")

    win.Free()
