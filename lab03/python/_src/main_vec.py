from mpi4py import MPI
import argparse
import numpy as np

comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()


class MembraneVec:
    def __init__(self, n: int, iters: int):
        self.p = 500
        self.T = 100
        self.max_iterations = iters

        self.a = 50
        self.n = n
        self.da = self.a / n
        self.c = self.p * self.da**2 / self.T

        self.h = None

    def start(self):
        n = self.n
        interior_rows = n - 2
        batch_size, remainder = divmod(interior_rows, MPI_size)

        y_start = 1 + MPI_rank * batch_size + min(MPI_rank, remainder)
        y_end = y_start + batch_size + (1 if MPI_rank < remainder else 0)
        local_rows = y_end - y_start

        local_h = np.zeros((local_rows + 2, n), dtype=float)
        local_h_next = np.zeros((local_rows + 2, n), dtype=float)

        up = MPI_rank - 1 if MPI_rank > 0 else MPI.PROC_NULL
        down = MPI_rank + 1 if MPI_rank < MPI_size - 1 else MPI.PROC_NULL

        for _ in range(self.max_iterations):
            if local_rows > 0:
                local_h_next[1:-1, 1:-1] = (
                    local_h[:-2, 1:-1]
                    + local_h[2:, 1:-1]
                    + local_h[1:-1, :-2]
                    + local_h[1:-1, 2:]
                    + self.c
                ) / 4.0

                reqs = [
                    comm.Isend(local_h_next[1], dest=up),
                    comm.Irecv(local_h_next[0], source=up),
                    comm.Irecv(local_h_next[-1], source=down),
                    comm.Isend(local_h_next[-2], dest=down),
                ]
                MPI.Request.Waitall(reqs)

            local_h, local_h_next = local_h_next, local_h

        rows_per_rank = [
            batch_size + (1 if r < remainder else 0) for r in range(MPI_size)
        ]
        counts = [rows * n for rows in rows_per_rank]
        displs = [sum(counts[:r]) for r in range(MPI_size)]

        if MPI_rank == 0:
            self.h = np.zeros((n, n), dtype=float)
            recvbuf = [self.h[1 : n - 1], counts, displs, MPI.DOUBLE]
        else:
            recvbuf = None

        if local_rows > 0:
            sendbuf = local_h[1:-1]
        else:
            sendbuf = np.empty(0, dtype=float)

        comm.Gatherv(sendbuf, recvbuf, root=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Membrana Jacobi (numpy) + MPI")
    parser.add_argument("--n", type=int, default=1000, help="rozmiar siatki N×N")
    parser.add_argument("--iters", type=int, default=100, help="liczba iteracji Jacobi")
    parser.add_argument(
        "--save", type=str, default=None, help="ścieżka zapisu h (.npy) na rank 0"
    )
    parser.add_argument("--verbose", action="store_true", help="wypisz macierz h")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        print(f"Hello, World! I am rank {MPI_rank} of {MPI_size}")

    membran = MembraneVec(args.n, args.iters)

    comm.Barrier()
    t0 = MPI.Wtime()
    membran.start()
    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    if MPI_rank == 0:
        print(f"vec,{args.n},{args.iters},{MPI_size},{elapsed:.6f}")
        if args.save:
            np.save(args.save, membran.h)
        if args.verbose:
            print(membran.h.shape)
            print(membran.h)
