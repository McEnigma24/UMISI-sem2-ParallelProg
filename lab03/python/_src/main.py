from mpi4py import MPI
import argparse
import numpy as np

comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()


class Membrane:
    def __init__(self, n: int, iters: int):
        self.p = 500
        self.T = 100
        self.max_iterations = iters

        self.a = 50
        self.how_many_partitions_1d = n
        self.da = self.a / self.how_many_partitions_1d

        shape = (self.how_many_partitions_1d, self.how_many_partitions_1d)
        self.h = np.zeros(shape, dtype=float)
        self.h_next = np.zeros(shape, dtype=float)

    def print_output(self):
        print(self.h.shape)
        print(self.h)

    def run_partition(self, y_start, y_end):
        for _ in range(self.max_iterations):
            for y in range(y_start, y_end):
                for x in range(1, self.how_many_partitions_1d - 1):
                    sum_of_all_neightbours = (
                        self.h[y - 1, x]
                        + self.h[y + 1, x]
                        + self.h[y, x - 1]
                        + self.h[y, x + 1]
                    )
                    self.h_next[y, x] = (
                        sum_of_all_neightbours + (self.p * self.da**2 / self.T)
                    ) / 4.0

            up = MPI_rank - 1 if MPI_rank > 0 else MPI.PROC_NULL
            down = MPI_rank + 1 if MPI_rank < MPI_size - 1 else MPI.PROC_NULL

            reqs = [
                comm.Isend(self.h_next[y_start], dest=up),
                comm.Irecv(self.h_next[y_start - 1], source=up),
                comm.Irecv(self.h_next[y_end], source=down),
                comm.Isend(self.h_next[y_end - 1], dest=down),
            ]
            MPI.Request.Waitall(reqs)

            self.h, self.h_next = self.h_next, self.h

    def start(self):
        interior_rows = self.how_many_partitions_1d - 2
        batch_size, remainder = divmod(interior_rows, MPI_size)

        y_start = 1 + MPI_rank * batch_size + min(MPI_rank, remainder)
        y_end = y_start + batch_size + (1 if MPI_rank < remainder else 0)

        self.run_partition(y_start, y_end)

        n = self.how_many_partitions_1d
        rows_per_rank = [
            batch_size + (1 if r < remainder else 0) for r in range(MPI_size)
        ]
        counts = [rows * n for rows in rows_per_rank]
        displs = [sum(counts[:r]) for r in range(MPI_size)]
        comm.Allgatherv(
            MPI.IN_PLACE, [self.h[1 : n - 1], counts, displs, MPI.DOUBLE]
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Membrana Jacobi (pętle) + MPI")
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

    membran = Membrane(args.n, args.iters)

    comm.Barrier()
    t0 = MPI.Wtime()
    membran.start()
    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    if MPI_rank == 0:
        print(f"loops,{args.n},{args.iters},{MPI_size},{elapsed:.6f}")
        if args.save:
            np.save(args.save, membran.h)
        if args.verbose:
            membran.print_output()
