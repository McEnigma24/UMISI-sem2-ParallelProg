from mpi4py import MPI
import numpy as np
import math

comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()

class Membrane:
    def __init__(self):
        self.p = 100        # ciśnienie na membranie
        self.T = 100        # napięcie membrany
        self.max_iterations = 10

        self.a = 50                                         # bok całej membrany
        self.how_many_partitions_1d = 1000                  # tyle partycji per 1 bok
        self.da = self.a // self.how_many_partitions_1d     # odległość pomiędzy komórkami

        shape = (self.how_many_partitions_1d, self.how_many_partitions_1d)
        self.h = np.zeros(shape, dtype=float)
        self.h_next = np.zeros(shape, dtype=float)

    def distance_between_points(self, x, y, other_x, other_y):
        # in our case its always da #
        return self.da

    # def clamp_values(self, x_start, x_end, y_start, y_end):
    #     x_start = math.clamp(x_start, 1, self.how_many_partitions_1d - 1)
    #     x_end = math.clamp(x_end, 1, self.how_many_partitions_1d - 1)
    #     y_start = math.clamp(y_start, 1, self.how_many_partitions_1d - 1)
    #     y_end = math.clamp(y_end, 1, self.how_many_partitions_1d - 1)
    #     return (x_start, x_end, y_start, y_end)

    def run_partition(self, y_start, y_end):

        for i in range(self.max_iterations):
            for y in range(y_start, y_end):
                for x in range(1, self.how_many_partitions_1d - 1):

                    sum_of_all_neightbours = (
                        self.h[y - 1, x] + self.h[y + 1, x]
                        + self.h[y, x - 1] + self.h[y, x + 1]
                    )
                    self.h_next[y, x] = (
                        sum_of_all_neightbours
                        + (self.p * self.da**2 / self.T)
                    ) / 4.0

            # Komunikacja jeśli jest coś    nad y_start   i   pod y_end

            self.h = self.h_next
            self.h_next = self.h

            # print(f"Completed iteration: {i}")

    def start(self):
        # dzielimy tylko wiersze wewnętrzne 1..n-2, wiersze 0 i n-1 to brzeg
        interior_rows = self.how_many_partitions_1d - 2
        batch_size, remainder = divmod(interior_rows, MPI_size)

        # pierwsze `remainder` procesów dostaje o 1 wiersz więcej
        y_start = 1 + MPI_rank * batch_size + min(MPI_rank, remainder)
        y_end = y_start + batch_size + (1 if MPI_rank < remainder else 0)   # wyłącznie, jak w range

        # n = 50, MPI_size = 5 -> 48 wierszy wewnętrznych = 5 * 9 + 3
        # rank    | 0  | 1  | 2  | 3  | 4
        # y_start | 1  | 11 | 21 | 31 | 40
        # y_end   | 11 | 21 | 31 | 40 | 49

        self.run_partition(y_start, y_end)




if __name__ == "__main__":

    print(f"Hello, World! I am rank {MPI_rank} of {MPI_size}")

    membran = Membrane()
    membran.run()
