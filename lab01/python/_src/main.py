from mpi4py import MPI

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Hello, World! I am rank {rank} of {size}")

    comm.Barrier()

    print(f"Hello, World! I am rank {rank} of {size} after barrier")


