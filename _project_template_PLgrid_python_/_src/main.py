from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# sequential algo
# parallel algo  => partition sequential algo + communitaion

if __name__ == "__main__":

    print(f"Hello, World! I am rank {rank} of {size}")

    comm.Barrier()

    print(f"Hello, World! I am rank {rank} of {size} after barrier")
