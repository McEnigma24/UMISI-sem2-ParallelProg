#include "__preprocessor__.h"
#include <mpi.h>


#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // CORE::clear_terminal(); // tests will NOT be VISIBLE with this line
    line("It just works");
    time_stamp("It just works - timestamped");



    // Inicjalizacja środowiska MPI
    MPI_Init(&argc, &argv);

    int world_size;
    // Pobranie całkowitej liczby procesów
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    // Pobranie identyfikatora (rank) obecnego procesu
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Wypisanie informacji
    std::cout << "Hello from process " << world_rank
              << " out of " << world_size << " processes!" << std::endl;

    // Zakończenie środowiska MPI
    MPI_Finalize();

    return 0;
}
#endif