#include <mpi.h>

#include "src/mpi/distributed.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    test_distributed(rank, mpi_size);

    MPI_Finalize();

    return 0;
}