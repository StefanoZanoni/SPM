#include <iostream>
#include <limits>
#include <mpi.h>

#include "src/sequential/sequential.h"
#include "src/fastflow/parallel.h"
#include "src/mpi/distributed.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_workers>" << std::endl;
        return 1;
    }
    long maxnw;

    try {
        const unsigned long ul_maxnw = std::stoul(argv[1]);
        if (ul_maxnw > static_cast<unsigned long>(std::numeric_limits<long>::max())) {
            std::cerr << "Invalid argument: " << argv[1] << " too many workers specified." << std::endl;
            return 1;
        }
        maxnw = static_cast<long>(ul_maxnw);
    } catch ([[maybe_unused]] const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is not a valid number." << std::endl;
        return 1;
    } catch ([[maybe_unused]] const std::out_of_range &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is out of range." << std::endl;
        return 1;
    }

    test_sequential();
    test_parallel(maxnw);

    MPI_Init(&argc, &argv);

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    test_distributed(rank, mpi_size);

    MPI_Finalize();

    return 0;
}
