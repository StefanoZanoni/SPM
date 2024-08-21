#include <vector>
#include <indicators/progress_bar.hpp>

#include "mpimatrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "distributed.h"

#include <numeric>

void test_distributed(const int rank, const int mpi_world_size) {
    constexpr int dimensions[4]{1024, 2048, 4096, 8192};
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};

    if (rank == 0)
        std::cout << "Processing distributed with " << mpi_world_size << " processes..." << std::endl;

    indicators::ProgressBar bar {
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PostfixText{"Initializing..."},
        indicators::option::ForegroundColor{indicators::Color::yellow},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::MaxProgress{4}
    };

    for (const int dimension : dimensions) {
        if (rank == 0)
            bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        MPIMatrix matrix{dimension, rank, mpi_world_size};
        const double executionTime = measureExecutionTime([&matrix]() {
             matrix.set_upper_diagonals();
        });

        // Gather execution times from all processes
        std::vector<double> all_execution_times(mpi_world_size);
        MPI_Gather(&executionTime, 1, MPI_DOUBLE, all_execution_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            const double total_execution_time = std::accumulate(all_execution_times.begin(), all_execution_times.end(), 0.0);
            results.emplace_back(std::vector{static_cast<double>(dimension), total_execution_time / mpi_world_size});
            bar.tick();
        }
    }

    if (rank == 0) {
        writeCSV<double>("distributed_" + std::to_string(mpi_world_size) + ".csv", headers, results);
    }

}