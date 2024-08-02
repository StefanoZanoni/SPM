#include <vector>
#include <indicators/progress_bar.hpp>
#include <mpi.h>
#include <mm_malloc.h>

#include "mpimatrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "distributed.h"

void test_distributed(int rank, int mpi_size) {
    const int start = 5;
    const int end = 5;
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};
    double executionTime;

    std::cout << "Processing distributed..." << std::endl;

    indicators::ProgressBar bar{
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
            indicators::option::MaxProgress{end - start + 1}
    };

    for (int dimension = start; dimension <= end; ++dimension) {
        bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        MPIMatrix matrix{rank, mpi_size, dimension};
        executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals();
        });
        matrix.print();
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
        bar.tick();
    }

    writeCSV<double>("sequential.csv", headers, results);

}
