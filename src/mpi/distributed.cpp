#include <vector>
#include <indicators/progress_bar.hpp>

#include "mpimatrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "distributed.h"

void test_distributed(const int rank, const int mpi_size) {
    constexpr int start = 5;
    constexpr int end = 5;
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};

    std::cout << "Processing distributed..." << std::endl;

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
            indicators::option::MaxProgress{end - start + 1}
    };

    for (int dimension = start; dimension <= end; ++dimension) {
        bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        MPIMatrix matrix{dimension, rank, mpi_size};
        const double executionTime = measureExecutionTime([&matrix]() {
             matrix.set_upper_diagonals();
         });
        matrix.print();
        results.emplace_back(std::vector{static_cast<double>(dimension), executionTime});
        bar.tick();
    }

    writeCSV<double>("sequential.csv", headers, results);

}
