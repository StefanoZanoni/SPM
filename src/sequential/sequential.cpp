#include <vector>
#include <indicators/progress_bar.hpp>

#include "seqmatrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "sequential.h"


void test_sequential() {
    constexpr long start = 1200;
    constexpr long end = 1300;
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};

    std::cout << "Processing sequentially..." << std::endl;

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

    for (long dimension = start; dimension <= end; ++dimension) {
        bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        SeqMatrix matrix{dimension};
        const double executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals();
        });
        results.emplace_back(std::vector{static_cast<double>(dimension), executionTime});
        bar.tick();
    }

    writeCSV<double>("sequential.csv", headers, results);
}
