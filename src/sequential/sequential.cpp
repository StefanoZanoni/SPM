#include <vector>
#include <indicators/progress_bar.hpp>

#include "seqmatrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "sequential.h"


void test_sequential() {
    constexpr int dimensions[4]{1024, 2048, 4096, 8192};
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
            indicators::option::MaxProgress{4}
    };

    for (const int dimension : dimensions) {
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