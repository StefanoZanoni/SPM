#include <vector>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include "../utils/timer.h"
#include "../utils/csv.h"

#include "parallel.h"
#include "ffmatrix.h"


void test_parallel(const long maxnw) {
    const unsigned int start = 1;
    const unsigned int end = 2048;
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};
    double executionTime;

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
            indicators::option::MaxProgress{end - start + 1 + 2}
    };
    std::cout << "\nProcessing with " << maxnw << " threads...\n" << std::endl;

    for (unsigned int dimension = start; dimension <= end; ++dimension) {
        bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        FFMatrix matrix{dimension};
        executionTime = measureExecutionTime([&matrix, maxnw]() {
            matrix.set_upper_diagonals(maxnw);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
        bar.tick();
    }

    {
        bar.set_option(indicators::option::PostfixText{"Processing dimension 4096"});
        FFMatrix matrix{4096};
        executionTime = measureExecutionTime([&matrix, maxnw]() {
            matrix.set_upper_diagonals(maxnw);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(4096), executionTime});
        bar.tick();
    }

    {
        bar.set_option(indicators::option::PostfixText{"Processing dimension 8192"});
        FFMatrix matrix{8192};
        executionTime = measureExecutionTime([&matrix, maxnw]() {
            matrix.set_upper_diagonals(maxnw);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(8192), executionTime});
        bar.tick();
    }

    writeCSV<double>("parallel_p.csv", headers, results);

    results.clear();

    std::cout << "\nProcessing with 1 thread...\n" << std::endl;
    bar.set_option(indicators::option::MaxProgress{end - start + 1 + 2});
    bar.set_progress(0);

    for (unsigned int dimension = start; dimension <= end; ++dimension) {
        bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
        FFMatrix matrix{dimension};
        executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals(1);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
        bar.tick();
    }

    {
        bar.set_option(indicators::option::PostfixText{"Processing dimension 4096"});
        FFMatrix matrix{4096};
        executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals(1);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(4096), executionTime});
        bar.tick();
    }

    {
        bar.set_option(indicators::option::PostfixText{"Processing dimension 8192"});
        FFMatrix matrix{8192};
        executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals(1);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(8192), executionTime});
        bar.tick();
    }

    writeCSV<double>("parallel_1.csv", headers, results);
}
