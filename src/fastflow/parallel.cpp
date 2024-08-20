#include <vector>
#include <indicators/progress_bar.hpp>

#include "../utils/timer.h"
#include "../utils/csv.h"

#include "parallel.h"
#include "ffmatrix.h"


void test_parallel(const long maxnw) {
    constexpr long start = 1200;
    constexpr long end = 1300;
    std::vector<std::vector<double>> results;
    const std::vector<std::string> headers{"Dimension", "Execution Time"};
    double executionTime;

    std::cout << "Processing in parallel with " << maxnw << " threads..." << std::endl;

    {
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
            FFMatrix matrix{dimension};
            executionTime = measureExecutionTime([&matrix, maxnw]() {
                matrix.set_upper_diagonals(maxnw);
            });
            results.emplace_back(std::vector{static_cast<double>(dimension), executionTime});
            bar.tick();
        }

        writeCSV<double>("parallel_p.csv", headers, results);
        results.clear();
    }

    {
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

        std::cout << "Processing in parallel with 1 thread..." << std::endl;
        bar.set_progress(0);

        for (long dimension = start; dimension <= end; ++dimension) {
            bar.set_option(indicators::option::PostfixText{"Processing dimension " + std::to_string(dimension)});
            FFMatrix matrix{dimension};
            executionTime = measureExecutionTime([&matrix]() {
                matrix.set_upper_diagonals(1);
            });
            results.emplace_back(std::vector{static_cast<double>(dimension), executionTime});
            bar.tick();
        }

        writeCSV<double>("parallel_1.csv", headers, results);
    }
}
