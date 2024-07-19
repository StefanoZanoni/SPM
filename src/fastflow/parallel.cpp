#include <vector>

#include "../utils/timer.h"
#include "../utils/csv.h"

#include "parallel.h"
#include "ffmatrix.h"


void test_parallel(const long maxnw) {
    unsigned int start = 1000;
    unsigned int end = 2000;
    std::vector<std::vector<double>> results;
    std::vector<std::string> headers{"Dimension", "Execution Time"};

    for (unsigned int dimension = start; dimension <= end; ++dimension) {
        FFMatrix matrix{dimension};
        double executionTime = measureExecutionTime([&matrix, maxnw]() {
            matrix.set_upper_diagonals(maxnw);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
    }
    writeCSV<double>("parallel_p.csv", headers, results);

    results.clear();

    for (unsigned int dimension = start; dimension <= end; ++dimension) {
        FFMatrix matrix{dimension};
        double executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals(1);
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
    }
    writeCSV<double>("parallel_1.csv", headers, results);
}
