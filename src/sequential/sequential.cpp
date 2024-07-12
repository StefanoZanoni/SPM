#include <vector>

#include "../utils/matrix.h"
#include "../utils/timer.h"
#include "../utils/csv.h"

#include "sequential.h"


void test_sequential() {
    std::vector<std::vector<double>> results;
    for (unsigned int dimension = 1; dimension <= 1000; ++dimension) {
        UTMatrix matrix{dimension};
        double executionTime = measureExecutionTime([&matrix]() {
            matrix.set_upper_diagonals();
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
    }
    std::vector<std::string> headers{"Dimension", "Execution Time"};
    writeCSV<double>("sequential.csv", headers, results);
}
