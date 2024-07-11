#include <vector>

#include "../utils/timer.h"
#include "../utils/csv.h"

#include "parallel.h"
#include "ffmatrix.h"


void testParallel() {
    std::vector<std::vector<double>> results;
    for (unsigned int dimension = 1; dimension <= 1000; ++dimension) {
        FFMatrix matrix{dimension};
        double executionTime = measureExecutionTime([&matrix]() {
            matrix.setUpperDiagonals();
        });
        results.emplace_back(std::vector<double>{static_cast<double>(dimension), executionTime});
    }
    std::vector<std::string> headers{"Dimension", "Execution Time"};
    writeCSV<double>("parallel.csv", headers, results);
}
