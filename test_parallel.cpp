#include "src/fastflow/parallel.h"
#include <iostream>
#include <limits>

int main(const int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_workers>" << std::endl;
        return 1;
    }
    long maxnw;

    try {
        const unsigned long ul_maxnw = std::stoul(argv[1]);
        if (ul_maxnw > static_cast<unsigned long>(std::numeric_limits<long>::max())) {
            std::cerr << "Invalid argument: " << argv[1] << " too many workers specified." << std::endl;
            return 1;
        }
        maxnw = static_cast<long>(ul_maxnw);
    } catch ([[maybe_unused]] const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is not a valid number." << std::endl;
        return 1;
    } catch ([[maybe_unused]] const std::out_of_range &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is out of range." << std::endl;
        return 1;
    }

    test_parallel(maxnw);

    return 0;

}