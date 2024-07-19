#include <iostream>

#include "src/sequential/sequential.h"
#include "src/fastflow/parallel.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_workers>" << std::endl;
        return 1;
    }

    long maxnw;
    try {
        maxnw = static_cast<long>(std::stoul(argv[1]));
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is not a valid number." << std::endl;
        return 1;
    } catch (const std::out_of_range &e) {
        std::cerr << "Invalid argument: " << argv[1] << " is out of range." << std::endl;
        return 1;
    }

    test_sequential();
    test_parallel(maxnw);
}
