#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <cmath>
#include <ff/parallel_for.hpp>

#include "../utils/matrix.h"

class FFMatrix final : public Matrix {

public:
    explicit FFMatrix(const long size) : Matrix(size) {}

    void set_upper_diagonals(const long maxnw = 0) const {
        ff::ParallelFor pf = (maxnw <= 0) ? ff::ParallelFor{true, true} : ff::ParallelFor{maxnw, true, true};

        // Iterate over upper diagonals
        for (long k = 1; k < size; ++k) {

            // Iterate over rows in parallel
            pf.parallel_for_static(0, size - k, 1, 0, [&](const long i) {

                alignas(64) double dot_product{0};
                for (long j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }

                data[index(i, i + k)] = std::cbrt(dot_product);

            });

        }
    }
};

#endif //SPM_FFMATRIX_H