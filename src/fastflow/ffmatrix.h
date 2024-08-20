#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <immintrin.h>
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

                // Use AVX2 to speed up the dot product calculation
                long j = 0;
                __m256d sum = _mm256_setzero_pd();
                for (; j <= k - 4; j += 4) {
                    const __m256d row = _mm256_loadu_pd(&data[index(i, i + j)]);
                    const __m256d column = _mm256_loadu_pd(&data_t[index(i + k, i + 1 + j)]);
                    sum = _mm256_fmadd_pd(row, column, sum);
                }

                // Sum the elements of the AVX register
                alignas(32) double dot_product[4];
                _mm256_store_pd(dot_product, sum);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                // Handle remaining elements
                for (; j < k; ++j) {
                    dot_product[0] += data[index(i, i + j)] * data_t[index(i + k, i + 1 + j) ];
                }

                const double value = std::cbrt(dot_product[0]);
                data[index(i, i + k)] = value;
                data_t[index(i + k, i)] = value;

            });

        }
    }
};

#endif //SPM_FFMATRIX_H