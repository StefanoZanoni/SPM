#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <immintrin.h>
#include <cmath>
#include <ff/parallel_for.hpp>

#include "../utils/matrix.h"

/**
 * \brief A class to represent an upper triangular matrix with parallel computation (using FastFlow) of the upper diagonals.
 */
class FFMatrix final : public Matrix {

public:

    /**
     * \brief Constructor to initialize the FastFlow matrix with a given size.
     * \param size The size of the matrix (number of rows and columns).
     */
    explicit FFMatrix(const long size) : Matrix(size) {}

    /**
     * \brief Set the upper diagonals of the matrix in parallel.
     * Each element of the upper diagonals is the cubic root of the dot product of the corresponding row and column.
     * \param maxnw The maximum number of workers (default is 0, which means auto-detect).
     */
    void set_upper_diagonals(const long maxnw = 0) const {
        ff::ParallelFor pf = (maxnw <= 0) ? ff::ParallelFor{true, true} : ff::ParallelFor{maxnw, true, true};

        // Iterate over upper diagonals
        for (long k = 1; k < size; ++k) {

            // Iterate over rows in parallel.
            pf.parallel_for_static(0, size - k, 1, 0, [&](const long i) {

                // Try to prefetch the next iteration first two 4 double vectors (row and column) into L3 cache

                // first element of the row and column
                if (i + 1 + k and i + 2 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 2)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 1)], _MM_HINT_T2);
                }
                // second element of the row and column
                if (i + 1 + k and i + 3 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 3)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 2)], _MM_HINT_T2);
                }
                // third element of the row and column
                if (i + 1 + k and i + 4 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 4)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 3)], _MM_HINT_T2);
                }
                // fourth element of the row and column
                if (i + 1 + k and i + 5 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 5)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 4)], _MM_HINT_T2);
                }

                // Use AVX to speed up the dot product calculation
                long j = 0;
                __m256d sum = _mm256_setzero_pd();
                for (; j <= k - 4; j += 4) {
                    const __m256d row = _mm256_loadu_pd(&data[index(i, i + j)]);
                    const __m256d column = _mm256_loadu_pd(&data_t[index(i + k, i + 1 + j)]);
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(row, column));
                }

                // Sum the elements of the AVX register
                alignas(32) double dot_product[4];
                _mm256_store_pd(dot_product, sum);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                // Handle remaining elements
                for (; j < k; ++j) {
                    dot_product[0] += data[index(i, i + j)] * data_t[index(i + k, i + 1 + j) ];
                }

                // Store the result in the current diagonal
                const double value = std::cbrt(dot_product[0]);
                data[index(i, i + k)] = value;
                data_t[index(i + k, i)] = value;

            });

        }
    }
};

#endif //SPM_FFMATRIX_H