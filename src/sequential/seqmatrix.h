#ifndef SPM_SEQMATRIX_H
#define SPM_SEQMATRIX_H

#include <immintrin.h>
#include <cmath>
#include "../utils/matrix.h"

/**
 * \brief A class to represent an upper triangular matrix with sequential upper diagonals computation.
 */
class SeqMatrix final : public Matrix {

public:

    /**
     * \brief Constructor to initialize the sequential matrix with a given size.
     * \param size The size of the matrix (number of rows and columns).
     */
    explicit SeqMatrix(const long size) : Matrix(size) {}

    /**
     * \brief Set the upper diagonals of the matrix.
     * Each element of the upper diagonals is the cubic root of the dot product of the corresponding row and column.
     */
    void set_upper_diagonals() const {
        alignas(32) double dot_product[4];

        // Iterate over upper diagonals
        for (long k = 1; k < size; ++k) {

            // Iterate over rows
            for (long i = 0; i < size - k; ++i) {

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
            }
        }
    }
};

#endif //SPM_SEQMATRIX_H