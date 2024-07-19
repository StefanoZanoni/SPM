#ifndef SPM_MATRIX_H
#define SPM_MATRIX_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <xmmintrin.h>
#include <cstdlib>
#include <mm_malloc.h>

class UTMatrix {

public:
    explicit UTMatrix(const size_t size) :
    size{size},
    data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32))}
    {
        for (size_t i = 0; i < size; ++i) {
            data[index(i, i)] = double (i + 1) / (double) size;
        }
    }

    ~UTMatrix() {
        _mm_free(data);
    }

    void set_upper_diagonals() {
        // Pin the current thread to the core 0 to improve cache performance
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

        size_t row_index;
        size_t column_index;
        __m256d vec_dot_product;
        __m256d row;
        __m256d column;
        alignas(32) double dot_product[4];

        for (size_t k = 1; k < size; ++k) {
            for (size_t i = 0; i < size - k; ++i) {

                // precompute indices
                row_index = index(i, i);
                column_index = index(i + 1, i + k);

                // Use AVX2 for SIMD operations
                vec_dot_product = _mm256_setzero_pd();
                size_t j = 0;
                for (; j + 3 < k; j += 4) {
                    row = _mm256_loadu_pd(&data[row_index + j]);
                    column = _mm256_loadu_pd(&data[column_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(row, column, vec_dot_product);
                }

                _mm256_store_pd(dot_product, vec_dot_product);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    dot_product[0] += data[row_index + j] * data[column_index + j];
                }

                data[row_index + k] = std::cbrt(dot_product[0]);
            }
        }

    }

    void print() const {
        std::ostringstream oss;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j >= i) {
                    oss << std::setw(9) << std::setprecision(6) << std::fixed << data[index(i, j)] << " ";
                } else {
                    oss << std::setw(10) << "0 ";
                }
            }
            oss << '\n';
        }
        oss << "\n";
        std::cout << oss.str();
    }

private:
    const size_t size;
    double* __restrict__ const data;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

};

#endif //SPM_MATRIX_H
