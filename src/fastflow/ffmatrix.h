#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <cstdlib>
#include <mm_malloc.h>
#include <ff/parallel_for.hpp>
#include <thread>

class FFMatrix {

public:
    explicit FFMatrix(const long size) : size{size},
            data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32))} {
        for (size_t i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }
    }

    ~FFMatrix() {
        _mm_free(data);
    }

    void set_upper_diagonals(const long maxnw = 0) {
        ff::ParallelFor pf = (maxnw <= 0) ? ff::ParallelFor{true, true} : ff::ParallelFor{maxnw, true, true};

        for (long k = 1; k < size; ++k) {
            pf.parallel_for_static(0, size - k, 1, 0, [&](const long i) {
                // precompute indices
                const size_t base_index = index(i, i);
                const size_t offset_index = index(i + 1, i + k);

                // Use AVX2 for SIMD operations
                __m256d vec_dot_product = _mm256_setzero_pd();
                __m256d vec1;
                __m256d vec2;

                // Compute dot product
                unsigned int j = 0;
                for (; j + 3 < k; j += 4) {
                    vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                // Sum the elements of vec_dot_product
                alignas(32) double dot_product[4];
                _mm256_store_pd(dot_product, vec_dot_product);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    dot_product[0] += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(dot_product[0]);
            });
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

    const long size;
    double* __restrict__ const data;

    [[nodiscard]] inline size_t index(size_t row, size_t column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

};

#endif //SPM_FFMATRIX_H