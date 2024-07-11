#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <cstdlib>
#include <mm_malloc.h>
#include <ff/parallel_for.hpp>

class FFMatrix {

public:
    explicit FFMatrix(unsigned int size) : size(size),
    data(static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32))) {
        if (!data) {
            throw std::runtime_error("Memory allocation failed");
        }
        for (unsigned int i = 0; i < size; ++i) {
            data[index(i, i)] = double (i + 1) / size;
        }
    }

    ~FFMatrix() {
        _mm_free(data);
    }

    void setUpperDiagonals() {
        ff::ParallelFor pf;
        for (unsigned int k = 1; k < size; ++k) {
            pf.parallel_for(0, size - k, 1, 0, [&](const long i) {
                double dot_product;

                // precompute indices
                unsigned int base_index = index(i, i);
                unsigned int offset_index = index(i + 1, i + k);

                // Use AVX2 for SIMD operations
                __m256d vec_dot_product = _mm256_setzero_pd();
                // Compute dot product
                unsigned int j = 0;
                for (; j + 3 < k; j += 4) {
                    __m256d vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    __m256d vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                // Sum the elements of vec_dot_product
                alignas(32) double temp[4];
                _mm256_store_pd(temp, vec_dot_product);
                dot_product = temp[0] + temp[1] + temp[2] + temp[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    dot_product += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(dot_product);
            });
        }
    }

    void print() const {
        for (unsigned int i = 0; i < size; ++i) {
            for (unsigned int j = 0; j < size; ++j) {
                if (j >= i) {
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[index(i, j)]<< " ";
                } else {
                    std::cout << std::setw(10) << "0 ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n" << std::endl;
    }

private:

    unsigned int size;
    alignas(32) double* data;

    [[nodiscard]] inline unsigned int index(unsigned int row, unsigned int column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

};

#endif //SPM_FFMATRIX_H
