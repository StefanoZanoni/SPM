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
    size(size),
    data(static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32)))
    {
        if (!data) {
            throw std::runtime_error("Memory allocation failed");
        }
        for (size_t i = 0; i < size; ++i) {
            data[index(i, i)] = double (i + 1) / size;
        }
    }

    ~UTMatrix() {
        _mm_free(data);
    }

    void setUpperDiagonals() {
        size_t base_index;
        size_t offset_index;
        __m256d vec_dot_product;
        __m256d vec1;
        __m256d vec2;
        alignas(32) double temp[4];
        double dot_product;

        for (size_t k = 1; k < size; ++k) {
            for (size_t i = 0; i < size - k; ++i) {

                // precompute indices
                base_index = index(i, i);
                offset_index = index(i + 1, i + k);

                // Use AVX2 for SIMD operations
                vec_dot_product = _mm256_setzero_pd();
                size_t j = 0;
                for (; j + 3 < k; j += 4) {
                    vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                _mm256_store_pd(temp, vec_dot_product);
                dot_product = temp[0] + temp[1] + temp[2] + temp[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    dot_product += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(dot_product);
            }
        }

    }

    void print() const {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j >= i) {
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[index(i, j)] << " ";
                } else {
                    std::cout << std::setw(10) << "0 ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n" << std::endl;
    }

private:
    const size_t size;
    alignas(32) double* const data;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

};

#endif //SPM_MATRIX_H
