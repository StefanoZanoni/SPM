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
            data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))} {
        for (long i = 0; i < size; ++i) {
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

                __m256d vec_dot_product = _mm256_setzero_pd();
                __m256d row;
                __m256d column;

                unsigned int j = 0;
                for (; j + 3 < k; j += 4) {
                    row = _mm256_loadu_pd(&data[index(i, i + j)]);

                    // The matrix is stored in row major order therefore the elements for the column are not contiguous
                    int indices[4] = {
                            static_cast<int>(index(i + j + 1, i + k)),
                            static_cast<int>(index(i + j + 2, i + k)),
                            static_cast<int>(index(i + j + 3, i + k)),
                            static_cast<int>(index(i + j + 4, i + k))
                    };
                    column = _mm256_i32gather_pd(&data[0], _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices)), 8);

                    vec_dot_product = _mm256_fmadd_pd(row, column, vec_dot_product);
                }

                alignas(32) double dot_product[4];
                _mm256_store_pd(dot_product, vec_dot_product);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                for (; j < k; ++j) {
                    dot_product[0] += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }

                data[index(i, i + k)] = std::cbrt(dot_product[0]);
            });
        }
    }

    void print() const {
        std::ostringstream oss;
        for (long i = 0; i < size; ++i) {
            for (long j = 0; j < size; ++j) {
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

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

};

#endif //SPM_FFMATRIX_H