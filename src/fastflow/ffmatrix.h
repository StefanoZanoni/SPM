#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ff/parallel_for.hpp>
#include <thread>
#include <mm_malloc.h>

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
                alignas(64) double dot_product = 0;
                for (long j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }

                data[index(i, i + k)] = std::cbrt(dot_product);
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