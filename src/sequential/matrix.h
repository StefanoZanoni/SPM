#ifndef SPM_MATRIX_H
#define SPM_MATRIX_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mm_malloc.h>

class UTMatrix {

public:
    explicit UTMatrix(const size_t size) :
    size{size},
    data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))}
    {
        for (size_t i = 0; i < size; ++i) {
            data[index(i, i)] = double (i + 1) / (double) size;
        }
    }

    ~UTMatrix() {
        _mm_free(data);
    }

    void set_upper_diagonals() {

        for (size_t k = 1; k < size; ++k) {
            for (size_t i = 0; i < size - k; ++i) {
                alignas(64) double dot_product = 0;
                for (size_t j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }
                data[index(i, i + k)] = dot_product;
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
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

};

#endif //SPM_MATRIX_H
