#ifndef SPM_MATRIX_H
#define SPM_MATRIX_H

#include <iostream>
#include <cmath>
#include <mm_malloc.h>

class SeqMatrix {

public:
    explicit SeqMatrix(const long size) :
    size{size},
    data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))}
    {
        for (long i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }
    }

    ~SeqMatrix() {
        _mm_free(data);
    }

    void set_upper_diagonals() const {

        // Iterate over upper diagonals
        for (long k = 1; k < size; ++k) {
            // Iterate over rows
            for (long i = 0; i < size - k; ++i) {

                alignas(64) double dot_product{0};
                for (long j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }
                data[index(i, i + k)] = std::cbrt(dot_product);
            }
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

    [[nodiscard]] long index(const long row, const long column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

};

#endif //SPM_MATRIX_H
