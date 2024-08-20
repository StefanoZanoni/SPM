#ifndef SPM_SEQMATRIX_H
#define SPM_SEQMATRIX_H

#include <iostream>
#include <cmath>

#include "../utils/matrix.h"

class SeqMatrix final : public Matrix {

public:
    explicit SeqMatrix(const long size) : Matrix(size) {}

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
};

#endif //SPM_SEQMATRIX_H