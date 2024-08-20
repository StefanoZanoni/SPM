#ifndef SPM_MATRIX_H
#define SPM_MATRIX_H

#include <iostream>
#include <iomanip>
#include <mm_malloc.h>

class Matrix {

public:
    explicit Matrix(const long size) :
    size{size},
    data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))}
    {
        for (long i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }
    }

    virtual ~Matrix() {
        _mm_free(data);
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
        std::cout << std::flush;
    }

protected:
    const long size;
    double* __restrict__ const data;

    [[nodiscard]] long index(const long row, const long column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

};

#endif //SPM_MATRIX_H