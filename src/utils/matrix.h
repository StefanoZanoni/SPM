#ifndef SPM_MATRIX_H
#define SPM_MATRIX_H

#include <iostream>
#include <iomanip>
#include <mm_malloc.h>

/**
 * \brief A class to represent an upper triangular matrix stored in a 1D array.
 */
class Matrix {

public:

    /**
     * \brief Constructor to initialize the matrix with a given size.
     * \param size The size of the matrix (number of rows and columns).
     */
    explicit Matrix(const long size) :
    size{size},
    // Allocate memory for the matrix and its transpose using aligned memory allocation (32 bytes) for AVX2 instructions
    data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32))},
    data_t{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32))}
    {
        // Initialize the matrix with the values on the main diagonal (1/size, 2/size, 3/size, ..., size/size)
        for (long i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
            data_t[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }

    }

    /**
     * \brief Destructor to free allocated memory.
     */
    virtual ~Matrix() {
        _mm_free(data);
        _mm_free(data_t);
    }

    /**
     * \brief Print the matrix to the standard output.
     */
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
        std::cout << oss.str() << std::flush;
    }

protected:
    const long size; ///< The size of the matrix (number of rows and columns).
    double* __restrict__ const data; ///< The data buffer for the matrix.
    double* __restrict__ const data_t;  ///< The data buffer for the transposed matrix.

    /**
     * \brief Calculate the index in the 1D array for a given row and column.
     * \param row The row index.
     * \param column The column index.
     * \return The index in the 1D array.
     */
    [[nodiscard]] long index(const long row, const long column) const {
        return row * (2 * size - row + 1) / 2 + column - row;
    }

};

#endif //SPM_MATRIX_H