#ifndef SPM_MPIMATRIX_H
#define SPM_MPIMATRIX_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mm_malloc.h>

class MPIMatrix {

public:
    explicit MPIMatrix(const int size, const int rank, const int mpi_world_size) :
            size{size},
            rows_per_proc{size / mpi_world_size},
            remainder{size % mpi_world_size},
            start_row{rank * rows_per_proc + std::min(rank, remainder)},
            end_row{start_row + rows_per_proc + (rank < remainder ? 1 : 0)},
            data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))},
            diagonal_buffer{new double[end_row - start_row]},
            combined_diagonal_buffer{new double[size]}
    {
        for (int i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }
    }

    ~MPIMatrix() {
        if (data) _mm_free(data);
        delete[] diagonal_buffer;
        delete[] combined_diagonal_buffer;
    }

    void set_upper_diagonals() {

        for (int k = 1; k < size; ++k) {

            for (int i = start_row; i < end_row && i < size - k; ++i) {
                alignas(64) double dot_product{0};
                for (int j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }
                data[index(i, i + k)] = std::cbrt(dot_product);
                diagonal_buffer[i] = data[index(i, i + k)];
            }
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgather(diagonal_buffer, end_row - start_row, MPI_DOUBLE,
                          combined_diagonal_buffer, end_row - start_row, MPI_DOUBLE, MPI_COMM_WORLD);

            for (int i = 0; i < size - k; ++i) {
                data[index(i, i + k)] = combined_diagonal_buffer[i];
            }

        }
    }

    void print() const {
        std::ostringstream oss;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
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
    const int size;
    const int rows_per_proc;
    const int remainder;
    const int start_row;
    const int end_row;

    double* __restrict__ const data;
    double* __restrict__ const diagonal_buffer;
    double* __restrict__ const combined_diagonal_buffer;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }
};

#endif //SPM_MPIMATRIX_H
