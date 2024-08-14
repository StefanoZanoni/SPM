#ifndef SPM_MPIMATRIX_H
#define SPM_MPIMATRIX_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mm_malloc.h>

/**
 * \brief A class to represent an upper triangular matrix (stored in a 1D array),
 * with the computation of the upper diagonals distributed across processes using MPI.
 */
class MPIMatrix {

public:
    /**
     * \brief Constructor.
     * \param size The size of the matrix (the number of rows and columns).
     * \param rank The rank of the MPI process.
     * \param mpi_world_size The number of MPI processes.
     */
    explicit MPIMatrix(const int size, const int rank, const int mpi_world_size) :
            size{size},
            rank{rank},
            rows_per_proc{size / mpi_world_size},
            remainder{size % mpi_world_size},
            start_row{rank * rows_per_proc + std::min(rank, remainder)},
            end_row{start_row + rows_per_proc + (rank < remainder ? 1 : 0)},
            data{static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64))},
            diagonal_buffer{new double[end_row - start_row]},
            combined_diagonal_buffer{new double[size]},
            recvcounts{mpi_world_size},
            displs{mpi_world_size}
    {
        std::cout << "rank: " << rank << std::endl;
        std::cout << std::flush;
        std::cout << "rows per process: " << rows_per_proc << " remainder: " << remainder << std::endl;
        std::cout << std::flush;
        std::cout << " start: " << start_row << " end: " << end_row << std::endl;
        std::cout << std::flush;

        for (int i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }

        for (int i = 0; i < mpi_world_size; ++i) {
            int proc_rows = size / mpi_world_size + (i < remainder ? 1 : 0);
            recvcounts[i] = proc_rows;
        }
        for (int i = 0; i < mpi_world_size; ++i) {
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    /**
     * \brief Destructor.
     */
    ~MPIMatrix() {
        if (data) _mm_free(data);
        delete[] diagonal_buffer;
        delete[] combined_diagonal_buffer;
    }

    /**
     * \brief Set the upper diagonals of the matrix. The diagonal elements are the cube root of the dot product of the
     * corresponding row and column. The dot product is calculated using the previous upper diagonals. The diagonal
     * elements are then gathered to all MPI processes.
     */
    void set_upper_diagonals() {

        // Iterate over the diagonals.
        for (int k = 1; k < size; ++k) {

            // Reset diagonal_buffer for safety
            std::fill(diagonal_buffer, diagonal_buffer + (end_row - start_row), 0);

            // Distribute across the rows.
            for (int i = start_row; i <= end_row && i < size - k; ++i) {

                alignas(64)
                double dot_product{0};
                for (int j = 0; j < k; ++j) {
                    dot_product += data[index(i, i + j)] * data[index(i + 1 + j, i + k)];
                }
                data[index(i, i + k)] = std::cbrt(dot_product);
                diagonal_buffer[i] = data[index(i, i + k)];

            }
            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "Process " << rank << " reached MPI_Gatherv at k=" << k << std::endl;
            // Gather the diagonal elements
            MPI_Gatherv(diagonal_buffer, end_row - start_row, MPI_DOUBLE,
                        combined_diagonal_buffer, recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "Process " << rank << " reached MPI_Bcast at k=" << k << std::endl;
            // Broadcast the combined buffer to all processes
            MPI_Bcast(combined_diagonal_buffer, size - k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Update the matrix for all the processes.
            for (int i = 0; i < size - k; ++i) {
                std::cout << combined_diagonal_buffer[i] << std::endl;
                data[index(i, i + k)] = combined_diagonal_buffer[i];
            }

        }
    }

    /**
     * \brief Print the upper triangular matrix.
     */
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
    const int size; ///< The size of the matrix (the number of rows and columns).
    const int rank; ///< The rank of the MPI process.
    const int rows_per_proc; ///< The number of rows per MPI process.
    const int remainder; ///< The remainder when size is divided by the number of MPI processes.
    const int start_row; ///< The starting row for this MPI process.
    const int end_row; ///< The ending row for this MPI process.

    double* __restrict__ const data; ///< 1D array to store the matrix.
    double* __restrict__ const diagonal_buffer; ///< Buffer for diagonal elements.
    double* __restrict__ const combined_diagonal_buffer; ///< Buffer for combined diagonal elements.
    std::vector<int> recvcounts;
    std::vector<int> displs;

    /**
     * \brief Calculate the index in the data array for a given row and column.
     * \param row The row index.
     * \param column The column index.
     * \return The index in the data array.
     */
    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }
};

#endif //SPM_MPIMATRIX_H
