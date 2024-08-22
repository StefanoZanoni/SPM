#ifndef SPM_MPIMATRIX_H
#define SPM_MPIMATRIX_H

#include <immintrin.h>
#include <mpi.h>
#include <cmath>
#include <mm_malloc.h>

/**
 * \brief A class to represent an upper triangular matrix (stored in a 1D array),
 * with the computation of the upper diagonals distributed across processes using MPI.
 */
class MPIMatrix final {

public:
    /**
     * \brief Constructor.
     * \param size The size of the matrix (the number of rows and columns).
     * \param rank The rank of the MPI process.
     * \param mpi_world_size The number of MPI processes.
     */
    MPIMatrix(const int size, const int rank, const int mpi_world_size) :
        size{size},
        rank{rank},
        mpi_world_size{mpi_world_size},
        rows_per_proc{size / mpi_world_size},
        remainder{size % mpi_world_size},
        start_row{rank * rows_per_proc + std::min(rank, remainder)},
        end_row{start_row + rows_per_proc + (rank < remainder ? 1 : 0)},
        data{end_row > start_row ? static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32)) : nullptr},
        data_t{end_row > start_row ? static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 32)) : nullptr},
        diagonal_buffer{end_row > start_row ? new double[end_row - start_row] : nullptr},
        combined_diagonal_buffer{end_row > start_row ? new double[size] : nullptr},
        recvcounts(new int[mpi_world_size]),
        displs(new int[mpi_world_size])
    {
        if (mpi_world_size != 1) {
            // Create a new communicator for processes with valid rows
            MPI_Comm_split(MPI_COMM_WORLD, end_row > start_row, rank, &comm);
        }

        if (end_row <= start_row) return;

        for (long i = 0; i < size; ++i) {
            data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
            data_t[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
        }

        for (int i = 0; i < mpi_world_size; ++i) {
            const int proc_rows = size / mpi_world_size + (i < remainder ? 1 : 0);
            recvcounts[i] = proc_rows;
        }
        for (int i = 0; i < mpi_world_size; ++i) {
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    ~MPIMatrix() {
        if (data) _mm_free(data);
        if (data_t) _mm_free(data_t);
        delete[] diagonal_buffer;
        delete[] combined_diagonal_buffer;
        delete[] recvcounts;
        delete[] displs;
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }

    void set_upper_diagonals() const {
        if (end_row <= start_row) return;
        alignas(32) double dot_product[4];

        // Iterate over the diagonals.
        for (int k = 1; k < size; ++k) {

            // Distribute across the rows.
            for (int i = start_row; i < end_row && i < size - k; ++i) {

                // Try to prefetch the next iteration first 4 double vectors (row and column) into L3 cache

                // first element of the row and column
                if (i + 1 + k and i + 2 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 2)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 1)], _MM_HINT_T2);
                }
                // second element of the row and column
                if (i + 1 + k and i + 3 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 3)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 2)], _MM_HINT_T2);
                }
                // third element of the row and column
                if (i + 1 + k and i + 4 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 4)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 3)], _MM_HINT_T2);
                }
                // fourth element of the row and column
                if (i + 1 + k and i + 5 < size - k) {
                    _mm_prefetch(&data[index(i + 1, i + 5)], _MM_HINT_T2);
                    _mm_prefetch(&data_t[index(i + 1 + k, i + 4)], _MM_HINT_T2);
                }

                // Use AVX2 to speed up the dot product calculation
                long j = 0;
                __m256d sum = _mm256_setzero_pd();
                for (; j <= k - 4; j += 4) {
                    const __m256d row = _mm256_loadu_pd(&data[index(i, i + j)]);
                    const __m256d column = _mm256_loadu_pd(&data_t[index(i + k, i + 1 + j)]);
                    sum = _mm256_fmadd_pd(row, column, sum);
                }

                // Sum the elements of the AVX register
                _mm256_store_pd(dot_product, sum);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                // Handle remaining elements
                for (; j < k; ++j) {
                    dot_product[0] += data[index(i, i + j)] * data_t[index(i + k, i + 1 + j) ];
                }

                const double value = std::cbrt(dot_product[0]);
                data[index(i, i + k)] = value;
                data_t[index(i + k, i)] = value;

                diagonal_buffer[i - start_row] = value;
            }

            if (mpi_world_size == 1) continue;

            // Non-blocking gather the diagonal elements
            MPI_Request gather_request;
            MPI_Igatherv(diagonal_buffer, end_row - start_row, MPI_DOUBLE,
                         combined_diagonal_buffer, recvcounts, displs, MPI_DOUBLE,
                         0, comm, &gather_request);
            // Wait for the gather to complete
            MPI_Wait(&gather_request, MPI_STATUS_IGNORE);

            // Non-blocking broadcast the combined buffer to all processes
            MPI_Request bcast_request;
            MPI_Ibcast(combined_diagonal_buffer, static_cast<int>(size) - k, MPI_DOUBLE, 0, comm, &bcast_request);
            // Wait for the broadcast to complete
            MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);

            // Update the matrix for all the processes.
            for (int i = 0; i < size - k; ++i) {
                data[index(i, i + k)] = combined_diagonal_buffer[i];
                data_t[index(i + k, i)] = combined_diagonal_buffer[i];
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
        std::cout << oss.str() << std::flush;
    }

private:

    const int size; ///< The size of the matrix (number of rows and columns).
    const int rank; ///< The rank of the MPI process.
    const int mpi_world_size; ///< The number of MPI processes.
    const int rows_per_proc; ///< The number of rows per MPI process.
    const int remainder; ///< The remainder when size is divided by the number of MPI processes.
    const int start_row; ///< The starting row for this MPI process.
    const int end_row; ///< The ending row for this MPI process.

    double* __restrict__ const data; ///< The data buffer for the matrix.
    double* __restrict__ const data_t; ///< The data buffer for the matrix transposed.
    double* __restrict__ const diagonal_buffer; ///< Buffer for diagonal elements.
    double* __restrict__ const combined_diagonal_buffer; ///< Buffer for combined diagonal elements.
    int* __restrict__ const recvcounts; ///< The number of elements to receive from each process.
    int* __restrict__ const displs; ///< The displacement of the receive buffer for each process.
    MPI_Comm comm{MPI_COMM_NULL};

    [[nodiscard]] long index(const long row, const long column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

};

#endif //SPM_MPIMATRIX_H
