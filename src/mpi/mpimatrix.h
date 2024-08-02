#ifndef SPM_MPIMATRIX_H
#define SPM_MPIMATRIX_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mm_malloc.h>

class MPIMatrix {

public:
    explicit MPIMatrix(const int rank, const int mpi_world_size, const int size) :
            rank{rank},
            size{size},
            rows_per_proc{size / mpi_world_size},
            remainder{size % mpi_world_size},
            recvcount{(rank < remainder) ? rows_per_proc + 1 : rows_per_proc},
            data{nullptr},
            recvbuf{nullptr},
            rowcounts{new int[mpi_world_size]},
            displs{new int[mpi_world_size]}
    {
        if (rank == 0) {
            data = static_cast<double*>(_mm_malloc(size * (size + 1) / 2 * sizeof(double), 64));
            for (size_t i = 0; i < size; ++i) {
                data[index(i, i)] = static_cast<double>(i + 1) / static_cast<double>(size);
            }
            for (int i = 0; i < mpi_world_size; ++i) {
                rowcounts[i] = (i < remainder) ? (rows_per_proc + 1) * size : rows_per_proc * size;
            }
            displs[0] = 0;
            for (int i = 1; i < mpi_world_size; ++i) {
                displs[i] = displs[i - 1] + rowcounts[i - 1];
            }
        }
    }

    ~MPIMatrix() {
        if (rank == 0) {
            _mm_free(data);
            delete[] rowcounts;
            delete[] displs;
        }
        _mm_free(recvbuf);
    }

    void set_upper_diagonals() {
        // Scatter the data to all processes
        scatter();

        // Compute upper diagonals for the portion of data each process received
        for (int k = 1; k < size; ++k) {
            for (int i = 0; i < recvcount - k; ++i) {
                alignas(64) double dot_product{0};
                for (int j = 0; j < k; ++j) {
                    dot_product += recvbuf[index(i, i + j)] * recvbuf[index(i + 1 + j, i + k)];
                }
                recvbuf[index(i, i + k)] = std::cbrt(dot_product);
            }
        }

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);

        // Gather the computed data back to the root process
        gather();
    }

    void print() const {
        if (rank != 0)
            return;

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
    const int rank;
    const int size;
    const int rows_per_proc;
    const int remainder;
    const int recvcount;

    double* __restrict__ data;
    double* __restrict__ recvbuf;
    int* __restrict__ rowcounts;
    int* __restrict__ displs;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row + 1)) / 2 + column - row;
    }

    void scatter() {
        recvbuf = static_cast<double*>(_mm_malloc(recvcount * size * sizeof(double), 64));
        MPI_Scatterv(data, rowcounts, displs, MPI_DOUBLE, recvbuf, recvcount * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void gather() {
        MPI_Gatherv(recvbuf, recvcount * size, MPI_DOUBLE, data, rowcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

};

#endif //SPM_MPIMATRIX_H
