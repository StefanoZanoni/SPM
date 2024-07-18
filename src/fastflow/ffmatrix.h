#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <immintrin.h>
#include <ff/parallel_for.hpp>
#include <thread>
#include <numa.h>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cassert>

class FFMatrix {
public:
    explicit FFMatrix(const long size, const long maxnw = std::thread::hardware_concurrency())
            : size{size}, maxnw{maxnw} {

        const size_t num_numa_nodes = numa_num_configured_nodes();
        data_parts.resize(num_numa_nodes);
        rows_per_node.resize(num_numa_nodes + 1);

        const size_t rows_per_numa = std::max(size / num_numa_nodes, size_t(1));
        const size_t alignment = 32;

        // Calculate maximum part_size for uniform allocation
        const size_t max_rows_per_node = rows_per_numa + (size % num_numa_nodes > 0 ? 1 : 0);
        const size_t max_part_size = max_rows_per_node * (2 * size - max_rows_per_node + 1) / 2;
        size_t space = max_part_size * sizeof(double) + alignment - 1;

        size_t start_row = 0;
        for (size_t node = 0; node < num_numa_nodes; ++node) {
            rows_per_node[node] = start_row;
            const size_t num_rows = std::min(rows_per_numa, size - start_row);

            void* raw_ptr = numa_alloc_onnode(space, node);
            if (!raw_ptr) {
                throw std::runtime_error("Memory allocation failed on NUMA node " + std::to_string(node));
            }
            void* aligned_ptr = std::align(alignment, max_part_size * sizeof(double), raw_ptr, space);
            if (!aligned_ptr) {
                numa_free(raw_ptr, space);
                throw std::runtime_error("Memory alignment failed on NUMA node " + std::to_string(node));
            }
            data_parts[node] = static_cast<double*>(aligned_ptr);

            start_row += num_rows;
        }
        rows_per_node[num_numa_nodes] = size;

        // Initialize the main diagonal
        for (long i = 0; i < size; ++i) {
            size_t node = node_for_row(i);
            size_t diagonal_element = local_index(node, i, i);
            data_parts[node][diagonal_element] = double(i + 1) / double(size);
        }
    }

    ~FFMatrix() {
        const size_t num_numa_nodes = numa_num_configured_nodes();
        for (size_t node = 0; node < num_numa_nodes; ++node) {
            size_t start_row = rows_per_node[node];
            size_t num_rows = rows_per_node[node + 1] - start_row;
            size_t part_size = num_rows * (2 * size - num_rows + 1) / 2;
            numa_free(data_parts[node], part_size * sizeof(double) + 32 - 1);
        }
    }

    void set_upper_diagonals() {
        ff::ParallelFor pf{true, true};

        for (long k = 1; k < size; ++k) {
            pf.parallel_for(0, size - k, 1, [&](const long i) {
                const size_t row_node = node_for_row(i);
                const size_t column_node = node_for_row(i + 1);

                double* const data_row = data_parts[row_node];
                double* const data_column = data_parts[column_node];

                const size_t local_row_idx = local_index(row_node, i, i);
                const size_t local_column_idx = local_index(column_node, i + 1, i + k);

                __m256d vec_dot_product = _mm256_setzero_pd();
                __m256d row;
                __m256d column;

                long j = 0;
                for (; j + 3 < k; j += 4) {
                    row = _mm256_loadu_pd(&data_row[local_row_idx + j]);
                    column = _mm256_loadu_pd(&data_column[local_column_idx + j]);
                    vec_dot_product = _mm256_fmadd_pd(row, column, vec_dot_product);
                }

                alignas(32) double dot_product[4];
                _mm256_store_pd(dot_product, vec_dot_product);
                dot_product[0] += dot_product[1] + dot_product[2] + dot_product[3];

                for (; j < k; ++j) {
                    dot_product[0] += data_row[local_row_idx + j] * data_column[local_column_idx + j];
                }

                const size_t result_node = node_for_row(i);
                double* const data_result = data_parts[result_node];
                const size_t local_result_idx = local_index(result_node, i, i + k);
                data_result[local_result_idx] = std::cbrt(dot_product[0]);
            });
        }
    }

    void print() const {
        for (long i = 0; i < size; ++i) {
            for (long j = 0; j < size; ++j) {
                if (j >= i) {
                    size_t node = node_for_row(i);
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data_parts[node][local_index(node, i, j)] << " ";
                } else {
                    std::cout << std::setw(10) << "0 ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n" << std::endl;
    }

private:
    const long size;
    const long maxnw;
    std::vector<double*> data_parts;
    std::vector<size_t> rows_per_node;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

    [[nodiscard]] inline size_t node_for_row(const size_t row) const {
        return std::upper_bound(rows_per_node.begin(), rows_per_node.end(), row) - rows_per_node.begin() - 1;
    }

    [[nodiscard]] inline size_t local_index(const size_t node, const size_t row, const size_t column) const {
        size_t start_row = rows_per_node[node];
        return index(row - start_row, column);
    }
};

#endif //SPM_FFMATRIX_H
