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

inline void bind_thread_to_cpu(const size_t cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

class FFMatrix {
public:
    explicit FFMatrix(const long size, const long maxnw = std::thread::hardware_concurrency())
            : size{size}, maxnw{maxnw}, total_elements{size * (size + 1) / 2}, num_numa_nodes{numa_num_configured_nodes()} {

        data_parts.resize(num_numa_nodes);
        element_start.resize(num_numa_nodes + 1);

        const size_t elements_per_node = std::max(total_elements / num_numa_nodes, size_t(1));
        const size_t alignment = 32;

        // Allocate memory on each NUMA node and align it to 32 bytes
        size_t start_element = 0;
        for (int node = 0; node < num_numa_nodes; ++node) {
            element_start[node] = start_element;
            const size_t part_size = std::min(elements_per_node, total_elements - start_element);
            size_t space = part_size * sizeof(double) + alignment - 1;

            void *raw_ptr = numa_alloc_onnode(space, node);
            if (!raw_ptr) {
                throw std::runtime_error("Memory allocation failed on NUMA node " + std::to_string(node));
            }
            void *aligned_ptr = std::align(alignment, part_size * sizeof(double), raw_ptr, space);
            if (!aligned_ptr) {
                numa_free(raw_ptr, part_size * sizeof(double) + alignment - 1);
                throw std::runtime_error("Memory alignment failed on NUMA node " + std::to_string(node));
            }
            data_parts[node] = static_cast<double*>(aligned_ptr);

            start_element += part_size;
        }
        element_start[num_numa_nodes] = total_elements;

        // Initialize the diagonal elements
        for (size_t i = 0; i < size; ++i) {
            size_t idx = index(i, i);
            size_t node = node_for_index(i, i);
            data_parts[node][local_index(i, i)] = double(i + 1) / double(size);
        }
    }

    ~FFMatrix() {
        for (size_t node = 0; node < num_numa_nodes; ++node) {
            const size_t part_size = element_start[node + 1] - element_start[node];
            numa_free(data_parts[node], part_size * sizeof(double));
        }
    }

    void set_upper_diagonals() {
        ff::ParallelFor pf{maxnw};

        for (long k = 1; k < size; ++k) {
            pf.parallel_for(0, size - k, 1, [&](const long i) {
                const size_t cpu_id = i % maxnw;
                bind_thread_to_cpu(cpu_id);

                const size_t node = node_for_index(i, i);
                double* const data = data_parts[node];

                const size_t base_index = local_index(i, i);
                const size_t offset_index = local_index(i + 1, i + k);

                __m256d vec_dot_product = _mm256_setzero_pd();
                __m256d vec1;
                __m256d vec2;

                size_t j = 0;
                for (; j + 3 < k; j += 4) {
                    vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                alignas(32) double temp[4];
                _mm256_store_pd(temp, vec_dot_product);
                temp[0] += temp[1] + temp[2] + temp[3];

                for (; j < k; ++j) {
                    temp[0] += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(temp[0]);
            });
        }
    }

    void print() const {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j >= i) {
                    const size_t node = node_for_index(i, j);
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data_parts[node][local_index(i, j)] << " ";
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
    const long total_elements;
    const size_t num_numa_nodes;
    std::vector<double*> data_parts;
    std::vector<size_t> element_start;

    [[nodiscard]] inline size_t index(const size_t row, const size_t column) const {
        return (row * (2 * size - row - 1)) / 2 + column;
    }

    inline void decompose_index(const size_t idx, size_t& row, size_t& column) const {
        row = (size_t)((std::sqrt(8 * idx + 1) - 1) / 2);
        column = idx - (row * (2 * size - row - 1)) / 2;
    }

    [[nodiscard]] inline size_t node_for_index(const size_t row, const size_t column) const {
        size_t global_index = index(row, column);
        return std::upper_bound(element_start.begin(), element_start.end(), global_index) - element_start.begin() - 1;
    }

    [[nodiscard]] inline size_t local_index(const size_t row, const size_t column) const {
        size_t global_index = index(row, column);
        size_t node = node_for_index(row, column);
        return global_index - element_start[node];
    }
};

#endif //SPM_FFMATRIX_H
