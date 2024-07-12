#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <immintrin.h>
#include <ff/parallel_for.hpp>
#include <thread>
#include <numa.h>
#include <vector>

inline void bind_thread_to_cpu(const size_t cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

class FFMatrix {

public:
    explicit FFMatrix(const long size, const long maxnw = std::thread::hardware_concurrency()) : size{size}, maxnw{maxnw} {

        const size_t num_numa_nodes = numa_num_configured_nodes();
        data_parts.resize(num_numa_nodes);
        const size_t part_size = (size * (size + 1) / 2) / num_numa_nodes;
        const size_t alignment = 32;
        size_t space = part_size * sizeof(double) + alignment - 1;

        for (int node = 0; node < num_numa_nodes; ++node) {
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
        }

        for (size_t node = 0; node < num_numa_nodes; ++node) {
            for (size_t i = 0; i < size; ++i) {
                data_parts[node][local_index(i, i)] = double(i + 1) / (double) size;
            }
        }
    }

    ~FFMatrix() {
        for (auto& part : data_parts) {
            numa_free(part, (size * (size + 1) / 2) / data_parts.size() * sizeof(double));
        }
    }

    void set_upper_diagonals() {
        ff::ParallelFor pf{maxnw};

        for (long k = 1; k < size; ++k) {
            pf.parallel_for(0, size - k, 1, [&](const long i) {

                // Bind thread to specific CPU with round-robin strategy
                const size_t cpu_id = i % maxnw;
                bind_thread_to_cpu(cpu_id);

                // Determine which NUMA node this thread should use
                const size_t node = cpu_id % data_parts.size();
                double* const data = data_parts[node];

                // precompute indices
                const size_t base_index = local_index(i, i);
                const size_t offset_index = local_index(i + 1, i + k);

                // Use AVX2 for SIMD operations
                __m256d vec_dot_product = _mm256_setzero_pd();
                __m256d vec1;
                __m256d vec2;

                size_t j = 0;
                for (; j + 3 < k; j += 4) {
                    vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                // Sum the elements of vec_dot_product
                alignas(32) double temp[4];
                _mm256_store_pd(temp, vec_dot_product);
                temp[0] += temp[1] + temp[2] + temp[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    temp[0] += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(temp[0]);
            });
        }
    }

    void print() {
        merge_results();

        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j >= i) {
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data_parts[0][local_index(i, j)]<< " ";
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

    [[nodiscard]] inline size_t local_index(const size_t row, const size_t column) const {
        const size_t global_index = (row * (2 * size - row - 1)) / 2 + column;
        const size_t part_size = (size * (size + 1) / 2) / data_parts.size();
        return global_index % part_size;
    }

    void merge_results() {
        const size_t total_size = size * (size + 1) / 2;
        // Allocate on NUMA node 0
        auto merged_data = static_cast<double* const>(numa_alloc_onnode(total_size * sizeof(double), 0));

        const size_t part_size = total_size / data_parts.size();
        for (size_t node = 0; node < data_parts.size(); ++node) {
            std::memcpy(merged_data + node * part_size, data_parts[node], part_size * sizeof(double));
            numa_free(data_parts[node], part_size * sizeof(double));
        }

        data_parts.clear();
        data_parts.push_back(merged_data);
    }
};

#endif //SPM_FFMATRIX_H
