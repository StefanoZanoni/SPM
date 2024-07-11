#ifndef SPM_FFMATRIX_H
#define SPM_FFMATRIX_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <cstdlib>
#include <mm_malloc.h>
#include <ff/parallel_for.hpp>
#include <thread>
#include <numa.h>
#include <pthread.h>
#include <vector>
#include <cstring>

class FFMatrix {

public:
    explicit FFMatrix(const unsigned int size, const unsigned int maxnw = std::thread::hardware_concurrency()) :
            size{size}, maxnw{maxnw} {
        int num_numa_nodes = numa_num_configured_nodes();
        data_parts.resize(num_numa_nodes);
        unsigned int part_size = (size * (size + 1) / 2) / num_numa_nodes;
        for (int node = 0; node < num_numa_nodes; ++node) {
            data_parts[node] = static_cast<double*>(numa_alloc_onnode(part_size * sizeof(double), node));
            if (!data_parts[node]) {
                throw std::runtime_error("Memory allocation failed on NUMA node " + std::to_string(node));
            }
        }
        for (unsigned int node = 0; node < num_numa_nodes; ++node) {
            for (unsigned int i = 0; i < size; ++i) {
                data_parts[node][local_index(i, i, node)] = double(i + 1) / size;
            }
        }
    }

    ~FFMatrix() {
        for (auto& part : data_parts) {
            numa_free(part, (size * (size + 1) / 2) / data_parts.size() * sizeof(double));
        }
    }

    void setUpperDiagonals() {
        ff::ParallelFor pf{maxnw};

        for (unsigned int k = 1; k < size; ++k) {
            pf.parallel_for(0, size - k, 1, [&](const long i) {
                // Bind thread to specific CPU
                int cpu_id = i % maxnw; // Simple round-robin assignment
                bindThreadToCPU(cpu_id);

                double dot_product;

                // Determine which NUMA node this thread should use
                int node = cpu_id % data_parts.size();
                double* data = data_parts[node];

                // precompute indices
                unsigned int base_index = local_index(i, i, node);
                unsigned int offset_index = local_index(i + 1, i + k, node);

                // Use AVX2 for SIMD operations
                __m256d vec_dot_product = _mm256_setzero_pd();
                // Compute dot product
                unsigned int j = 0;
                for (; j + 3 < k; j += 4) {
                    if (j + 4 < k) {
                        _mm_prefetch(reinterpret_cast<const char *>(&data[base_index + j + 4]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(&data[offset_index + j + 4]), _MM_HINT_T0);
                    }

                    __m256d vec1 = _mm256_loadu_pd(&data[base_index + j]);
                    __m256d vec2 = _mm256_loadu_pd(&data[offset_index + j]);
                    vec_dot_product = _mm256_fmadd_pd(vec1, vec2, vec_dot_product);
                }

                // Sum the elements of vec_dot_product
                alignas(32) double temp[4];
                _mm256_store_pd(temp, vec_dot_product);
                dot_product = temp[0] + temp[1] + temp[2] + temp[3];

                // Handle the remaining elements
                for (; j < k; ++j) {
                    dot_product += data[base_index + j] * data[offset_index + j];
                }

                data[base_index + k] = std::cbrt(dot_product);
            });
        }

        mergeResults();
    }

    void print() const {
        for (unsigned int i = 0; i < size; ++i) {
            for (unsigned int j = 0; j < size; ++j) {
                if (j >= i) {
                    std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data_parts[0][local_index(i, j, 0)]<< " ";
                } else {
                    std::cout << std::setw(10) << "0 ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n" << std::endl;
    }

private:

    const unsigned int size;
    const unsigned int maxnw;
    std::vector<double*> data_parts;

    [[nodiscard]] inline unsigned int local_index(unsigned int row, unsigned int column, unsigned int node) const {
        unsigned int global_index = (row * (2 * size - row - 1)) / 2 + column;
        unsigned int part_size = (size * (size + 1) / 2) / data_parts.size();
        return global_index % part_size;
    }

    void bindThreadToCPU(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    void mergeResults() {
        unsigned int total_size = size * (size + 1) / 2;
        // Allocate on NUMA node 0
        double* merged_data = static_cast<double*>(numa_alloc_onnode(total_size * sizeof(double), 0));

        unsigned int part_size = total_size / data_parts.size();
        for (unsigned int node = 0; node < data_parts.size(); ++node) {
            std::memcpy(merged_data + node * part_size, data_parts[node], part_size * sizeof(double));
            numa_free(data_parts[node], part_size * sizeof(double));
        }

        data_parts.clear();
        data_parts.push_back(merged_data);
    }
};

#endif //SPM_FFMATRIX_H
