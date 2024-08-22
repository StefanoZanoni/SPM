import numpy as np
import json

sequential_results_file = './results/sequential.csv'
parallel_1_results_file = './results/parallel_1.csv'
distributed_1_results_file = './results/distributed_1.csv'

sequential = np.genfromtxt(sequential_results_file, delimiter=',', skip_header=1)
parallel_1 = np.genfromtxt(parallel_1_results_file, delimiter=',', skip_header=1)
distributed_1 = np.genfromtxt(distributed_1_results_file, delimiter=',', skip_header=1)

dimensions = sequential[:, 0]
dimensions = [int(dimension) for dimension in dimensions]
sequential_times = sequential[:, 1]
parallel_1_times = parallel_1[:, 1]
distributed_1_times = distributed_1[:, 1]

speedups = []
efficiencies = []
scalabilities = []
parallel_p_all_times = []
distributed_d_all_times = []

total_workers = [2, 4, 8, 16]
for workers in total_workers:
    parallel_p = np.genfromtxt(f'./results/parallel_{workers}.csv', delimiter=',', skip_header=1)
    distributed_p = np.genfromtxt(f'./results/distributed_{workers}.csv', delimiter=',', skip_header=1)

    parallel_p_times = parallel_p[:, 1]
    distributed_d_times = distributed_p[:, 1]
    parallel_p_all_times.append(parallel_p_times)
    distributed_d_all_times.append(distributed_d_times)

    parallel_speedup = sequential_times / parallel_p_times
    distributed_speedup = sequential_times / distributed_d_times
    speedups.append((parallel_speedup, distributed_speedup))

    parallel_efficiency = parallel_speedup / workers
    distributed_efficiency = distributed_speedup / workers
    efficiencies.append((parallel_efficiency, distributed_efficiency))

    parallel_scalability = parallel_1_times / parallel_p_times
    distributed_scalability = distributed_1_times / distributed_d_times
    scalabilities.append((parallel_scalability, distributed_scalability))

parallel_speedups = [speedup[0] for speedup in speedups]
distributed_speedups = [speedup[1] for speedup in speedups]

parallel_efficiencies = [efficiency[0] for efficiency in efficiencies]
distributed_efficiencies = [efficiency[1] for efficiency in efficiencies]

parallel_scalabilities = [scalability[0] for scalability in scalabilities]
distributed_scalabilities = [scalability[1] for scalability in scalabilities]

for i, dimension in zip(range(len(dimensions)), dimensions):
    statistics = {'sequential execution time (s)': float(sequential_times[i]),
                  'parallel 1 worker execution time (s)': float(parallel_1_times[i]),
                  'distributed 1 worker execution time (s)': float(distributed_1_times[i])}
    for j, workers in zip(range(len(total_workers)), total_workers):
        statistics[f'parallel {workers} workers execution time (s)'] = float(parallel_p_all_times[j][i])
        statistics[f'distributed {workers} workers execution time (s)'] = float(distributed_d_all_times[j][i])
        statistics[f'parallel {workers} workers speedup'] = float(parallel_speedups[j][i])
        statistics[f'distributed {workers} workers speedup'] = float(distributed_speedups[j][i])
        statistics[f'parallel {workers} workers efficiency'] = float(parallel_efficiencies[j][i])
        statistics[f'distributed {workers} workers efficiency'] = float(distributed_efficiencies[j][i])
        statistics[f'parallel {workers} workers scalability'] = float(parallel_scalabilities[j][i])
        statistics[f'distributed {workers} workers scalability'] = float(distributed_scalabilities[j][i])
    with open(f'./statistics/statistics_{dimension}.json', 'w') as file:
        json.dump(statistics, file, indent=4)
