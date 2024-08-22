import matplotlib.pyplot as plt
import numpy as np

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

# -------------------------- aggregated plots -------------------------- #

# execution times
plt.figure(figsize=(16, 10))
plt.xlabel('Dimensions')
plt.ylabel('Execution time (s)')
plt.plot(dimensions, sequential_times, label='Sequential', marker='o')
plt.plot(dimensions, parallel_1_times, label='Parallel 1 worker', marker='o')
plt.plot(dimensions, distributed_1_times, label='Distributed 1 worker', marker='o')
for i, workers in zip(range(len(total_workers)), total_workers):
    plt.plot(dimensions, parallel_p_all_times[i], label=f'Parallel {workers} workers', marker='o')
    plt.plot(dimensions, distributed_d_all_times[i], label=f'Distributed {workers} workers', marker='o')
plt.title('Execution times')
plt.legend()
plt.grid(True)
plt.savefig('./plots/execution_times.png')
plt.close()

# speedups
parallel_speedups = [speedup[0] for speedup in speedups]
distributed_speedups = [speedup[1] for speedup in speedups]

plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Speedup')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [parallel_speedup[i] for parallel_speedup in parallel_speedups],
             label=f'dimension {dimension}', marker='o')
plt.title('Parallel speedups')
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_speedups.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Speedup')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [distributed_speedup[i] for distributed_speedup in distributed_speedups],
             label=f'dimension {dimension}', marker='o')
plt.title('Distributed speedups')
plt.legend()
plt.grid(True)
plt.savefig('./plots/distributed_speedups.png')
plt.close()

# efficiencies
parallel_efficiencies = [efficiency[0] for efficiency in efficiencies]
distributed_efficiencies = [efficiency[1] for efficiency in efficiencies]

plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Efficiency')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [parallel_efficiency[i] for parallel_efficiency in parallel_efficiencies],
             label=f'dimension {dimension}', marker='o')
plt.title('Parallel efficiencies')
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_efficiencies.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Efficiency')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [distributed_efficiency[i] for distributed_efficiency in distributed_efficiencies],
             label=f'dimension {dimension}', marker='o')
plt.title('Distributed efficiencies')
plt.legend()
plt.grid(True)
plt.savefig('./plots/distributed_efficiencies.png')
plt.close()

# scalabilities
parallel_scalabilities = [scalability[0] for scalability in scalabilities]
distributed_scalabilities = [scalability[1] for scalability in scalabilities]

plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Scalability')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [parallel_scalability[i] for parallel_scalability in parallel_scalabilities],
             label=f'dimension {dimension}', marker='o')

plt.title(f'Parallel scalabilities')
plt.legend()
plt.grid(True)
plt.savefig(f'./plots/parallel_scalabilities.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.xlabel('Workers')
plt.ylabel('Scalability')
for i, dimension in zip(range(len(dimensions)), dimensions):
    plt.plot(total_workers, [distributed_scalability[i] for distributed_scalability in distributed_scalabilities],
             label=f'dimension {dimension}', marker='o')
plt.title(f'Distributed scalabilities')
plt.legend()
plt.grid(True)
plt.savefig(f'./plots/distributed_scalabilities.png')
plt.close()
