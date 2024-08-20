import numpy as np
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1, help='Number of workers used in parallel execution')
    return parser.parse_args()


args = parse_arguments()
num_workers = args.workers

sequential_data = np.genfromtxt('./results/sequential.csv', delimiter=',', skip_header=1)
parallel_p_data = np.genfromtxt('./results/parallel_p.csv', delimiter=',', skip_header=1)
parallel_1_data = np.genfromtxt('./results/parallel_1.csv', delimiter=',', skip_header=1)
distributed_d_data = np.genfromtxt('./results/distributed_d.csv', delimiter=',', skip_header=1)
distributed_1_data = np.genfromtxt('./results/distributed_1.csv', delimiter=',', skip_header=1)

mean_execution_time_sequential = np.mean(sequential_data[:, 1])
std_execution_time_sequential = np.std(sequential_data[:, 1])

mean_execution_time_parallel_p = np.mean(parallel_p_data[:, 1])
std_execution_time_parallel_p = np.std(parallel_p_data[:, 1])

mean_execution_time_parallel_1 = np.mean(parallel_1_data[:, 1])
std_execution_time_parallel_1 = np.std(parallel_1_data[:, 1])

mean_execution_time_distributed_d = np.mean(distributed_d_data[:, 1])
std_execution_time_distributed_d = np.std(distributed_d_data[:, 1])

parallel_speedup = np.array([sequential_data[i, 1] / parallel_p_data[i, 1] for i in range(len(sequential_data))])
mean_parallel_speedup = np.mean(parallel_speedup)
std_parallel_speedup = np.std(parallel_speedup)

parallel_efficiency = np.array([parallel_speedup[i] / num_workers for i in range(len(parallel_speedup))])
mean_parallel_efficiency = np.mean(parallel_efficiency)
std_parallel_efficiency = np.std(parallel_efficiency)

parallel_scalability = np.array([parallel_1_data[i, 1] / parallel_p_data[i, 1] for i in range(len(parallel_p_data))])
mean_parallel_scalability = np.mean(parallel_scalability)
std_parallel_scalability = np.std(parallel_scalability)

distributed_speedup = np.array([sequential_data[i, 1] / distributed_d_data[i, 1] for i in range(len(sequential_data))])
mean_distributed_speedup = np.mean(distributed_speedup)
std_distributed_speedup = np.std(distributed_speedup)

distributed_efficiency = np.array([distributed_speedup[i] / num_workers for i in range(len(distributed_speedup))])
mean_distributed_efficiency = np.mean(distributed_efficiency)
std_distributed_efficiency = np.std(distributed_efficiency)

distributed_scalability = np.array([distributed_1_data[i, 1] / distributed_d_data[i, 1]
                                    for i in range(len(distributed_d_data))])
mean_distributed_scalability = np.mean(distributed_scalability)
std_distributed_scalability = np.std(distributed_scalability)

with open('./statistics/statistics.json', 'w') as file:
    json.dump({
        'num_workers': num_workers,
        'mean_execution_time_sequential': mean_execution_time_sequential,
        'std_execution_time_sequential': std_execution_time_sequential,
        f'mean_execution_time_parallel_{num_workers}': mean_execution_time_parallel_p,
        f'std_execution_time_parallel_{num_workers}': std_execution_time_parallel_p,
        'mean_execution_time_parallel_1': mean_execution_time_parallel_1,
        'std_execution_time_parallel_1': std_execution_time_parallel_1,
        'mean_parallel_speedup': mean_parallel_speedup,
        'std_parallel_speedup': std_parallel_speedup,
        'mean_parallel_efficiency': mean_parallel_efficiency,
        'std_parallel_efficiency': std_parallel_efficiency,
        'mean_parallel_scalability': mean_parallel_scalability,
        'std_parallel_scalability': std_parallel_scalability,
        'mean_execution_time_distributed_d': mean_execution_time_distributed_d,
        'std_execution_time_distributed_d': std_execution_time_distributed_d,
        'mean_execution_time_distributed_1': np.mean(distributed_1_data[:, 1]),
        'std_execution_time_distributed_1': np.std(distributed_1_data[:, 1]),
        'mean_distributed_speedup': mean_distributed_speedup,
        'std_distributed_speedup': std_distributed_speedup,
        'mean_distributed_efficiency': mean_distributed_efficiency,
        'std_distributed_efficiency': std_distributed_efficiency,
        'mean_distributed_scalability': mean_distributed_scalability,
        'std_distributed_scalability': std_distributed_scalability
    }, file, indent=4)
