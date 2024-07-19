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

mean_execution_time_sequential = np.mean(sequential_data[:, 1])
std_execution_time_sequential = np.std(sequential_data[:, 1])

mean_execution_time_parallel_p = np.mean(parallel_p_data[:, 1])
std_execution_time_parallel_p = np.std(parallel_p_data[:, 1])

mean_execution_time_parallel_1 = np.mean(parallel_1_data[:, 1])
std_execution_time_parallel_1 = np.std(parallel_1_data[:, 1])

speedup = np.array([sequential_data[i, 1] / parallel_p_data[i, 1] for i in range(len(sequential_data))])
mean_speedup = np.mean(speedup)
std_speedup = np.std(speedup)

efficiency = np.array([speedup[i] / num_workers for i in range(len(speedup))])
mean_efficiency = np.mean(efficiency)
std_efficiency = np.std(efficiency)

scalability = np.array([parallel_1_data[i, 1] / parallel_p_data[i, 1] for i in range(len(parallel_p_data))])
mean_scalability = np.mean(scalability)
std_scalability = np.std(scalability)

with open('./statistics/statistics.json', 'w') as file:
    json.dump({
        'num_workers': num_workers,
        'mean_execution_time_sequential': mean_execution_time_sequential,
        'std_execution_time_sequential': std_execution_time_sequential,
        f'mean_execution_time_parallel_{num_workers}': mean_execution_time_parallel_p,
        f'std_execution_time_parallel_{num_workers}': std_execution_time_parallel_p,
        'mean_execution_time_parallel_1': mean_execution_time_parallel_1,
        'std_execution_time_parallel_1': std_execution_time_parallel_1,
        'mean_speedup': mean_speedup,
        'std_speedup': std_speedup,
        'mean_efficiency': mean_efficiency,
        'std_efficiency': std_efficiency,
        'mean_scalability': mean_scalability,
        'std_scalability': std_scalability
    }, file, indent=4)
