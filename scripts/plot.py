import os
import csv
import psutil
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1, help='Number of workers used in parallel execution')
    return parser.parse_args()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='full')[:len(data)]


def open_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        data = [row for row in csv_reader]
    return headers, data


def plot_data_and_trend(x_data, y_data, y_data_ma, x_label, y_label, title, file_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label='Data')
    plt.plot(x_data, y_data_ma, label='Trend', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()


# Get the number of workers
args = parse_arguments()
num_workers = args.workers
if num_workers <= 0:
    num_workers = psutil.cpu_count(logical=False)

cwd = os.getcwd()
file_path_sequential = os.path.join(cwd, './results/sequential.csv')
file_path_parallel_p = os.path.join(cwd, f'./results/parallel_{num_workers}.csv')
file_path_parallel_1 = os.path.join(cwd, './results/parallel_1.csv')
file_path_distributed_d = os.path.join(cwd, f'./results/distributed_{num_workers}.csv')
file_path_distributed_1 = os.path.join(cwd, './results/distributed_1.csv')

# Read sequential data
headers, data = open_csv(file_path_sequential)
x_data = [float(row[0]) for row in data]
y_data_sequential = [float(row[1]) for row in data]

# Read parallel data with p workers
_, data = open_csv(file_path_parallel_p)
y_data_parallel_p = [float(row[1]) for row in data]

# Read parallel data with 1 worker
_, data = open_csv(file_path_parallel_1)
y_data_parallel_1 = [float(row[1]) for row in data]

# Read distributed data with d processes
_, data = open_csv(file_path_distributed_d)
y_data_distributed_d = [float(row[1]) for row in data]

# Read distributed data with 1 process
_, data = open_csv(file_path_distributed_1)
y_data_distributed_1 = [float(row[1]) for row in data]

# Calculate parallel_speedup
parallel_speedup = [y_data_sequential[i] / y_data_parallel_p[i] for i in range(len(y_data_sequential))]

# Calculate parallel_scalability
parallel_scalability = [y_data_parallel_1[i] / y_data_parallel_p[i] for i in range(len(y_data_parallel_p))]

# Calculate parallel_efficiency
parallel_efficiency = [parallel_speedup[i] / num_workers for i in range(len(parallel_speedup))]

# Calculate distributed_speedup
distributed_speedup = [y_data_sequential[i] / y_data_distributed_d[i] for i in range(len(y_data_sequential))]

# Calculate distributed_scalability
distributed_scalability = [y_data_distributed_1[i] / y_data_distributed_d[i] for i in range(len(y_data_distributed_d))]

# Calculate distributed_efficiency
distributed_efficiency = [distributed_speedup[i] / num_workers for i in range(len(distributed_speedup))]

# Calculate moving averages
window_size = 50
y_data_sequential_ma = moving_average(y_data_sequential, window_size)
y_data_parallel_p_ma = moving_average(y_data_parallel_p, window_size)
y_data_parallel_1_ma = moving_average(y_data_parallel_1, window_size)
y_data_distributed_d_ma = moving_average(y_data_distributed_d, window_size)
y_data_distributed_1_ma = moving_average(y_data_distributed_1, window_size)
parallel_speedup_ma = moving_average(parallel_speedup, window_size)
parallel_scalability_ma = moving_average(parallel_scalability, window_size)
parallel_efficiency_ma = moving_average(parallel_efficiency, window_size)
distributed_speedup_ma = moving_average(distributed_speedup, window_size)
distributed_scalability_ma = moving_average(distributed_scalability, window_size)
distributed_efficiency_ma = moving_average(distributed_efficiency, window_size)

# Plot sequential
plot_data_and_trend(x_data, y_data_sequential, y_data_sequential_ma, headers[0], headers[1] + ' (s)',
                    'Plot of {} vs {}'.format(headers[1], headers[0]), './plots/sequential.png')

# Plot parallel p workers
plot_data_and_trend(x_data, y_data_parallel_p, y_data_parallel_p_ma, headers[0], headers[1] + ' (s)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    f'./plots/parallel_{num_workers}.png')

# Plot parallel 1 worker
plot_data_and_trend(x_data, y_data_parallel_1, y_data_parallel_1_ma, headers[0], headers[1] + ' (s)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    './plots/parallel_1.png')

# Plot parallel_speedup
plot_data_and_trend(x_data, parallel_speedup, parallel_speedup_ma, headers[0], 'Speedup',
                    'Plot of {} vs {} ({} workers)'.format('Speedup', headers[0], num_workers),
                    './plots/parallel_speedup.png')

# Plot parallel_scalability
plot_data_and_trend(x_data, parallel_scalability, parallel_scalability_ma, headers[0], 'Scalability',
                    'Plot of {} vs {} ({} workers)'.format('Scalability', headers[0], num_workers),
                    './plots/parallel_scalability.png')

# Plot parallel_efficiency
plot_data_and_trend(x_data, parallel_efficiency, parallel_efficiency_ma, headers[0], 'Efficiency',
                    'Plot of {} vs {} ({} workers)'.format('Efficiency', headers[0], num_workers),
                    './plots/parallel_efficiency.png')

# Plot distributed d processes
plot_data_and_trend(x_data, y_data_distributed_d, y_data_distributed_d_ma, headers[0], headers[1] + ' (s)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    f'./plots/distributed_{num_workers}.png')

# Plot distributed 1 process
plot_data_and_trend(x_data, y_data_distributed_1, y_data_distributed_1_ma, headers[0], headers[1] + ' (s)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    './plots/distributed_1.png')

# Plot distributed_speedup
plot_data_and_trend(x_data, distributed_speedup, distributed_speedup_ma, headers[0], 'Speedup',
                    'Plot of {} vs {} ({} workers)'.format('Speedup', headers[0], num_workers),
                    './plots/distributed_speedup.png')

# Plot distributed_scalability
plot_data_and_trend(x_data, distributed_scalability, distributed_scalability_ma, headers[0], 'Scalability',
                    'Plot of {} vs {} ({} workers)'.format('Scalability', headers[0], num_workers),
                    './plots/distributed_scalability.png')

# Plot distributed_efficiency
plot_data_and_trend(x_data, distributed_efficiency, distributed_efficiency_ma, headers[0], 'Efficiency',
                    'Plot of {} vs {} ({} workers)'.format('Efficiency', headers[0], num_workers),
                    './plots/distributed_efficiency.png')
