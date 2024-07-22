import os
import csv
import psutil
import argparse
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()
file_path_sequential = os.path.join(cwd, './results/sequential.csv')
file_path_parallel_p = os.path.join(cwd, './results/parallel_p.csv')
file_path_parallel_1 = os.path.join(cwd, './results/parallel_1.csv')


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

# Calculate speedup
speedup = [y_data_sequential[i] / y_data_parallel_p[i] for i in range(len(y_data_sequential))]

# Calculate scalability
scalability = [y_data_parallel_1[i] / y_data_parallel_p[i] for i in range(len(y_data_parallel_p))]

# Calculate efficiency
efficiency = [speedup[i] / num_workers for i in range(len(speedup))]

# Calculate moving averages
window_size = 50
y_data_sequential_ma = moving_average(y_data_sequential, window_size)
y_data_parallel_p_ma = moving_average(y_data_parallel_p, window_size)
y_data_parallel_1_ma = moving_average(y_data_parallel_1, window_size)
speedup_ma = moving_average(speedup, window_size)
scalability_ma = moving_average(scalability, window_size)
efficiency_ma = moving_average(efficiency, window_size)

# Plot sequential
plot_data_and_trend(x_data, y_data_sequential, y_data_sequential_ma, headers[0], headers[1] + ' (ms)',
                    'Plot of {} vs {}'.format(headers[1], headers[0]), './plots/sequential.png')

# Plot parallel p workers
plot_data_and_trend(x_data, y_data_parallel_p, y_data_parallel_p_ma, headers[0], headers[1] + ' (ms)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    f'./plots/parallel_{num_workers}.png')

# Plot parallel 1 worker
plot_data_and_trend(x_data, y_data_parallel_1, y_data_parallel_1_ma, headers[0], headers[1] + ' (ms)',
                    'Plot of {} vs {} ({} workers)'.format(headers[1], headers[0], num_workers),
                    './plots/parallel_1.png')

# Plot speedup
plot_data_and_trend(x_data, speedup, speedup_ma, headers[0], 'Speedup',
                    'Plot of {} vs {} ({} workers)'.format('Speedup', headers[0], num_workers),
                    './plots/parallel_speedup.png')

# Plot scalability
plot_data_and_trend(x_data, scalability, scalability_ma, headers[0], 'Scalability',
                    'Plot of {} vs {} ({} workers)'.format('Scalability', headers[0], num_workers),
                    './plots/parallel_scalability.png')

# Plot efficiency
plot_data_and_trend(x_data, efficiency, efficiency_ma, headers[0], 'Efficiency',
                    'Plot of {} vs {} ({} workers)'.format('Efficiency', headers[0], num_workers),
                    './plots/parallel_efficiency.png')

