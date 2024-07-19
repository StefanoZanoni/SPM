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


args = parse_arguments()
num_workers = args.workers

if num_workers <= 0:
    num_workers = psutil.cpu_count(logical=False)

cwd = os.getcwd()
file_path_sequential = os.path.join(cwd, './results/sequential.csv')
file_path_parallel_p = os.path.join(cwd, './results/parallel_p.csv')
file_path_parallel_1 = os.path.join(cwd, './results/parallel_1.csv')


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='full')[:len(data)]


with open(file_path_sequential, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data_sequential = [float(row[0]) for row in data]
y_data_sequential = [float(row[1]) for row in data]

with open(file_path_parallel_p, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data_parallel_p = [float(row[0]) for row in data]
y_data_parallel_p = [float(row[1]) for row in data]

# Calculate moving averages
window_size = 50
y_data_sequential_ma = moving_average(y_data_sequential, window_size)
y_data_parallel_p_ma = moving_average(y_data_parallel_p, window_size)
x_data_sequential_ma = x_data_sequential[:len(y_data_sequential_ma)]
x_data_parallel_p_ma = x_data_parallel_p[:len(y_data_parallel_p_ma)]

plt.figure(figsize=(10, 6))
plt.plot(x_data_sequential, y_data_sequential, label='Sequential')
plt.plot(x_data_sequential_ma, y_data_sequential_ma, label='Sequential trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.legend()
plt.grid(True)
plt.savefig('./plots/sequential.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x_data_parallel_p, y_data_parallel_p, label=f'Parallel (workers={num_workers})')
plt.plot(x_data_parallel_p_ma, y_data_parallel_p_ma, label='Parallel trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel.png')
plt.close()

speedup = [y_data_sequential[i] / y_data_parallel_p[i] for i in range(len(y_data_sequential))]
speedup_ma = moving_average(speedup, window_size)
x_data_speedup_ma = x_data_parallel_p[:len(speedup_ma)]

plt.figure(figsize=(10, 6))
plt.plot(x_data_parallel_p, speedup, label=f'Speedup (workers={num_workers})')
plt.plot(x_data_speedup_ma, speedup_ma, label='Speedup trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel('Speedup')
plt.title('Plot of the speedup')
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_speedup.png')
plt.close()

with open(file_path_parallel_1, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data_parallel_1 = [float(row[0]) for row in data]
y_data_parallel_1 = [float(row[1]) for row in data]

x_data_parallel_1_ma = x_data_parallel_1[:len(y_data_parallel_1)]
y_data_parallel_1_ma = moving_average(y_data_parallel_1, window_size)

plt.figure(figsize=(10, 6))
plt.plot(x_data_parallel_1, y_data_parallel_1, label='Parallel (workers=1)')
plt.plot(x_data_parallel_1_ma, y_data_parallel_1_ma, label='Parallel trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_1.png')
plt.close()

scalability = [y_data_parallel_1[i] / y_data_parallel_p[i] for i in range(len(y_data_parallel_p))]
scalability_ma = moving_average(scalability, window_size)

plt.figure(figsize=(10, 6))
plt.plot(x_data_parallel_p, scalability, label=f'Scalability (workers={num_workers})')
plt.plot(x_data_speedup_ma, speedup_ma, label='Scalability trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel('Scalability')
plt.title('Plot of the scalability')
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_scalability.png')
plt.close()

efficiency = [speedup[i] / num_workers for i in range(len(speedup))]
efficiency_ma = moving_average(efficiency, window_size)

plt.figure(figsize=(10, 6))
plt.plot(x_data_parallel_p, efficiency, label=f'Efficiency (workers={num_workers})')
plt.plot(x_data_speedup_ma, efficiency_ma, label='Efficiency trend', linestyle='--')
plt.xlabel(headers[0])
plt.ylabel('Efficiency')
plt.title('Plot of the efficiency')
plt.legend()
plt.grid(True)
plt.savefig('./plots/parallel_efficiency.png')
plt.close()
