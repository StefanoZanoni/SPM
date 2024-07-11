import os
import csv
import matplotlib.pyplot as plt

cwd = os.getcwd()
file_path_sequential = os.path.join(cwd, './results/sequential.csv')
file_path_parallel = os.path.join(cwd, './results/parallel.csv')

# Read the CSV file
with open(file_path_sequential, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data_sequential = [float(row[0]) for row in data]
y_data_sequential = [float(row[1]) for row in data]

with open(file_path_parallel, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data_parallel = [float(row[0]) for row in data]
y_data_parallel = [float(row[1]) for row in data]

# Plot the data
plt.figure(figsize=(8, 8))
plt.plot(x_data_sequential, y_data_sequential)
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.grid(True)
plt.savefig('./plots/sequential.png')
plt.close()

# Plot the data
plt.figure(figsize=(8, 8))
plt.plot(x_data_parallel, y_data_parallel)
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.grid(True)
plt.savefig('./plots/parallel.png')
plt.close()
