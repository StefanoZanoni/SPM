import os
import csv
import matplotlib.pyplot as plt

cwd = os.getcwd()
file_path = os.path.join(cwd, './results/sequential.csv')

# Read the CSV file
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    data = [row for row in csv_reader]

x_data = [float(row[0]) for row in data]
y_data = [float(row[1]) for row in data]

# Plot the data
plt.plot(x_data, y_data)
plt.xlabel(headers[0])
plt.ylabel(headers[1] + ' (ms)')
plt.title('Plot of {} vs {}'.format(headers[1], headers[0]))
plt.grid(True)
plt.savefig('./plots/sequential.png')
