#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")" || exit

export CXX=mpicxx

cmake .
make

# Determine the number of workers
if [ -z "$1" ]; then
    # Get the number of cores per socket
    cores_per_socket=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
    # Get the number of sockets
    sockets=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
    # Calculate the total number of physical cores
    num_workers=$((cores_per_socket * sockets))
else
  num_workers=$1
fi

mpirun -n "$num_workers" ./build/SPM "$num_workers"
#python3 ./scripts/plot.py --workers "$num_workers"
#python3 ./scripts/statistics.py --workers "$num_workers"
