#!/bin/bash

cd "$(dirname "$0")" || exit

# Compile the project
export CXX=mpicxx
cmake .
make

chmod +x ./fastflow-master/ff/mapping_string.sh > /dev/null
echo y > ./fastflow-master/ff/mapping_string.sh

# List of CPU counts to test
cpu_counts=(2 4 8 16 20)

# Run the sequential application
./build/sequential
./build/parallel 1
mpirun -n 1 ./build/distributed

# Run the parallel and distributed applications
for cpus in "${cpu_counts[@]}"; do
    ./build/parallel "${cpus}"
    mpirun -n "${cpus}" --oversubscribe ./build/distributed
done

python3 ./scripts/plot.py
python3 ./scripts/statistics.py
