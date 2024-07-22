#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")" || exit

cmake .
make

# Determine the number of workers
if [ -z "$1" ]; then
  num_workers=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
else
  num_workers=$1
fi

./build/SPM "$num_workers"
python3 ./scripts/plot.py --workers "$num_workers"
python3 ./scripts/statistics.py --workers "$num_workers"
