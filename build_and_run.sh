#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")" || exit

cmake .
make

./build/SPM
python3 ./scripts/plot.py