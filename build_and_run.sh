#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")" || exit

cmake .
make

./build/SPM 8
python3 ./scripts/plot.py --workers 8
