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

mkdir -p "slurm_scripts"
mkdir -p "slurm_scripts/sequential"
mkdir -p "slurm_scripts/parallel"
mkdir -p "slurm_scripts/distributed"

# Create SLURM scripts for sequential application
cat <<EOT > slurm_scripts/sequential/slurm_sequential.sh
#!/bin/bash
#SBATCH --job-name=sz_s                                               # Job name
#SBATCH --output=slurm_scripts/sequential/output_sequential%j.txt               # Output file
#SBATCH --error=slurm_scripts/sequential/error_sequential%j.txt                 # Error file
#SBATCH --nodes=1                                                               # Number of nodes
#SBATCH --ntasks=1                                                              # Number of tasks
#SBATCH --cpus-per-task=1                                                       # Number of CPU cores per task
#SBATCH --time=01:00:00                                                         # Time limit hrs:min:sec
#SBATCH --partition=normal                                                      # Partition name

# Load necessary modules
module load gnu12/12.2.0
module load openmpi4/4.1.5

# Run the sequential application
srun ./build/sequential
srun ./build/parallel 1
srun --mpi=pmix -n 1 ./build/distributed
EOT

# Create SLURM scripts for multi-threaded application
for cpus in "${cpu_counts[@]}"; do
    cat <<EOT > slurm_scripts/parallel/slurm_parallel_"${cpus}".sh
#!/bin/bash
#SBATCH --job-name=sz_p_${cpus}                                    # Job name
#SBATCH --output=slurm_scripts/distributed/output_parallel_${cpus}cpus_%j.txt  # Output file
#SBATCH --error=slurm_scripts/distributed/error_parallel_${cpus}cpus_%j.txt    # Error file
#SBATCH --nodes=1                                                              # Number of nodes
#SBATCH --ntasks=1                                                             # Number of tasks
#SBATCH --cpus-per-task=${cpus}                                                # Number of CPU cores per task
#SBATCH --time=01:00:00                                                        # Time limit hrs:min:sec
#SBATCH --partition=normal                                                     # Partition name

# Load necessary modules
module load gnu12/12.2.0
module load openmpi4/4.1.5

# Run the multi-threaded application
srun ./build/parallel ${cpus}
EOT
done

# Create SLURM scripts for multi-process application
for cpus in "${cpu_counts[@]}"; do
    cat <<EOT > slurm_scripts/distributed/slurm_distributed_"${cpus}".sh
#!/bin/bash
#SBATCH --job-name=sz_d_${cpus}                                    # Job name
#SBATCH --output=slurm_scripts/distributed/output_distributed_${cpus}cpus_%j.txt  # Output file
#SBATCH --error=slurm_scripts/distributed/error_distributed_${cpus}cpus_%j.txt    # Error file
#SBATCH --nodes=4                                                                 # Number of nodes
#SBATCH --ntasks=${cpus}                                                          # Number of tasks
#SBATCH --cpus-per-task=1                                                         # Number of CPU cores per task
#SBATCH --time=01:00:00                                                           # Time limit hrs:min:sec
#SBATCH --partition=normal                                                        # Partition name

# Load necessary modules
module load gnu12/12.2.0
module load openmpi4/4.1.5

# Run the multi-process application
srun --mpi=pmix -n ${cpus} ./build/distributed
EOT
done

chmod +x slurm_scripts/sequential/slurm_sequential.sh
sbatch --wait slurm_scripts/sequential/slurm_sequential.sh

for cpus in "${cpu_counts[@]}"; do
    chmod +x slurm_scripts/parallel/slurm_parallel_"${cpus}".sh
    chmod +x slurm_scripts/distributed/slurm_distributed_"${cpus}".sh
    sbatch --wait slurm_scripts/parallel/slurm_parallel_"${cpus}".sh
    sbatch --wait slurm_scripts/distributed/slurm_distributed_"${cpus}".sh
done

python3 ./scripts/plot.py
python3 ./scripts/statistics.py