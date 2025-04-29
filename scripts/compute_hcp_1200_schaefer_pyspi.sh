#!/bin/bash

#SBATCH --job-name=compute_pyspi
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --time=1-00:00:00

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=2

spi_list=( $(cat "resources/spi_lists/spi_list_select_300s_147.txt") )

spis_per_task=30
spi_start_idx=$(( SLURM_ARRAY_TASK_ID * spis_per_task ))
num_spis=${#spi_list[@]}

for ((ii=0; ii<spis_per_task; ii++)); do
    spi_idx=$((spi_start_idx + ii))
    if (( spi_idx >= num_spis )); then
        break
    fi

    spi=${spi_list[spi_idx]}
    uv run python scripts/compute_hcp_1200_schaefer_pyspi.py $spi
done
