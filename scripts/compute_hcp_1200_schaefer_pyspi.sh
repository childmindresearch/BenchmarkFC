#!/bin/bash

#SBATCH --job-name=compute_pyspi
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

set -ex

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=2

spi_list=( $(cat "resources/spi_lists/spi_list_select_1200s_209.txt") )

for spi in ${spi_list[@]}; do
    uv run python scripts/compute_hcp_1200_schaefer_pyspi.py $spi
done
