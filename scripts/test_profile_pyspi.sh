#!/bin/bash

#SBATCH --job-name=test_profile_pyspi
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-246
#SBATCH --time=04:00:00

set -ex

# export all environment variables
set -a
source .env
set +a

[[ -n "$PROJECT_ROOT" ]]

export OMP_NUM_THREADS=2

spi_list=( $(cat "resources/spi_lists/spi_list_distinct_247.txt") )
spi=${spi_list[SLURM_ARRAY_TASK_ID]}

sub="181131"
out_dir="data/hcp_1200_rfmri_schaefer_pyspi_test_profile"

uv run --no-sync python scripts/compute_hcp_1200_schaefer_pyspi.py \
    --out-dir $out_dir $spi $sub
