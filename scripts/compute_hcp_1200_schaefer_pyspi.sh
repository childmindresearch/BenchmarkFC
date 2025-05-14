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

spi_list_path="resources/spi_lists/spi_list_select_300s_142.txt"
sub_list_path="resources/subject_lists/hcp_complete_data_867_subject_list.txt"

subs_per_batch=175
num_total_subs=867
sub_start_idx=$(( SLURM_ARRAY_TASK_ID * subs_per_batch ))
sub_stop_idx=$(( sub_start_idx + subs_per_batch ))
sub_list=$(sed -n "$((sub_start_idx+1)),${sub_stop_idx}p" ${sub_list_path})

parallel --jobs 64 \
    uv run --no-sync python scripts/compute_hcp_1200_schaefer_pyspi.py \
    {1} {2} :::: ${spi_list_path} ::: ${sub_list}
