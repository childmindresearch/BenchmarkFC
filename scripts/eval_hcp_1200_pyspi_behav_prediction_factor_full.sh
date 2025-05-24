#!/bin/bash

#SBATCH --job-name=eval_pyspi_predict
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-567
#SBATCH --time=00:30:00

# Prediction of 4 factor targets using all SPIs.

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=4

spi_list=( $(cat "resources/spi_lists/spi_list_select_300s_142.txt") )
targets=( Dissatisfaction Cognition Support Emotion )

num_spis=142
target_idx=$(( SLURM_ARRAY_TASK_ID / num_spis ))
spi_idx=$(( SLURM_ARRAY_TASK_ID % num_spis))

spi=${spi_list[spi_idx]}
target=${targets[target_idx]}

seed="2142"
out_dir="results/hcp_1200_pyspi_behav_prediction_factor_full"

echo $spi $target

uv run --no-sync python scripts/eval_hcp_1200_pyspi_behav_prediction.py \
    --spi ${spi} --target ${target} --seed ${seed} --out-dir ${out_dir}
