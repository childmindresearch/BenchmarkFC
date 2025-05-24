#!/bin/bash

#SBATCH --job-name=eval_pyspi_predict
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-58
#SBATCH --time=00:30:00

# Prediction of all 55 targets + 4 factors using empirical covariance.
# Goal is to get a baseline of prediction for each target.

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=4

spi="cov_EmpiricalCovariance"

# 55 behavioral measures
targets=( $(cat resources/column_lists/55behaviors_with_var.txt) )
# Prepend factors
targets=( Dissatisfaction Cognition Support Emotion ${targets[@]} )

target=${targets[SLURM_ARRAY_TASK_ID]}

seed="3293"
out_dir="results/hcp_1200_pyspi_behav_prediction_all_targets_cov"

echo $spi $target

uv run --no-sync python scripts/eval_hcp_1200_pyspi_behav_prediction.py \
    --spi ${spi} --target ${target} --seed ${seed} --out-dir ${out_dir}
