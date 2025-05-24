#!/bin/bash

#SBATCH --job-name=eval_pyspi_predict
#SBATCH --partition=RM-shared
#SBATCH --ntasks=8
#SBATCH --array=0-3
#SBATCH --time=02:00:00

# Permutation test with permuted targets to get empirical null distribution.

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=8

# empirical covariance spi
spi="cov_EmpiricalCovariance"

# Factor targets
targets=( Dissatisfaction Cognition Support Emotion )
target=${targets[SLURM_ARRAY_TASK_ID]}

seed="6275"
out_dir="results/hcp_1200_pyspi_behav_prediction_perm_test"
n_splits=200

echo $spi $target

uv run --no-sync python scripts/eval_hcp_1200_pyspi_behav_prediction.py \
    --spi $spi \
    --target $target \
    --n-splits $n_splits \
    --perm-test \
    --seed $seed \
    --out-dir $out_dir
