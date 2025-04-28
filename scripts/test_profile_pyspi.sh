#!/bin/bash

#SBATCH --job-name=test_profile_pyspi
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-283
#SBATCH --time=05:00:00

set -ex

# export all environment variables
set -a
source .env
set +a

[[ -n "$PROJECT_ROOT" ]]

# testing n < d, n = d and scaling n and d
n_samples="[50,100,200,400,400]"
n_features="[100,100,100,100,200]"
parc_size=200

uv run python scripts/test_profile_pyspi.py \
    --spi_id $SLURM_ARRAY_TASK_ID \
    --n-samples $n_samples --n-features $n_features --parc-size $parc_size
