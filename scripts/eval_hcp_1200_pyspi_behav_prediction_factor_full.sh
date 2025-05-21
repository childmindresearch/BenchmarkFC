#!/bin/bash

#SBATCH --job-name=eval_pyspi_predict
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-567
#SBATCH --time=02:00:00

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
echo $spi $target

seed="2142"
out_dir="results/hcp_1200_pyspi_behav_prediction_factor_full"

uv run --no-sync python scripts/eval_hcp_1200_pyspi_behav_prediction.py \
    --spi ${spi} --target ${target} --seed ${seed} --out-dir ${out_dir}
