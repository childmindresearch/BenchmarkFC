#!/bin/bash

#SBATCH --job-name=eval_skarf_predict
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-95
#SBATCH --time=00:30:00

# Prediction of 4 factor targets using 12 skarf funcs.

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=4

lags=( 0 1 )

func_list=(
    cov_empirical
    cov_graphicallasso
    prec_empirical
    prec_graphicallasso
    linear_ols
    linear_ridge
    linear_lasso
    linear_enet
    linear_lasso-pos
    linear_enet-pos
    linear_pls
    linear_pca-ridge
)

targets=( Dissatisfaction Cognition Support Emotion )

num_funcs=${#func_list[@]}
num_targets=${#targets[@]}

lag_idx=$(( SLURM_ARRAY_TASK_ID / (num_funcs * num_targets) ))
target_func_idx=$(( SLURM_ARRAY_TASK_ID % (num_funcs * num_targets) ))
target_idx=$(( target_func_idx / num_funcs ))
func_idx=$(( target_func_idx % num_funcs ))

lag=${lags[lag_idx]}
target=${targets[target_idx]}
func=${func_list[func_idx]}

seed="2142"
out_dir="results/hcp_1200_skarf_behav_prediction_factor_full"

echo $lag $target $func

uv run --no-sync python scripts/eval_hcp_1200_skarf_behav_prediction.py \
    --func ${func} --target ${target} --lag ${lag} --seed ${seed} --out-dir ${out_dir}
