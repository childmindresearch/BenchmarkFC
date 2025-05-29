#!/bin/bash

#SBATCH --job-name=test_profile_skarf
#SBATCH --partition=RM-shared
#SBATCH --ntasks=4
#SBATCH --array=0-11
#SBATCH --time=01:00:00

set -ex

# export all environment variables
set -a
source .env
set +a

[[ -n "$PROJECT_ROOT" ]]

export OMP_NUM_THREADS=2

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

func=${func_list[SLURM_ARRAY_TASK_ID]}

sub="181131"
lags="0 1"
order=1

out_dir="data/hcp_1200_rfmri_schaefer_skarf_test_profile"

for lag in $lags; do
    uv run --no-sync python scripts/compute_hcp_1200_schaefer_skarf.py \
        --lag $lag --order $order --out-dir $out_dir $func $sub
done
