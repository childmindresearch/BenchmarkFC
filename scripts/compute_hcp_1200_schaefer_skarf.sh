#!/bin/bash

#SBATCH --job-name=compute_skarf
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

set -e

# export all environment variables
set -a
source .env
set +a

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
sub_list_path="resources/subject_lists/hcp_complete_data_867_subject_list.txt"

lags="0 1"
order=1
out_dir="data/hcp_1200_rfmri_schaefer_skarf"

parallel --jobs 64 \
    uv run --no-sync python scripts/compute_hcp_1200_schaefer_skarf.py \
    --lag {1} --order ${order} --out-dir ${out_dir} \
    {2} {3} ::: ${lags} ::: ${func_list[@]} :::: ${sub_list_path}
