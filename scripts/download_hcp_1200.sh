#!/bin/bash

#SBATCH --job-name=download_hcp_1200
#SBATCH --partition=RM-shared
#SBATCH --ntasks=8
#SBATCH --time=1-00:00:00

set -ex

# export all environment variables
set -a
source .env
set +a

# check that HCP_1200_DIR is set
[[ -n "$HCP_1200_DIR" ]]

aws s3 sync s3://hcp-openaccess/HCP_1200/ "${HCP_1200_DIR}" \
  --exclude "*" \
  --include "*/MNINonLinear/Results/rfMRI_*/rfMRI_*_Atlas_MSMAll*.dtseries.nii" \
  --include "*/MNINonLinear/Results/rfMRI_*/Movement_Regressors.txt"
