#!/bin/bash

#SBATCH --job-name=compute_pyspi
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --time=12:00:00

set -e

# export all environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=2

spi_list="prec_EmpiricalCovariance prec_EllipticEnvelope prec_GraphicalLassoCV prec_LedoitWolf prec_MinCovDet prec_OAS prec_ShrunkCovariance tlmi_kraskov_NN-4"
sub_list_path="resources/subject_lists/hcp_complete_data_867_subject_list.txt"

parallel --jobs 64 \
    uv run --no-sync python scripts/compute_hcp_1200_schaefer_pyspi.py \
    {1} {2} ::: ${spi_list} :::: ${sub_list_path}
