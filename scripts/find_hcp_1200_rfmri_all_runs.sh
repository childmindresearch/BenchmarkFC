#!/bin/bash

set -ex

# check that HCP_1200_DIR is set
[[ -n "$HCP_1200_DIR" ]]
[[ -n "$PROJECT_ROOT" ]]

CWD=$(pwd)

cd "$HCP_1200_DIR"
find . -name '*rfMRI*MSMAll_hp2000_clean.dtseries.nii' | cut -d / -f 2,5 | sort \
    > "${PROJECT_ROOT}/resources/hcp_1200_rfmri_all_runs.txt"

cd $CWD
