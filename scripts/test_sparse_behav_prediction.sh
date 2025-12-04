#!/bin/bash
# Test script for sparse behavioral prediction on a small subset of subjects

set -e

# Activate environment
source /home/aerkent/skarf-experiments/.venv/bin/activate

# Source .env file to get all environment variables
set -a
source /home/aerkent/skarf-experiments/.env
set +a

# Data path
DATA_PATH="/srv/projects/skarf/data_aggregation/hcp_1200_rfmri_schaefer.parquet"

# Test parameters
N_SUBJECTS=50
SPARSITY=0.8
TARGET="Cognition"
SEED=2142

# Output directory for test results
OUT_DIR="${PROJECT_ROOT}/results/hcp_1200_sparse_behav_prediction_test"

echo "=========================================="
echo "Testing Sparse Behavioral Prediction"
echo "=========================================="
echo "Using ${N_SUBJECTS} subjects"
echo "Sparsity: ${SPARSITY}"
echo "Target: ${TARGET}"
echo ""

# Test 1: PySPI - Empirical Covariance
echo "[1/3] Testing pyspi - cov_EmpiricalCovariance..."
uv run python scripts/eval_hcp_1200_sparse_behav_prediction.py \
    --method pyspi \
    --func cov_EmpiricalCovariance \
    --data-path "${DATA_PATH}" \
    --target "${TARGET}" \
    --sparsity ${SPARSITY} \
    --n-subjects ${N_SUBJECTS} \
    --seed ${SEED} \
    --out-dir "${OUT_DIR}"

# Test 2: Skarf - Empirical Covariance
echo "[2/3] Testing skarf - cov_empirical..."
uv run python scripts/eval_hcp_1200_sparse_behav_prediction.py \
    --method skarf \
    --func cov_empirical \
    --data-path "${DATA_PATH}" \
    --target "${TARGET}" \
    --sparsity ${SPARSITY} \
    --n-subjects ${N_SUBJECTS} \
    --seed ${SEED} \
    --out-dir "${OUT_DIR}"

# Test 3: Skarf - Sparse Linear
echo "[3/3] Testing skarf - linear_lasso..."
uv run python scripts/eval_hcp_1200_sparse_behav_prediction.py \
    --method skarf \
    --func linear_lasso \
    --data-path "${DATA_PATH}" \
    --target "${TARGET}" \
    --sparsity ${SPARSITY} \
    --n-subjects ${N_SUBJECTS} \
    --seed ${SEED} \
    --out-dir "${OUT_DIR}"

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Results saved to: ${OUT_DIR}"
echo ""
echo "To check results:"
echo "  ls ${OUT_DIR}/sparsity-0.80*/*/target-${TARGET}/"
