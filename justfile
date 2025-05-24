
download_schaefer:
    bash scripts/download_schaefer_parcellations.sh

download_hcp_1200:
    mkdir -p logs/download_hcp_1200 2>/dev/null
    sbatch -o logs/download_hcp_1200/slurm-%j.out scripts/download_hcp_1200.sh

download_misc_files:
    bash scripts/download_misc_files.sh

compute_hcp_1200_rfmri_fd:
    uv run python scripts/compute_hcp_1200_rfmri_fd.py

analyze_hcp_1200_rfmri_fd:
    uv run jupyter execute --inplace notebooks/analyze_hcp_1200_rfmri_fd.ipynb

filter_hcp_1200_subjects:
    uv run jupyter execute --inplace notebooks/filter_hcp_1200_subjects.ipynb

visualize_raw_hcp_timeseries:
    uv run jupyter execute --inplace notebooks/visualize_raw_hcp_timeseries.ipynb

test_synthetic_timeseries_preprocessing:
    uv run jupyter execute --inplace notebooks/test_synthetic_timeseries_preprocessing.ipynb

test_hcp_timeseries_preprocessing:
    uv run jupyter execute --inplace notebooks/test_hcp_timeseries_preprocessing.ipynb

test_hcp_timeseries_filtering:
    uv run jupyter execute --inplace notebooks/test_hcp_timeseries_filtering.ipynb

extract_hcp_schaefer_timeseries:
    mkdir -p logs/extract_hcp_schaefer_timeseries 2>/dev/null
    sbatch \
    --job-name extract_hcp \
    --nodes 1 \
    --partition RM \
    --time 04:00:00 \
    --export=ALL \
    --output logs/extract_hcp_schaefer_timeseries/slurm-%j.out \
    scripts/extract_hcp_schaefer_timeseries.py

visualize_parcellated_hcp_timeseries:
    uv run jupyter execute --inplace notebooks/visualize_parcellated_hcp_timeseries.ipynb

export_spi_configs:
    uv run python scripts/export_spi_configs.py

test_profile_pyspi:
    mkdir -p logs/test_profile_pyspi 2>/dev/null
    sbatch -o logs/test_profile_pyspi/slurm-%A_%a.out scripts/test_profile_pyspi.sh

analyze_test_profile_pyspi:
    uv run jupyter execute --inplace notebooks/analyze_test_profile_pyspi.ipynb

compute_hcp_1200_schaefer_pyspi:
    mkdir -p logs/compute_hcp_1200_schaefer_pyspi 2>/dev/null
    sbatch \
    -o logs/compute_hcp_1200_schaefer_pyspi/slurm-%A_%a.out \
    scripts/compute_hcp_1200_schaefer_pyspi.sh

# Remove behavioral measures from the original list of 58 that have insufficient
# variance across subjects.
filter_hcp_1200_behav_measures:
    uv run python scripts/filter_hcp_1200_behav_measures.py

# Analyze HCP behavioral measures.
# - Visualize raw and cleaned histograms for each measure.
# - Compute population-wide factors and visualize.
analyze_hcp_1200_behav:
    uv run jupyter execute --inplace notebooks/analyze_hcp_1200_behav.ipynb

# Baseline FC - behavioral prediction predicting 55 targets + 4 factors using only
# empirical covariance.
eval_hcp_1200_pyspi_behav_prediction_all_targets_cov:
    mkdir -p logs/eval_hcp_1200_pyspi_behav_prediction_all_targets_cov 2>/dev/null
    sbatch \
    -o logs/eval_hcp_1200_pyspi_behav_prediction_all_targets_cov/slurm-%A_%a.out \
    scripts/eval_hcp_1200_pyspi_behav_prediction_all_targets_cov.sh

# Permutation test to get empirical null distribution.
# Using empirical covariance and 4 factors.
eval_hcp_1200_pyspi_behav_prediction_perm_test:
    mkdir -p logs/eval_hcp_1200_pyspi_behav_prediction_perm_test 2>/dev/null
    sbatch \
    -o logs/eval_hcp_1200_pyspi_behav_prediction_perm_test/slurm-%A_%a.out \
    scripts/eval_hcp_1200_pyspi_behav_prediction_perm_test.sh

# FC - behavioral prediction predicting 4 factor-based average measures, independently
# using each of the 142 PySPI SPIs as features.
eval_hcp_1200_pyspi_behav_prediction_factor_full:
    mkdir -p logs/eval_hcp_1200_pyspi_behav_prediction_factor_full 2>/dev/null
    sbatch \
    -o logs/eval_hcp_1200_pyspi_behav_prediction_factor_full/slurm-%A_%a.out \
    scripts/eval_hcp_1200_pyspi_behav_prediction_factor_full.sh
