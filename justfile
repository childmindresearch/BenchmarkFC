# Download cifti format fsLR 32k 7 network Schaefer parcellations from Yeo Lab
# repository.
download_schaefer:
    bash scripts/download_schaefer_parcellations.sh

# Download HCP-1200 preprocessed fsLR 32k resting state data and motion parameters from
# the hcp-openaccess S3 bucket.
download_hcp_1200:
    mkdir -p logs/download_hcp_1200 2>/dev/null
    sbatch -o logs/download_hcp_1200/slurm-%j.out scripts/download_hcp_1200.sh

# Download other misc resource files
# - HCP subject lists
# - HCP behavioral column lists
# - HCP phenotypic data lookup table
download_misc_files:
    bash scripts/download_misc_files.sh

# Compute framewise displacement (FD), mean FD, and motion spikes using motion parameter
# data from HCP. Follows procedure from Li et al., 2019.
compute_hcp_1200_rfmri_fd:
    uv run python scripts/compute_hcp_1200_rfmri_fd.py

# Motion analysis and filtering.
# - Compute subject and volume level FD thresholds.
# - Plot histograms of motion data.
analyze_hcp_1200_rfmri_fd:
    uv run jupyter execute --inplace notebooks/analyze_hcp_1200_rfmri_fd.ipynb

# Generate final HCP subject list, including subjects with complete behavioral and
# resting data after motion filtering.
filter_hcp_1200_subjects:
    uv run jupyter execute --inplace notebooks/filter_hcp_1200_subjects.ipynb

# Generate videos of raw HCP data. Not strictly a necessary step, but useful for getting
# a sense of the data.
visualize_raw_hcp_timeseries:
    uv run jupyter execute --inplace notebooks/visualize_raw_hcp_timeseries.ipynb

# Visually check the impact of different timeseries preprocessing steps on synthetic
# time series data.
test_synthetic_timeseries_preprocessing:
    uv run jupyter execute --inplace notebooks/test_synthetic_timeseries_preprocessing.ipynb

# Similar check of timeseries preprocessing, but for actual HCP data.
test_hcp_timeseries_preprocessing:
    uv run jupyter execute --inplace notebooks/test_hcp_timeseries_preprocessing.ipynb

# Closer inspection of the effect of padding type on boundary effects introduced by
# bandpass filtering.
test_hcp_timeseries_filtering:
    uv run jupyter execute --inplace notebooks/test_hcp_timeseries_filtering.ipynb

# Extract preprocessed parcellated HCP time series from cifti space 32k fsLR data.
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

# Visualize final preprocessed parcellated HCP time series.
visualize_parcellated_hcp_timeseries:
    uv run jupyter execute --inplace notebooks/visualize_parcellated_hcp_timeseries.ipynb

# Loading the PySPI SPIs requires importing a lot of modules. Here we just do it once
# and reused the cached JSON of SPI configs. Expected to generate a complete list of 284
# SPIs in resources/spi_lists.
export_spi_configs:
    uv run python scripts/export_spi_configs.py

# Run all SPIs on a single example subject with a generous time constraint.
test_profile_pyspi:
    mkdir -p logs/test_profile_pyspi 2>/dev/null
    sbatch -o logs/test_profile_pyspi/slurm-%A_%a.out scripts/test_profile_pyspi.sh

# Analyze PySPI SPI test results, plot run time and memory usage, and filter SPIs for
# errors and long run time. Expected to produce a list of selected SPIs in
# resources/spi_lists.
analyze_test_profile_pyspi:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_test_profile_pyspi.ipynb

# Compute selected PySPI SPI matrices on all of HCP 1200.
# NOTE: long-running large slurm job.
compute_hcp_1200_schaefer_pyspi:
    mkdir -p logs/compute_hcp_1200_schaefer_pyspi 2>/dev/null
    sbatch \
    -o logs/compute_hcp_1200_schaefer_pyspi/slurm-%A_%a.out \
    scripts/compute_hcp_1200_schaefer_pyspi.sh

# Fill in any missing/retry SPIs as needed.
compute_hcp_1200_schaefer_pyspi_subset:
    mkdir -p logs/compute_hcp_1200_schaefer_pyspi 2>/dev/null
    sbatch \
    -o logs/compute_hcp_1200_schaefer_pyspi/slurm-%j.out \
    scripts/compute_hcp_1200_schaefer_pyspi_subset.sh

# Remove behavioral measures from the original list of 58 that have insufficient
# variance across subjects.
filter_hcp_1200_behav_measures:
    uv run python scripts/filter_hcp_1200_behav_measures.py

# Analyze HCP behavioral measures.
# - Visualize raw and cleaned histograms for each measure.
# - Compute population-wide factors and visualize.
analyze_hcp_1200_behav:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_behav.ipynb

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
# using each of the PySPI SPIs as features.
eval_hcp_1200_pyspi_behav_prediction_factor_full:
    mkdir -p logs/eval_hcp_1200_pyspi_behav_prediction_factor_full 2>/dev/null
    sbatch \
    -o logs/eval_hcp_1200_pyspi_behav_prediction_factor_full/slurm-%A_%a.out \
    scripts/eval_hcp_1200_pyspi_behav_prediction_factor_full.sh

# Analysis and figures of PySPI behavioral prediction results.
analyze_hcp_1200_pyspi_behav_prediction:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_pyspi_behav_prediction.ipynb

# Test that skarf functions run without error, meet runtime cutoff, and produce sensible
# results.
test_profile_skarf:
    mkdir -p logs/test_profile_skarf 2>/dev/null
    sbatch -o logs/test_profile_skarf/slurm-%A_%a.out scripts/test_profile_skarf.sh

# Comute skarf matrices for all of HCP 1200.
compute_hcp_1200_schaefer_skarf:
    mkdir -p logs/compute_hcp_1200_schaefer_skarf 2>/dev/null
    sbatch \
    -o logs/compute_hcp_1200_schaefer_skarf/slurm-%j.out \
    scripts/compute_hcp_1200_schaefer_skarf.sh

# FC - behavioral prediction predicting 4 factor-based average measures, independently
# using 12 skarf funcs. Same as previous eval for pyspi.
eval_hcp_1200_skarf_behav_prediction_factor_full:
    mkdir -p logs/eval_hcp_1200_skarf_behav_prediction_factor_full 2>/dev/null
    sbatch \
    -o logs/eval_hcp_1200_skarf_behav_prediction_factor_full/slurm-%A_%a.out \
    scripts/eval_hcp_1200_skarf_behav_prediction_factor_full.sh

# Analysis and figures of skarf behavioral prediction results.
analyze_hcp_1200_skarf_behav_prediction:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_skarf_behav_prediction.ipynb

# Test sparse behavioral prediction on a small subset of subjects (50) to verify
# functionality before running on full dataset. Tests 3 method/func combinations:
# pyspi cov_EmpiricalCovariance, skarf cov_empirical, and skarf linear_lasso.
test_sparse_behav_prediction:
    bash scripts/test_sparse_behav_prediction.sh

# FC - behavioral prediction with 0.8 sparsity imposed on all connectivity matrices
# (both pyspi and skarf). Predicts Cognition target using all 161 method/func combinations
# (149 pyspi + 12 skarf) with joblib parallel execution (6 jobs × 4 cores = 24 cores).
eval_hcp_1200_sparse_behav_prediction_all:
    mkdir -p logs/eval_hcp_1200_sparse_behav_prediction_all 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_sparse_behav_prediction_parallel.py 2>&1 | tee logs/eval_hcp_1200_sparse_behav_prediction_all/run_$(date +%Y%m%d_%H%M%S).log

# Analysis and figures comparing sparse vs non-sparse behavioral prediction results.
# Analyzes results with 0.8 imposed sparsity on all methods (pyspi and skarf).
analyze_hcp_1200_sparse_behav_prediction:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_sparse_behav_prediction.ipynb

# FC - homotopic FC benchmark across PySPI and skarf.
# Computes rank-based Schaefer-7 network-block within-network homotopy scores and
# deterministic between-network cross-hemisphere empirical references for all valid
# pyspi combos plus skarf lag-0/lag-1 variants.
eval_hcp_1200_homotopic_fc:
    mkdir -p logs/eval_hcp_1200_homotopic_fc 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_homotopic_fc_parallel.py 2>&1 | tee logs/eval_hcp_1200_homotopic_fc/run_$(date +%Y%m%d_%H%M%S).log

# Analysis and figures for homotopic FC benchmark results.
analyze_hcp_1200_homotopic_fc:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_homotopic_fc.ipynb

# FC - demographics (gender + age) prediction from functional connectivity.
# Predicts gender (KRR + threshold, He2020) and age (KRR) for all pyspi combos plus
# skarf lag-0/lag-1 variants with configurable joblib parallel execution.
eval_hcp_1200_demographics_prediction:
    mkdir -p logs/eval_hcp_1200_demographics_prediction 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_demographics_prediction_parallel.py 2>&1 | tee logs/eval_hcp_1200_demographics_prediction/run_$(date +%Y%m%d_%H%M%S).log

# Analysis and figures for demographics (gender + age) prediction results.
analyze_hcp_1200_demographics_prediction:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_demographics_prediction.ipynb

# FC - weight-distance benchmark across PySPI and skarf.
# Computes signed Spearman correlations between Schaefer-200 centroid distances and
# FC weights, with Alexander-Bloch spin-null summaries and directed upper/lower
# triangle handling.
eval_hcp_1200_weight_distance:
    mkdir -p logs/eval_hcp_1200_weight_distance 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_weight_distance_parallel.py 2>&1 | tee logs/eval_hcp_1200_weight_distance/run_$(date +%Y%m%d_%H%M%S).log

# Analysis and figures for the weight-distance benchmark results.
analyze_hcp_1200_weight_distance:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_weight_distance.ipynb

# Ensemble (krakencoder) pipeline
# ---------------------------------------------------------------------------
# Fuses the top-5/10/15 ranked FC methods (from the combined leaderboard) into
# new connectivity matrices via krakencoder, fused per run (one ensemble
# matrix per subject per session/run, matching the main aggregated parquet's
# [sub, ses, run] structure) so reliability metrics can be computed for
# ensemble combos, then re-runs the same benchmark suite on the fused
# matrices.

# Phase 1 - select top-N ranked methods and export their per-subject averaged
# input matrices for krakencoder bridge training (krakencoder's precomputed
# fusion-latent target is inherently subject-level, so bridge training still
# uses subject-averaged inputs - only the final ensemble reconstruction in
# Phase 3 fuses per run).
select_ensemble_methods:
    uv run --env-file .env python scripts/select_ensemble_methods.py

export_ensemble_input_matrices:
    uv run --env-file .env python scripts/export_ensemble_input_matrices.py

# Phase 2 - fetch krakencoder's precomputed fusion latents/subject split, build
# per-flavor bridging inputs, and train each flavor's encoder/decoder into the
# pretrained fusion latent space.
fetch_krakencoder_fusion_target:
    uv run --env-file .env python scripts/fetch_krakencoder_fusion_target.py

build_krakencoder_bridge_inputs:
    uv run --env-file .env python scripts/build_krakencoder_bridge_inputs.py

train_ensemble_krakencoder_bridge:
    uv run --env-file .env python scripts/train_ensemble_krakencoder_bridge.py

# Phase 3 - apply reconstruction rules (simple/weighted average, reference
# decoder) per run and write the ensemble parquet in the same schema as the main
# aggregated parquet.
build_ensemble_matrices:
    uv run --env-file .env python scripts/build_ensemble_matrices.py

# Phase 4 - re-run the benchmark suite (behavioral prediction, demographics,
# info density, reliability) on the ensemble combos.
eval_hcp_1200_ensemble_behav_prediction:
    mkdir -p logs/eval_hcp_1200_ensemble_behav_prediction 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_ensemble_behav_prediction.py 2>&1 | tee logs/eval_hcp_1200_ensemble_behav_prediction/run_$(date +%Y%m%d_%H%M%S).log

eval_hcp_1200_ensemble_demographics_prediction:
    mkdir -p logs/eval_hcp_1200_ensemble_demographics_prediction 2>/dev/null
    uv run --env-file .env python scripts/eval_hcp_1200_ensemble_demographics_prediction.py 2>&1 | tee logs/eval_hcp_1200_ensemble_demographics_prediction/run_$(date +%Y%m%d_%H%M%S).log

compute_ensemble_info_density:
    mkdir -p logs/compute_ensemble_info_density 2>/dev/null
    uv run --env-file .env python scripts/compute_ensemble_info_density.py 2>&1 | tee logs/compute_ensemble_info_density/run_$(date +%Y%m%d_%H%M%S).log

compute_ensemble_reliability:
    mkdir -p logs/compute_ensemble_reliability 2>/dev/null
    uv run --env-file .env python scripts/compute_ensemble_reliability.py 2>&1 | tee logs/compute_ensemble_reliability/run_$(date +%Y%m%d_%H%M%S).log

# Phase 5 - re-run the combined ranking notebook to merge ensemble rows into
# the leaderboard (writes combined_benchmark_scores_ranked.csv).
analyze_hcp_1200_benchmark_scores_combined:
    uv run --env-file .env jupyter execute --inplace notebooks/analyze_hcp_1200_benchmark_scores_combined.ipynb