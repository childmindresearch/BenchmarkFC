
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
