
download_schaefer:
	bash scripts/download_schaefer_parcellations.sh

download_hcp_1200:
	mkdir -p logs/download_hcp_1200 2>/dev/null
	sbatch -o logs/download_hcp_1200/slurm-%j.out scripts/download_hcp_1200.sh

download_misc_files:
    bash scripts/download_misc_files.sh

find_hcp_1200_rfmri_all_runs:
    bash scripts/find_hcp_1200_rfmri_all_runs.sh

compute_hcp_1200_rfmri_fd:
    uv run python scripts/compute_hcp_1200_rfmri_fd.py
