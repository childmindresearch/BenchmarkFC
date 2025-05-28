## Methods

### Subject filtering

Commands:

- `compute_hcp_1200_rfmri_fd`
- `analyze_hcp_1200_rfmri_fd`
- `filter_hcp_1200_subjects`

Code:

- [compute_hcp_1200_rfmri_fd](../scripts/compute_hcp_1200_rfmri_fd.py)
- [analyze_hcp_1200_rfmri_fd.ipynb](../notebooks/analyze_hcp_1200_rfmri_fd.ipynb)
- [filter_hcp_1200_subjects](../notebooks/filter_hcp_1200_subjects.ipynb)

Figures:

![hcp_1200_rfmri_fd](../results/hcp_1200_rfmri_fd/hcp_1200_rfmri_fd.png)

Our initial subject pool included subjects from HCP-1200 with complete behavioral data for the 58 behavioral measures (Li et al., 2019, and others) and all four runs of 3T resting-state data (987 subjects).

We excluded subjects with mean framewise displacement (FD) greater than 0.3, or with one or more runs with at least 50% of volumes exceeding an FD motion spike threshold of 0.4. This resulted in a final sample size of 867 subjects.

FD was computed based on the movement parameters distributed with the preprocessed HCP-1200 data using the (Power et al., 2012) formulation. The two FD thresholds were computed by rounding the boxplot outlier threshold (75% + 1.5 IQR) for subject mean FD and volume-wise FD respectively. Subject-wise mean FD was averaged over all four resting-state runs.
