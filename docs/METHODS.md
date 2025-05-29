## Methods

### Subject filtering

Commands:

- `compute_hcp_1200_rfmri_fd`
- `analyze_hcp_1200_rfmri_fd`
- `filter_hcp_1200_subjects`

Code:

- [compute_hcp_1200_rfmri_fd.py](../scripts/compute_hcp_1200_rfmri_fd.py)
- [analyze_hcp_1200_rfmri_fd.ipynb](../notebooks/analyze_hcp_1200_rfmri_fd.ipynb)
- [filter_hcp_1200_subjects.ipynb](../notebooks/filter_hcp_1200_subjects.ipynb)

Figures:

![hcp_1200_rfmri_fd](../results/hcp_1200_rfmri_fd/hcp_1200_rfmri_fd.png)

> Histograms and boxplots of motion statistics. (left) instantaneous FD per volume, (middle) mean FD per subject, (right) proportion of motion spike volumes per run. Thresholds selected by rounding the boxplot outlier thresholds.

Our initial subject pool included subjects from HCP-1200 with complete behavioral data for the 58 behavioral measures (Li et al., 2019, and others) and all four runs of 3T resting-state data (987 subjects).

We excluded subjects with mean framewise displacement (FD) greater than 0.3, or with one or more runs with at least 50% of volumes exceeding an FD motion spike threshold of 0.4. This resulted in a final sample size of 867 subjects.

FD was computed based on the movement parameters distributed with the preprocessed HCP-1200 data using the (Power et al., 2012) formulation. The two FD thresholds were computed by rounding the boxplot outlier threshold (75% + 1.5 IQR) for subject mean FD and volume-wise FD respectively. Subject-wise mean FD was averaged over all four resting-state runs.


### Time series preprocessing

Commands:

- `visualize_raw_hcp_timeseries`
- `test_synthetic_timeseries_preprocessing`
- `test_hcp_timeseries_preprocessing`
- `test_hcp_timeseries_filtering`
- `extract_hcp_schaefer_timeseries`

Code:

- [extract_hcp_schaefer_timeseries.py](../scripts/extract_hcp_schaefer_timeseries.py)

Figures:

![hcp_frame_stdev_filter_pad_odd_even](../results/test_hcp_timeseries_filtering/hcp_frame_stdev_filter_pad_odd_even.png)

> Framewise intensity standard deviation for time series with different bandpass filtering. Note that odd padding introduces a large edge effect, which is reduced in even padding. (See e.g. [scipy.signal.filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) for description of padding type.)

![hcp_timeseries_preprocessing](../results/test_hcp_timeseries_preprocessing/hcp_timeseries_preprocessing.png)

> HCP time series preprocessed using different strategies. From left to right, (i) raw time series with motion spikes indicated with transparent horizontal bands, (ii) standard scaling (mean zero unit variance) only, (iii) "base" preprocessing following (Li et al., 2019) with `nilearn.signal.clean` defaults, (iv) no motion spike interpolation, (v) bandpass filtering with even instead of odd padding, (vi) adding GSR, final preprocessing strategy.

HCP cifti-space 32k minimally preprocessed time series data were post-processed following prior works, e.g. (Li et al., 2019; Kong et al., 2023). Specific steps were standard scaling to mean zero unit variance, bandpass filtering (0.009 < f < 0.08) with even padding to reduce edge artifacts, global signal regression (GSR) including both the mean signal and temporal derivative, and ROI time series averaging using the Schaefer 2018 parcellation. Unlike previous works, we did not interpolate over high motion frames, since the original HCP preprocessed data are already denoised, and we found that the interpolation introduces new artifacts. Time series post-processing was implemented using `nilearn.signal.clean`.


### PySPI SPI selection
