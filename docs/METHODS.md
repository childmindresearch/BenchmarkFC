## Methods

### Skarf

Vector autoregressive (VAR) models are common in the fMRI literature (e.g. Goebel2003, Harrison2003, Valdes-Sosa2004, Roebroeck2005, Chang2010, Rogers2010, Zalesky2014, Liegeois2017). Various regularized VAR models have also been explored (Valdes-Sosa2005, Eavani2015, Li2015). However, these methods have not gained widespread adoption. We implement a convenient package to explore regularized VAR models for modeling fMRI time series dynamics.

A linear vector autoregressive (VAR) model of order $p$ (denoted VAR($p$)) predicts the current value of a $d$-dimensional time series $\mathbf{x}_{t} \in \mathbb{R}^d$ as a linear combination of its $p$ most recent past values. Specifically, the model assumes

$$
\mathbf{x}_{t} = \sum_{i=1}^{p} \mathbf{A}_i \mathbf{x}_{t-i} + \boldsymbol{\varepsilon}_t,
$$

where each $\mathbf{A}_i \in \mathbb{R}^{d \times d}$ is a coefficient matrix for lag $i$, and $\boldsymbol{\varepsilon}_t$ is a zero-mean noise term.

To prevent overfitting and improve generalization, we incorporated regularization into the estimation procedure. The optimization problem takes the form

$$
\min_{\{\mathbf{A}_i\}} \sum_t \left\| \mathbf{x}_{t} - \sum_{i=1}^{p} \mathbf{A}_i \mathbf{x}_{t-i} \right\|_2^2 + \sum_i \mathcal{R}(\mathbf{A}_i),
$$

where $\mathcal{R}$ is a regularization term. We explored several choices for $\mathcal{R}$, including the Frobenius norm penalty (ridge regularization), the elementwise $\ell_1$-norm (lasso), a convex combination of both (elastic net). We also consider regularization in the form of a low-rank constraint, implemented by factorizing the $\mathbf{A}_i$ as the product of two low-rank matrices (partial least squares, principal component ridge regression).

We also consider a special case of time-series "self-representation" or "co-smoothing", where the model predicts the current time step given the current vector of activity.

$$
\min_{\mathbf{A}} \sum_t \left\| \mathbf{x}_{t} - \mathbf{A}\mathbf{x}_t \right\|_2^2 + \mathcal{R}(\mathbf{A}),
$$

To prevent a degenerate identity prediction, we constrain $\text{diag}(\mathbf{A}) = 0$ (`per_target=True` option in the package API).

To tune the hyperparameters, we split the training time series into contiguous segments and apply leave-one-segment-out cross-validation.

We leverage the `scikit-learn` library of linear model estimators to implement these models (`sklearn.linear_model`). This approach takes advantage of the fact that VAR models are simply linear regression models with temporally shifted targets (and a sliding window design matrix in case $p > 1$). (But see also [`sktime.forecasting.VARReduce`](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.var_reduce.VARReduce.html).)

### PySPI

We compare the skarf functional connectivity estimators with a large set of time series statistics of pairwise interaction (SPIs) implemented in the [PySPI package](https://github.com/DynamicsAndNeuralSystems/pyspi). This library collects individual SPI estimators from other packages, and unifies them in a common API for convenient evaluation.

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

Commands:

- `export_spi_configs`
- `test_profile_pyspi`
- `analyze_test_profile_pyspi`

The initial pool of PySPI SPIs included all 284 SPIs available in PySPI v1.1.1. We first excluded all squared SPI variants, since these are redundant with other SPIs. We then excluded all SPIs that took longer than 5 minutes to run on a sample subject's data. The 5 minute threshold was selected by applying an inflexion point (i.e. knee point) cutoff to the sorted SPI run times.
