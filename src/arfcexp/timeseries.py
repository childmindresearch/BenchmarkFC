import warnings
from typing import Literal

import nilearn.signal
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.sparse import csr_array

EPS = np.finfo(np.float32).eps


def parc_one_hot_encode(parc: np.ndarray, sparse: bool = False) -> np.ndarray:
    """Get one hot encoding of the parcellation.

    Args:
        parc: parcellation, shape (dim,), with values in 0, ..., num_rois, where 0 is
            background.

    Returns:
        parc_one_hot: one hot encoding of parcellation, shape (num_rois, dim)
    """
    assert parc.ndim == 1
    parc = parc.astype(np.int32)

    # assuming rois = 0, ..., n with 0 = background
    parc_ids = np.unique(parc)
    assert parc_ids[0] == 0
    num_rois = parc_ids[-1]
    if not len(parc_ids) == num_rois + 1:
        warnings.warn(
            f"Parcellation max index is {num_rois}, "
            f"but only {len(parc_ids) - 1} unique nonzero IDs found.",
            RuntimeWarning,
        )

    # one hot parcellation matrix, shape (num_rois, dim)
    if sparse:
        mask = parc > 0
        (col_ind,) = mask.nonzero()
        row_ind = parc[mask] - 1
        values = np.ones(len(row_ind), dtype=bool)
        parc_one_hot = csr_array(
            (values, (row_ind, col_ind)), shape=(num_rois, len(parc))
        )
    else:
        parc_one_hot = parc_ids[1:, None] == parc
    return parc_one_hot


def extract_timeseries(series: np.ndarray, parc_one_hot: np.ndarray) -> np.ndarray:
    """Extract parcellated time series.

    Args:
        series: full time series (num_samples, dim)
        parc_one_hot: one hot encoding of parcellation (num_rois, dim)
        data_mask: valid data mask (dim,)

    Returns:
        parc_series: parcellated time series (num_samples, num_rois)
    """
    parc_one_hot = parc_one_hot.astype(series.dtype)

    # don't include verts with missing data
    data_mask = np.var(series, axis=0) > EPS
    parc_one_hot = parc_one_hot * data_mask

    # transpose makes working with dense/sparse work
    parc_one_hot = parc_one_hot.T

    # normalize weights to sum to 1
    # Nb, empty parcels will be all zero
    parc_counts = np.asarray(parc_one_hot.sum(axis=0))
    parc_one_hot = parc_one_hot / np.maximum(parc_counts, 1)

    # per roi averaging
    parc_series = series @ parc_one_hot
    return parc_series


def preprocess_timeseries(
    series: np.ndarray,
    tr: float,
    sample_mask: np.ndarray | None = None,
    interp: Literal["spline", "pchip"] | None = None,
    gsr: bool = True,
    low_pass: float = 0.08,
    high_pass: float = 0.009,
    padtype: Literal["even", "odd"] = "even",
) -> np.ndarray:
    """
    Preprocessing following Li et al., 2019; Kong et al., 2023.

    - GSR
    - band-pass filter 0.009 < f < 0.08

    Note, not interpolating over censored time points because it sometimes fails and
    introduces artifacts. Better to have real data, even if noisy, than fake data with
    artifacts.

    Note, even padtype used, rather than default odd padtype. Odd padding introduces bad
    boundary effects due to (I guess) shiting the time series mean.
    """
    series, bad_mask = fill_na(series)

    valid_mask = np.std(series, axis=0) > np.finfo(np.float16).eps
    valid_series = series[:, valid_mask]

    if interp and sample_mask is not None:
        valid_series = interpolate_missing_frames(
            valid_series, sample_mask, interp=interp
        )

    if gsr:
        # regress global signal and its temporal derivative, following Li 2019.
        global_signal = np.mean(valid_series, axis=1)
        global_signal_derivative = np.diff(global_signal, prepend=global_signal[0])
        confounds = np.stack([global_signal, global_signal_derivative], axis=1)
    else:
        confounds = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        valid_clean_series = nilearn.signal.clean(
            valid_series,
            standardize="zscore_sample",
            confounds=confounds,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            butterworth__padtype=padtype,
        )

    clean_series = np.zeros(series.shape, dtype=np.float32)
    clean_series[:, valid_mask] = valid_clean_series
    clean_series = np.where(bad_mask, np.nan, clean_series)
    return clean_series


def fill_na(series: np.ndarray):
    """Fill nan/inf values in time series with column means."""
    bad_mask = np.isnan(series) | np.isinf(series)
    column_mean = np.nanmean(series, axis=0)
    column_mean = np.where(np.isnan(column_mean), np.nanmean(series), column_mean)
    series = np.where(bad_mask, column_mean, series)
    return series, bad_mask


def interpolate_missing_frames(
    series: np.ndarray,
    sample_mask: np.ndarray,
    interp: Literal["spline", "pchip"] = "pchip",
) -> np.ndarray:
    """Interpolate missing frames in time series using cubic spline or pchip.

    Args:
        series: time series, (n_samples, n_features)
        sample_mask: mask of included samples (True = include), shape (n_samples).
        interp: interpolation method

    Returns:
        interp_series: interpolated series, shape (n_samples, n_features)
    """
    x = np.arange(len(series))
    sample_mask = sample_mask > 0
    if interp == "spline":
        interpolator = CubicSpline(x[sample_mask], series[sample_mask], axis=0)
    else:
        interpolator = PchipInterpolator(x[sample_mask], series[sample_mask], axis=0)
    interp_series = series.copy()
    interp_series[~sample_mask] = interpolator(x[~sample_mask])
    return interp_series
