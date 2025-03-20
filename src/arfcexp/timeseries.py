import warnings

import nilearn.signal
import numpy as np
from scipy.interpolate import CubicSpline
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
    num_rois = len(parc_ids) - 1
    assert np.all(parc_ids == np.arange(num_rois + 1))

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
    # nb this will produce NaNs if a roi has no valid data
    parc_one_hot = parc_one_hot / parc_one_hot.sum(axis=0)

    # per roi averaging
    parc_series = series @ parc_one_hot
    return parc_series


def preprocess_timeseries(
    series: np.ndarray,
    tr: float,
    gsr: bool = True,
    sample_mask: np.ndarray | None = None,
    low_pass: float = 0.08,
    high_pass: float = 0.009,
    pad: int | None = 4,
) -> np.ndarray:
    """
    Preprocessing following Li et al., 2019; Kong et al., 2023.

    - Interpolate censored frames (cubic spline)
    - GSR
    - band-pass filter 0.009 < f < 0.08
    """
    if sample_mask is not None:
        series = interpolate_missing_frames(series, sample_mask)

    series, _ = fill_na(series)

    confounds = np.nanmean(series, axis=1) if gsr else None

    if pad:
        pad_mask = np.ones(len(series), dtype=bool)
        series = np.pad(series, [(pad, pad), (0, 0)])
        pad_mask = np.pad(pad_mask, pad)
        if confounds is not None:
            confounds = np.pad(confounds, pad)
    else:
        pad_mask = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        clean_series = nilearn.signal.clean(
            series,
            standardize="zscore_sample",
            confounds=confounds,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            # Treat padded frames as missing, extrapolate and then trim.
            sample_mask=pad_mask,
            extrapolate=True,
        )
    return clean_series


def fill_na(series: np.ndarray):
    """Fill nan/inf values in time series with column means."""
    bad_mask = np.isnan(series) | np.isinf(series)
    column_mean = np.nanmean(series, axis=0)
    column_mean = np.where(np.isnan(column_mean), np.nanmean(series), column_mean)
    series = np.where(bad_mask, column_mean, series)
    return series, bad_mask


def interpolate_missing_frames(
    series: np.ndarray, sample_mask: np.ndarray
) -> np.ndarray:
    """Interpolate missing frames in time series using cubic spline.

    Args:
        series: time series, (n_samples, n_features)
        sample_mask: mask of included samples (True = include), shape (n_samples).

    Returns:
        interp_series: interpolated series, shape (n_samples, n_features)
    """
    x = np.arange(len(series))
    sample_mask = sample_mask > 0
    interp = CubicSpline(x[sample_mask], series[sample_mask], axis=0)
    interp_series = series.copy()
    interp_series[~sample_mask] = interp(x[~sample_mask])
    return interp_series
