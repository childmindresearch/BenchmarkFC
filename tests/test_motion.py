import logging

import numpy as np
from scipy import ndimage

from arfcexp.motion import censor_motion_spikes, compute_fd, remove_short_segments


def test_compute_fd():
    rng = np.random.default_rng(42)
    motion_params = 0.1 * rng.normal(size=(100, 6))
    motion_derivs = np.diff(motion_params, prepend=motion_params[:1], axis=0)
    motion_regressors = np.concatenate([motion_params, motion_derivs], axis=1)
    fd_values = compute_fd(motion_regressors)
    mean_fd = np.mean(fd_values[1:])
    logging.info("FD: %.3f", mean_fd)
    assert 0 < mean_fd < 1.0


def test_remove_short_segments():
    rng = np.random.default_rng(42)
    sample_mask = rng.random(size=(100,)) < 0.5
    sample_mask_filt = remove_short_segments(sample_mask, threshold=3)

    segment_label, _ = ndimage.label(sample_mask_filt)
    counts = np.bincount(segment_label)[1:]
    assert counts.min() == 3


def test_censor_motion_spikes():
    rng = np.random.default_rng(42)
    motion_values = rng.random(size=(300,))
    sample_mask = censor_motion_spikes(
        motion_values, threshold=0.9, segment_threshold=5
    )

    # Check that all included volumes have value below threshold.
    assert np.all(motion_values[sample_mask] < 0.9)

    # Check that the shortest segment of censored volumes is 4
    # (one before and two after).
    label, _ = ndimage.label(~sample_mask)
    counts = np.bincount(label)[1:]
    assert counts.min() == 4

    # Check that the shortest segment of included volumes is 5.
    label, _ = ndimage.label(sample_mask)
    counts = np.bincount(label)[1:]
    assert counts.min() == 5
