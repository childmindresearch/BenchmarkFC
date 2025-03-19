import logging

import numpy as np

from arfcexp.compute_fd import compute_hcp_mean_fd


def test_compute_hcp_mean_fd():
    rng = np.random.default_rng(42)
    motion_params = 0.1 * rng.normal(size=(100, 6))
    motion_derivs = np.diff(motion_params, prepend=motion_params[:1], axis=0)
    motion_regressors = np.concatenate([motion_params, motion_derivs], axis=1)
    fd = compute_hcp_mean_fd(motion_regressors)
    logging.info("FD: %.3f", fd)
    assert 0 < fd < 1.0
