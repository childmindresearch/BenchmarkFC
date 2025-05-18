import numpy as np

from arfcexp.transforms import ClipOutliers


def test_clip_outliers():
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((100, 10))
    X2 = rng.uniform(-20, 20, (100, 10))
    outlier_mask = rng.random((100, 10)) < 0.05
    X = np.where(outlier_mask, X2, X1)
    nan_mask = rng.random((100, 10)) < 0.05
    X = np.where(nan_mask, np.nan, X)

    X_clip = ClipOutliers().fit_transform(X)
    assert np.all(np.abs(X_clip[~nan_mask]) < 4)
    assert np.all(np.isnan(X_clip) == nan_mask)
