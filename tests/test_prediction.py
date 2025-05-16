import numpy as np
from scipy.stats import pearsonr

from arfcexp.prediction import corr_score


def test_corr_score():
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal((100, 5))
    y_pred = rng.standard_normal((100, 5))
    score = corr_score(y_true, y_pred)

    for ii in range(5):
        r_value = pearsonr(y_true[:, ii], y_pred[:, ii]).statistic
        assert np.isclose(score[ii], r_value)
