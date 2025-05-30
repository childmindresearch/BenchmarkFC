from typing import NamedTuple

import numpy as np


class ICCResult(NamedTuple):
    icc1: np.ndarray
    icc2: np.ndarray
    icc3: np.ndarray
    icc1k: np.ndarray
    icc2k: np.ndarray
    icc3k: np.ndarray


@np.errstate(divide="ignore")
def batch_icc(data: np.ndarray) -> ICCResult:
    """
    Compute batch intraclass correlation coefficients.

    Args:
        x: array of repeated measuresments tables, shape (..., n_subs, n_ratings)

    Returns:
        Named tuple of batch ICC results. NaN values can result when the number of
        subjects/ratings is < 2.

    Example:
        >>> import numpy as np
        >>> # example wine data copied from pingouin
        >>> data = np.array(
        ...     [[1, 2, 0, 1],
        ...     [1, 3, 3, 2],
        ...     [3, 8, 1, 4],
        ...     [6, 4, 3, 3],
        ...     [6, 5, 5, 6],
        ...     [7, 5, 6, 2],
        ...     [8, 7, 7, 9],
        ...     [9, 9, 9, 8]]
        ... )
        >>> icc = batch_icc(data)
        >>> print(icc.icc2k)
        0.9144500359453631

    References:
        https://github.com/raphaelvallat/pingouin/blob/b563de7f4c7ea35467d40a675d02a0018c67dc24/src/pingouin/reliability.py#L159
        https://github.com/cran/psych/blob/724ef67cfd88aa5e8cf17f297cbf94e0c72f874d/R/ICC.R#L4
        https://github.com/TingsterX/ReX/blob/main/R/lme_ICC.R
        Koo & Li, 2016
        McGraw & Wong, 1996
    """
    assert data.ndim >= 2, "data should be at least 2d"

    # subject (row) and ratings (column) axes
    II, JJ = -2, -1
    _, KK = data.shape[II], data.shape[JJ]

    # handle missing values with list-wise deletion
    # missing values coded as zeros
    mask = ~np.isnan(data)
    row_mask = np.all(mask, axis=JJ, keepdims=True)
    mask = np.repeat(row_mask, axis=JJ, repeats=KK)
    col_mask = np.any(mask, axis=II, keepdims=True)
    data = np.where(mask, data, 0.0)

    # number of effective rows/cols per table might vary depending on missing values
    # results will be nan if n_row or n_col < 2
    n_row = np.sum(row_mask, axis=II, keepdims=True)
    n_col = np.sum(col_mask, axis=JJ, keepdims=True)
    n_total = np.sum(mask, axis=(II, JJ), keepdims=True)

    grand_mean = np.sum(data, axis=(II, JJ), keepdims=True) / n_total
    row_mean = np.sum(data, axis=JJ, keepdims=True) / n_col
    col_mean = np.sum(data, axis=II, keepdims=True) / n_row

    # simple sum and mean of squares, no mixed effects model required
    ss_row = n_col * np.sum((row_mean - grand_mean) ** 2, axis=II, keepdims=True)
    ss_col = n_row * np.sum((col_mean - grand_mean) ** 2, axis=JJ, keepdims=True)
    ss_tot = np.sum((data - grand_mean) ** 2, axis=(II, JJ), keepdims=True)
    ss_err = ss_tot - (ss_row + ss_col)

    # max with 0 in case < 2
    df_row = np.maximum(n_row - 1, 0)
    df_col = np.maximum(n_col - 1, 0)
    df_err = df_row * df_col

    msr = ss_row / df_row
    msc = ss_col / df_col
    mse = ss_err / df_err
    msw = (ss_col + ss_err) / (df_col + df_err)

    # ICC formulas from pingouin intraclass_corr following notation of Koo & Li, 2016
    icc1 = squeeze((msr - msw) / (msr + df_col * msw))
    icc2 = squeeze((msr - mse) / (msr + df_col * mse + n_col * (msc - mse) / n_row))
    icc3 = squeeze((msr - mse) / (msr + df_col * mse))
    icc1k = squeeze((msr - msw) / msr)
    icc2k = squeeze((msr - mse) / (msr + (msc - mse) / n_row))
    icc3k = squeeze((msr - mse) / msr)
    return ICCResult(icc1, icc2, icc3, icc1k, icc2k, icc3k)


def squeeze(x: np.ndarray) -> float | np.ndarray:
    x = np.squeeze(x)
    if x.ndim == 0:
        x = x.item()
    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
