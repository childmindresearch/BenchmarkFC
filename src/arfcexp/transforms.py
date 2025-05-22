from functools import lru_cache
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    MetaEstimatorMixin,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
    scale,
)

from .hcp import load_hcp_behav_factors_topk, load_hcp_behav_columns


HCP_BEHAV_COLUMNS = load_hcp_behav_columns()
HCP_BEHAV_COLUMNS_MAP = {col: ii for ii, col in enumerate(HCP_BEHAV_COLUMNS)}

HCP_COVARIATE_COLUMNS = ["Gender", "Mean_FD", "Age_in_Yrs"]

# Load lazily, bc may not exist.
_cache_load_hcp_behav_factors_topk = lru_cache(load_hcp_behav_factors_topk)


class SampleScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply standard scaling to each sample individually."""

    def fit(self, X: np.ndarray, y: None = None) -> Self:
        validate_data(self, X, ensure_all_finite="allow-nan")
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        X = scale(X, axis=1)
        return X


class ClipOutliers(TransformerMixin, BaseEstimator):
    """Clip outliers based on boxplot rule."""

    threshold_low_: np.ndarray
    threshold_high_: np.ndarray

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: None = None) -> Self:
        validate_data(self, X, ensure_all_finite="allow-nan")
        q1, q3 = np.nanquantile(X, [0.25, 0.75], axis=0)
        iqr = q3 - q1
        threshold_low = q1 - self.threshold * iqr
        threshold_high = q3 + self.threshold * iqr
        self.threshold_low_ = threshold_low
        self.threshold_high_ = threshold_high
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        X = np.clip(X, self.threshold_low_, self.threshold_high_)
        return X


class HCPBehavTargetTransform(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    covariate_transformer_: ColumnTransformer
    covariate_regressor_: LinearRegression
    clip_outliers_: ClipOutliers | None
    scaler_: StandardScaler
    re_scaler_: StandardScaler | None
    quantizer_: KBinsDiscretizer | None

    def __init__(
        self,
        target_name: str | None = None,
        clip: bool = True,
        n_bins: int | None = None,
    ):
        super().__init__()
        self.target_name = target_name
        self.clip = clip
        self.n_bins = n_bins

    def fit_transform(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        """Fit HCP behavioral target transform.

        Args:
            X: HCP behavioral dataframe indexed by subject ID. First three columns are
                the covariates Gender, Mean_FD, Age_in_Yrs. Remaining columns are the
                behavioral measures themselves.
        """
        covariates, y = self._split_covariates_y(X)

        # Subset target columns
        ind, weight = self._get_target_indices_weight()
        if ind is not None:
            y = y[:, ind]

        # Encode covariates
        covariate_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), ["Gender"]),
                ("num", StandardScaler(), ["Mean_FD", "Age_in_Yrs"]),
            ]
        )
        covariates = covariate_transformer.fit_transform(covariates)

        # Fit OLS nuisance regression and residualize.
        covariate_regressor = LinearRegression().fit(covariates, y)
        y_res = y - covariate_regressor.predict(covariates)

        # Clip outliers.
        if self.clip:
            clip_outliers = ClipOutliers().fit(y_res)
            y_res = clip_outliers.transform(y_res)
        else:
            clip_outliers = None

        # Standard scale targets.
        scaler = StandardScaler()
        y_res_scaled = scaler.fit_transform(y_res)

        # Take weighted average of targets and re-scale to have variance 1.
        if weight is not None:
            y_res_scaled = y_res_scaled @ weight[:, None]
            re_scaler = StandardScaler(with_mean=False)
            y_res_scaled = re_scaler.fit_transform(y_res_scaled)
        else:
            re_scaler = None

        # Quantize the the continuous target(s) to a fixed number of quantile bins.
        if self.n_bins:
            quantizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            y_res_scaled = quantizer.fit_transform(y_res_scaled).astype(np.int32)
        else:
            quantizer = None

        # Squeeze to single target.
        if y_res_scaled.shape[1] == 1:
            y_res_scaled = np.squeeze(y_res_scaled, 1)

        self.covariate_transformer_ = covariate_transformer
        self.covariate_regressor_ = covariate_regressor
        self.clip_outliers_ = clip_outliers
        self.scaler_ = scaler
        self.re_scaler_ = re_scaler
        self.quantizer_ = quantizer

        return y_res_scaled

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self.fit_transform(X)
        return self

    def transform(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self)

        # Note, in this context X = targets
        covariates, y = self._split_covariates_y(X)

        # Subset target columns
        ind, weight = self._get_target_indices_weight()
        if ind is not None:
            y = y[:, ind]

        # Nuisance regression using pre-estimated coefficients
        covariates = self.covariate_transformer_.transform(covariates)
        y_res = y - self.covariate_regressor_.predict(covariates)

        # Clip outliers using pre-estimated thresholds
        if self.clip:
            y_res = self.clip_outliers_.transform(y_res)

        # Standard scale each target
        y_res_scaled = self.scaler_.transform(y_res)

        # Weighted average and re-scale to unit variance
        if weight is not None:
            y_res_scaled = y_res_scaled @ weight[:, None]
            y_res_scaled = self.re_scaler_.transform(y_res_scaled)

        # Quantize to discrete quantile bins.
        if self.n_bins:
            y_res_scaled = self.quantizer_.transform(y_res_scaled).astype(np.int32)

        # Squeeze to single target.
        if y_res_scaled.shape[1] == 1:
            y_res_scaled = np.squeeze(y_res_scaled, 1)
        return y_res_scaled

    def _split_covariates_y(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        # Split into covariates and behavioral targets.
        # Keep covariates as a dataframe bc will use names later, but cast targets to ndarray.
        covariates = X.loc[:, HCP_COVARIATE_COLUMNS]
        y = check_array(X.loc[:, HCP_BEHAV_COLUMNS])
        return covariates, y

    def _get_target_indices_weight(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get the column indices and average weights for the selected target."""
        if self.target_name is None:
            ind, weight = None, None
        elif self.target_name in HCP_BEHAV_COLUMNS_MAP:
            ind = np.array([HCP_BEHAV_COLUMNS_MAP[self.target_name]])
            weight = None
        else:
            hcp_factors_topk = _cache_load_hcp_behav_factors_topk()
            assert self.target_name in hcp_factors_topk, (
                f"Invalid target name {self.target_name}"
            )

            weight = np.array(hcp_factors_topk[self.target_name])
            (ind,) = weight.nonzero()
            weight = weight[ind]
        return ind, weight
