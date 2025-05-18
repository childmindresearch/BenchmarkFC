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
    OneHotEncoder,
    StandardScaler,
    scale,
)


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
    clip_outliers_: ClipOutliers
    scaler_: StandardScaler

    def __init__(self, clip: bool = True):
        super().__init__()
        self.clip = clip

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        y = check_array(y, ensure_2d=False)
        if y.ndim == 1:
            y = y[:, None]

        # Extract covariates and transform before regression.
        covariates = X.loc[:, ["Gender", "Mean_FD", "Age_in_Yrs"]]
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

        # Standard scale targets (after removing variance due to nuisance factors).
        scaler = StandardScaler().fit(y_res)

        self.covariate_transformer_ = covariate_transformer
        self.covariate_regressor_ = covariate_regressor
        if self.clip:
            self.clip_outliers_ = clip_outliers
        self.scaler_ = scaler
        return self

    def transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        y = check_array(y, ensure_2d=False)
        is_1d = y.ndim == 1
        if is_1d:
            y = y[:, None]

        covariates = X.loc[:, ["Gender", "Mean_FD", "Age_in_Yrs"]]
        covariates = self.covariate_transformer_.transform(covariates)
        y_res = y - self.covariate_regressor_.predict(covariates)

        if self.clip:
            y_res = self.clip_outliers_.transform(y_res)

        y_res_scaled = self.scaler_.transform(y_res)

        if is_1d:
            y_res_scaled = np.squeeze(y_res_scaled, axis=1)
        return y_res_scaled

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X, y)
