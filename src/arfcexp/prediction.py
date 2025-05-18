from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    MetaEstimatorMixin,
    clone,
    check_is_fitted,
)
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .hcp import load_hcp_behav_factors_topk, load_hcp_behav_columns
from .transforms import HCPBehavTargetTransform, SampleScaler

HCP_BEHAV_FACTORS_TOPK = load_hcp_behav_factors_topk()
HCP_BEHAV_COLUMNS = load_hcp_behav_columns()
EPS = np.finfo(np.float32).eps


class HCPBehavRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    """HCP behavioral prediction model, following Yeo lab.

    `regressor` can be any linear regression model, but `KernelRidge` is typical to
    follow previous works.

    `feature_name` is the name of the feature column in the input `X` dataframe. Other
    expected columns are covariates: `Gender`, `Age_in_Yrs`, `Mean_FD`.

    The targets `y` should be an array or dataframe of behavioral targets. The columns
    **must be** the Yeo lab 58 behavioral columns, in order.

    `target_name` is the name of the specific behavioral target to predict. Can be
    `None` to predict all jointly, any of the 58 columns, or one of the 4 factors:
    Dissatisfaction, Cognition, Support, or Emotion.

    The steps of the model fit are:
        - Extract the features array from X
        - Standard scale **each sample** independently to mean 0, stdev 1. (When
          combined with KRR and cosine kernel, this replicates the pearson correlation
          kernel used in Yeo lab works.)
        - Impute NaN with zero.
        - Preprocess targets (nuisance regression, clip outliers, standard scale).
        - Extract targets (top-k average for factor targets).
        - Fit regression model.
    """

    feature_transform_: Pipeline
    target_transform_: HCPBehavTargetTransform
    target_scaler_: StandardScaler
    regressor_: LinearRegression

    def __init__(
        self,
        regressor: LinearRegression,
        feature_name: str = "Matrix",
        target_name: str | None = None,
    ):
        super().__init__()
        self.regressor = regressor
        self.feature_name = feature_name
        self.target_name = target_name

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **fit_params) -> Self:
        # Each feature vector scaled (sample wise) to be mean zero stdev 1.
        # Then NaNs filled with zeros.
        # Note, sample-wise scaling combined with kernel ridge with cosine kernel
        # replicates the typicall KRR procedure from the Yeo Lab.
        feature_transform = Pipeline(
            [
                ("scale", SampleScaler()),
                (
                    "impute",
                    SimpleImputer(strategy="constant", keep_empty_features=True),
                ),
            ]
        )
        features = self._get_features(X)
        features = feature_transform.fit_transform(features)

        target_transform = HCPBehavTargetTransform()
        # Note, we transform y first before pulling out the target because some targets
        # are top-k averages for the behavioral factors.
        targets = target_transform.fit_transform(X, y)
        targets = self._get_targets(targets)

        # Rescale the factor average targets
        if self.target_name in HCP_BEHAV_FACTORS_TOPK:
            target_scaler = StandardScaler(with_mean=False)
            targets = target_scaler.fit_transform(targets[:, None]).squeeze(1)

        regressor: LinearRegression = clone(self.regressor)
        regressor.fit(features, targets, **fit_params)

        self.feature_transform_ = feature_transform
        self.target_transform_ = target_transform
        if self.target_name in HCP_BEHAV_FACTORS_TOPK:
            self.target_scaler_ = target_scaler
        self.regressor_ = regressor
        return self

    def predict(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self)
        features = self._get_features(X)
        features = self.feature_transform_.transform(features)
        preds = self.regressor_.predict(features)
        return preds

    def transform_targets(self, X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        targets = self.target_transform_.transform(X, y)
        targets = self._get_targets(targets)
        if self.target_name in HCP_BEHAV_FACTORS_TOPK:
            targets = self.target_scaler_.transform(targets[:, None]).squeeze(1)
        return targets

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        # Nb, there is another strategy where predictions are inverse transformed.
        # https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
        # But the current approach is more consistent with prior works, e.g.  Kong 2023.
        check_is_fitted(self)
        preds = self.predict(X)
        targets = self.transform_targets(X, y)
        score = -mean_squared_error(targets, preds)
        return score

    def _get_features(self, X: pd.DataFrame) -> np.ndarray:
        # Get features, shape (N, D)
        features = X.loc[:, self.feature_name].values
        features: np.ndarray = np.stack(features)
        features = features.reshape(len(features), -1)

        # Cast to double to guard against any numerical issues, but also mostly just to
        # stop the warning from `scale`.
        features = features.astype(np.float64)
        return features

    def _get_targets(self, y: np.ndarray) -> np.ndarray:
        if self.target_name is None:
            targets = y
        elif self.target_name in HCP_BEHAV_FACTORS_TOPK:
            weight = np.asarray(HCP_BEHAV_FACTORS_TOPK[self.target_name])
            targets = y @ weight
        else:
            idx = HCP_BEHAV_COLUMNS.index(self.target_name)
            targets = y[:, idx]
        return targets


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true - y_true.mean(axis=0)
    y_true = y_true / np.maximum(np.linalg.norm(y_true, axis=0), EPS)

    y_pred = y_pred - y_pred.mean(axis=0)
    y_pred = y_pred / np.maximum(np.linalg.norm(y_pred, axis=0), EPS)

    corr = np.sum(y_true * y_pred, axis=0)
    return corr
