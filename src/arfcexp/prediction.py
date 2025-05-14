from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    Normalizer,
    OneHotEncoder,
    StandardScaler,
    normalize,
    scale,
)
from sklearn.pipeline import Pipeline


class SampleScaler(Normalizer):
    """Apply standard scaling to each sample individually."""

    def __init__(self):
        super().__init__()

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        return scale(X, axis=1)


class HCPPhenoTargetTransform(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    scaler_: StandardScaler
    covariate_transformer_: ColumnTransformer
    covariate_regressor_: LinearRegression

    def __init__(self, clip_threshold: float | None = 3.0):
        super().__init__()
        self.clip_threshold = clip_threshold

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        y = check_array(y, ensure_2d=False)
        if y.ndim == 1:
            y = y[:, None]

        # Standard scale targets.
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)
        if self.clip_threshold is not None:
            y_scaled = np.clip(y_scaled, -self.clip_threshold, self.clip_threshold)

        # Extract covariates and transform before regression.
        covariates = X.loc[:, ["Gender", "Mean_FD", "Age_in_Yrs"]]
        covariate_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), ["Gender"]),
                ("num", StandardScaler(), ["Mean_FD", "Age_in_Yrs"]),
            ]
        )
        covariates = covariate_transformer.fit_transform(covariates)

        # Fit OLS nuisance regression.
        covariate_regressor = LinearRegression().fit(covariates, y_scaled)

        self.scaler_ = scaler
        self.covariate_transformer_ = covariate_transformer
        self.covariate_regressor_ = covariate_regressor
        return self

    def transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        y = check_array(y, ensure_2d=False)
        is_1d = y.ndim == 1
        if is_1d:
            y = y[:, None]

        y_scaled = self.scaler_.transform(y)
        if self.clip_threshold is not None:
            y_scaled = np.clip(y_scaled, -self.clip_threshold, self.clip_threshold)

        covariates = X.loc[:, ["Gender", "Mean_FD", "Age_in_Yrs"]]
        covariates = self.covariate_transformer_.transform(covariates)

        y_scaled_res = y_scaled - self.covariate_regressor_.predict(covariates)

        # Clip again in case nuisance regression predictions are bad.
        if self.clip_threshold is not None:
            y_scaled_res = np.clip(
                y_scaled_res, -self.clip_threshold, self.clip_threshold
            )

        if is_1d:
            y_scaled_res = np.squeeze(y_scaled_res, axis=1)
        return y_scaled_res

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X, y)


class HCPPhenoRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    feature_transform_: Pipeline
    target_transform_: HCPPhenoTargetTransform
    regressor_: LinearRegression

    def __init__(
        self,
        regressor: LinearRegression,
        feature_name: str = "Features",
    ):
        super().__init__()
        self.regressor = regressor
        self.feature_name = feature_name

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_params) -> Self:
        # Get features, shape (N, D)
        features = X.loc[:, self.feature_name].values
        features: np.ndarray = np.stack(features)
        features = features.reshape(len(features), -1)

        # Each feature vector scaled (sample wise) to be mean zero stdev 1.
        # Then NaNs filled with zeros.
        # Note, sample-wise scaling combined with kernel ridge with cosine kernel
        # replicates the typicall KRR procedure from the Yeo Lab.
        feature_transform = Pipeline(
            [("scale", SampleScaler()), ("impute", SimpleImputer(strategy="constant"))]
        )
        features = feature_transform.fit_transform(features)

        target_transform = HCPPhenoTargetTransform()
        targets = target_transform.fit_transform(X, y)

        regressor: LinearRegression = clone(self.regressor)
        regressor.fit(features, targets, **fit_params)

        self.feature_transform_ = feature_transform
        self.target_transform_ = target_transform
        self.regressor_ = regressor
        return self

    def predict(self, X: pd.DataFrame, y: None = None) -> np.ndarray:
        check_is_fitted(self)

        features = X.loc[:, self.feature_name].values
        features: np.ndarray = np.stack(features)
        features = features.reshape(len(features), -1)

        features = self.feature_transform_.transform(features)
        preds = self.regressor_.predict(features)
        return preds

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        # Nb, there is another strategy where predictions are inverse transformed before
        # scoring. But the current approach is more consistent with prior works, e.g.
        # Kong 2023.
        # https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
        preds = self.predict(X)
        targets = self.target_transform_.transform(X, y)
        return np.mean(corr_score(targets, preds))


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = normalize(y_true - y_true.mean(axis=0), axis=0)
    y_pred = normalize(y_pred - y_pred.mean(axis=0), axis=0)
    corr = np.sum(y_true * y_pred, axis=0)
    return corr
