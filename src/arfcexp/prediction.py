from copy import deepcopy
from typing import Self

import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    check_is_fitted,
)
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils._tags import get_tags

EPS = np.finfo(np.float32).eps


class BaseTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray: ...


class TargetTransformEstimator(MetaEstimatorMixin, BaseEstimator):
    """Similar to `TransformedTargetRegressor`, but scoring on transformed targets."""

    estimator_: Pipeline
    target_transform_: BaseTransformer

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.estimator_type = sub_estimator_tags.estimator_type
        tags.classifier_tags = deepcopy(sub_estimator_tags.classifier_tags)
        tags.regressor_tags = deepcopy(sub_estimator_tags.regressor_tags)
        # allows cross-validation to see 'precomputed' metrics
        tags.input_tags.pairwise = sub_estimator_tags.input_tags.pairwise
        tags.input_tags.sparse = sub_estimator_tags.input_tags.sparse
        tags.array_api_support = sub_estimator_tags.array_api_support
        return tags

    def __init__(
        self,
        estimator: Pipeline,
        target_transform: BaseTransformer,
        scoring: str | None = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.target_transform = target_transform
        self.scoring = scoring

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> Self:
        # Note, not fully generic since I'm not doing any param routing, but assuming
        # all params go to estimator fit. Also not doing data validation.

        target_transform: BaseTransformer = clone(self.target_transform)
        y = target_transform.fit_transform(y)

        estimator: Pipeline = clone(self.estimator)
        estimator.fit(X, y, **fit_params)

        self.target_transform_ = target_transform
        self.estimator_ = estimator
        return self

    def predict(self, X: np.ndarray, y: None = None) -> np.ndarray:
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Nb, there is another strategy where predictions are inverse transformed.
        # https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
        # But the current approach is more consistent with prior works, e.g.  Kong 2023.
        check_is_fitted(self)
        y = self.target_transform_.transform(y)
        if self.scoring is None:
            score = self.estimator_.score(X, y)
        else:
            scorer = get_scorer(self.scoring)
            score = scorer(self.estimator_, X, y)
        return score


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true - y_true.mean(axis=0)
    y_true = y_true / np.maximum(np.linalg.norm(y_true, axis=0), EPS)

    y_pred = y_pred - y_pred.mean(axis=0)
    y_pred = y_pred / np.maximum(np.linalg.norm(y_pred, axis=0), EPS)

    corr = np.sum(y_true * y_pred, axis=0)
    return corr
