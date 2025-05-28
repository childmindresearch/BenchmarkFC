import numpy as np
import pytest
import sklearn
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from arfcexp import skarf_utils


@pytest.mark.parametrize("order,lag", [(1, 1), (2, 0)])
@pytest.mark.parametrize("name", skarf_utils.AVAILABLE_SKARF_FUNCS)
def test_create_skarf_func(name: str, order: int, lag: int):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 10))
    segments = np.repeat(np.arange(4), 10)
    groups = np.repeat(np.arange(2), 20)

    with sklearn.config_context(enable_metadata_routing=True):
        model, needs_groups = skarf_utils.create_skarf_func(
            name, cv=LeaveOneGroupOut(), order=order, lag=lag
        )
        params = {"segments": segments}
        if needs_groups:
            params["groups"] = groups
        model.fit(X, **params)

    if isinstance(model, GridSearchCV):
        var = model.best_estimator_
    else:
        var = model

    assert (var.order, var.lag) == (order, lag)
    assert var.coef_.shape == (order, 10, 10)
