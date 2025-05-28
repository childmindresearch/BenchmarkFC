import numpy as np
from sklearn.covariance import EmpiricalCovariance, GraphicalLassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, Ridge, RidgeCV

from skarf.var import BaseVAR, CovarianceVAR, LinearVAR

AVAILABLE_SKARF_FUNCS = [
    "cov_empirical",
    "cov_graphicallasso",
    "prec_empirical",
    "prec_graphicallasso",
    "linear_ols",
    "linear_ridge",
    "linear_lasso",
    "linear_enet",
    "linear_lasso-pos",
    "linear_enet-pos",
    "linear_pls",
    "linear_pca-ridge",
]


def create_skarf_func(
    name: str, *, cv: BaseCrossValidator, **kwargs
) -> tuple[BaseVAR, bool]:
    needs_groups = True

    match name:
        case "cov_empirical":
            var = CovarianceVAR(
                EmpiricalCovariance(), degree=2, per_target=True, **kwargs
            )
            needs_groups = False
        case "cov_graphicallasso":
            var = CovarianceVAR(GraphicalLassoCV(), degree=2, per_target=True, **kwargs)
            needs_groups = False
        case "prec_empirical":
            var = CovarianceVAR(
                EmpiricalCovariance(),
                degree=2,
                per_target=True,
                use_precision=True,
                **kwargs,
            )
            needs_groups = False
        case "prec_graphicallasso":
            var = CovarianceVAR(
                GraphicalLassoCV(),
                degree=2,
                per_target=True,
                use_precision=True,
                **kwargs,
            )
            needs_groups = False
        case "linear_ols":
            var = LinearVAR(LinearRegression(), per_target=True, **kwargs)
            needs_groups = False
        case "linear_ridge":
            var = LinearVAR(
                RidgeCV(alphas=np.logspace(-1, 3, 10), cv=cv), per_target=True, **kwargs
            )
        case "linear_lasso":
            var = LinearVAR(
                LassoCV(eps=0.01, n_alphas=10, cv=cv), per_target=True, **kwargs
            )
        case "linear_enet":
            var = LinearVAR(
                ElasticNetCV(eps=0.01, n_alphas=10, cv=cv), per_target=True, **kwargs
            )
        case "linear_lasso-pos":
            var = LinearVAR(
                LassoCV(eps=0.01, n_alphas=10, positive=True, cv=cv),
                per_target=True,
                **kwargs,
            )
        case "linear_enet-pos":
            var = LinearVAR(
                ElasticNetCV(eps=0.01, n_alphas=10, positive=True, cv=cv),
                per_target=True,
                **kwargs,
            )
        case "linear_pls":
            var = GridSearchCV(
                _with_routing(LinearVAR(PLSRegression(), per_target=False, **kwargs)),
                param_grid={"estimator__n_components": [8, 16, 32]},
                cv=cv,
            )
        case "linear_pca-ridge":
            var = GridSearchCV(
                _with_routing(
                    LinearVAR(Ridge(), per_target=True, decomposition=PCA(), **kwargs)
                ),
                param_grid={
                    "decomposition__n_components": [8, 16, 32],
                    "estimator__alpha": [0.1, 1, 10, 100, 1000],
                },
                cv=cv,
            )
        case _:
            raise NotImplementedError(f"Skarf func {name} not implemented.")
    return var, needs_groups


def _with_routing(var: LinearVAR) -> LinearVAR:
    return (
        var.set_fit_request(segments=True)
        .set_fit_request(groups=False)
        .set_score_request(segments=True)
    )
