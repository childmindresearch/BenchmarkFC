# TODO

## Autoregressive prediction evaluation

The skarf estimators are optimized for autoregressive time series prediction. Which estimator predicts the time series the best? Is time series prediction correlated with behavioral trait prediction.

The autoregressive prediction scores are already computed as part of skarf estimation. There are three scores: "train" (within-run), "val" (within-session), "test" (across-session). For consistent comparison with the behavioral prediction analyses, we can compute average scores for 20 random test splits of subjects, and plot similarly. (See [`eval_hcp_1200_skarf_behav_prediction.py`](../scripts/eval_hcp_1200_skarf_behav_prediction.py) for how to generate the splits.)

## Test-retest reliability

Evaluate the test-retest reliability of the PySPI and Skarf matrices. Possible test-retest reliability measures we could use are:

- intra-class correlation (ICC) ([`icc.py`](../src/arfcexp/icc.py)).
- subject identifiability index (Tian2021).

Again, for consistent evaluation, average reliability scores can be computed over 20 random test splits of subjects.

## Graph metrics

Skarf matrices can be naturally viewed as estimates of the underlying communication graph. Especially when using sparsity inducing regularization and non-negativity constraints. In earlier preliminary analyses, we compared the structure of functional connectivity graphs estimated using skarf vs Pearson correlation. We found that skarf matrices tend to be (1) less reliable edge wise (across session), (2) higher "entropy", (3) similarly reliable (across session) in terms of higher-level graph structure (clustering and principal gradients).

There is a set of graph metrics implemented in [`arfcexp.graph_metrics.py`](../src/arfcexp/graph_metrics.py) (edge similarity, gradient similarity, cluster similarity, spectral entropy). Using the PySPI and skarf matrices computed already, compare their graph structure using these metrics.

## Group-level skarf estimates

So far we have estimated skarf matrices for individual subjects and runs only. However, we can also estimate group-level matrices, by jointly fitting a skarf matrix to a matrix of all subject/run time series concatenated along the time dimension. This could be an interesting way to jointly estimate the group graph structure (though see also related work)
