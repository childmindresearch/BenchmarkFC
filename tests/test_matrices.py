from pathlib import Path

import numpy as np
import pandas as pd

from arfcexp import matrices


def test_load_avg_mats_and_impose_sparsity_can_skip_threshold(monkeypatch):
    expected = pd.DataFrame(
        {
            "Count": [1, 0],
            "Matrix": [np.array([1.0, 2.0, 3.0, 4.0]), None],
        },
        index=["sub-1", "sub-2"],
    )
    calls = {}

    def fake_load_avg_mats_from_parquet(
        parquet_path,
        method,
        func,
        sub_list,
        lag=0,
        fill_missing=True,
        reader=None,
    ):
        calls["fill_missing"] = fill_missing
        return expected.copy(deep=True)

    monkeypatch.setattr(matrices, "load_avg_mats_from_parquet", fake_load_avg_mats_from_parquet)

    result = matrices.load_avg_mats_and_impose_sparsity(
        Path("/tmp/fake.parquet"),
        "pyspi",
        "cov_EmpiricalCovariance",
        ["sub-1", "sub-2"],
        sparsity=None,
        fill_missing=False,
    )

    assert calls["fill_missing"] is False
    assert np.array_equal(result.loc["sub-1", "Matrix"], expected.loc["sub-1", "Matrix"])
    assert result.loc["sub-2", "Matrix"] is None