import numpy as np
import pandas as pd

from arfcexp.hcp import load_hcp_gender, load_hcp_age


def load_demographics_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demographics",
        "scripts/eval_hcp_1200_demographics_prediction.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_hcp_gender():
    gender = load_hcp_gender()
    assert isinstance(gender, pd.Series)
    assert gender.name == "Gender"
    # Should only contain 0 (M) and 1 (F).
    assert set(gender.dropna().unique()) == {0, 1}
    assert len(gender) > 0


def test_load_hcp_age():
    age = load_hcp_age()
    assert isinstance(age, pd.Series)
    assert age.name == "Age_in_Yrs"
    # HCP ages are in the range ~22-37.
    assert age.min() >= 20
    assert age.max() <= 40
    assert len(age) > 0


def test_gender_accuracy():
    mod = load_demographics_module()

    compute_gender_accuracy = mod.compute_gender_accuracy
    find_optimal_threshold = mod.find_optimal_threshold

    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.2, 0.3, 0.6, 0.7, 0.8])

    acc = compute_gender_accuracy(y_true, y_pred, threshold=0.5)
    assert acc == 1.0

    # Wrong predictions.
    y_pred_bad = np.array([0.6, 0.7, 0.2, 0.3, 0.8])
    acc_bad = compute_gender_accuracy(y_true, y_pred_bad, threshold=0.5)
    assert acc_bad == 1 / 5

    # Optimal threshold should find a good split.
    thresh, acc_opt = find_optimal_threshold(y_true, y_pred)
    assert acc_opt == 1.0
    assert 0.3 < thresh < 0.6


def test_requested_tasks_and_resume_helpers(tmp_path):
    mod = load_demographics_module()

    assert mod.get_requested_tasks("gender") == ("gender",)
    assert mod.get_requested_tasks("both") == ("gender", "age")

    params = {
        "method": "pyspi",
        "func": "cov_EmpiricalCovariance",
        "method_id": 0,
        "parc_size": 200,
        "pool": 3,
        "sparsity": 0.0,
        "n_splits": 2,
        "perm_test": False,
        "lag": 0,
        "seed": 2142,
    }
    task_out_dir = mod.get_task_out_dir("gender", params, tmp_path)

    assert not mod.is_task_complete(task_out_dir, 2)

    task_out_dir.mkdir(parents=True)
    for split in range(2):
        (task_out_dir / f"split-{split}__state.pkl").write_bytes(b"test")

    assert mod.is_task_complete(task_out_dir, 2)
