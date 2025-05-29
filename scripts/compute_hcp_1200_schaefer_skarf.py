import json
import logging
import os
import pickle
import time
import tracemalloc
import warnings
from pathlib import Path

# Single-threaded script, parallelize outside
# Need to set before importing other packages to take effect.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "2"

import datasets as hfds
import numpy as np
import pyarrow as pa
import sklearn
import typer
from pyarrow import feather
from skarf.var import BaseVAR
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from arfcexp.skarf_utils import AVAILABLE_SKARF_FUNCS, create_skarf_func, get_skarf_coef

NUM_CV_SPLITS = 5
TS_LENGTH = 1200

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

logging.getLogger("skarf").setLevel(logging.WARNING)

sklearn.set_config(enable_metadata_routing=True)
hfds.utils.disable_progress_bar()


def main(
    func: str,
    sub: str,
    parc_size: int = 200,
    pool: int = 3,
    lag: int = 0,
    order: int = 1,
    out_dir: str | None = None,
):
    project_root = Path(os.environ["PROJECT_ROOT"])

    func_id = AVAILABLE_SKARF_FUNCS.index(func)

    sub_list_path = (
        project_root / "resources/subject_lists/hcp_complete_data_867_subject_list.txt"
    )
    sub_list = sub_list_path.read_text().strip().split()
    sub_id_map = {sub: ii for ii, sub in enumerate(sub_list)}
    sub_id = sub_id_map[sub]

    logging.info(
        "Computing skarf on HCP 1200 rfMRI Schaefer:\n"
        f"\t{func_id=:03d} {sub_id=:03d} {func=} {sub=} {parc_size=} {pool=} "
        f"{lag=} {order=}"
    )

    if out_dir is None:
        out_dir = project_root / "data/hcp_1200_rfmri_schaefer_skarf"
    else:
        out_dir = Path(out_dir)

    out_path = (
        out_dir
        / f"parc-{parc_size}__pool-{pool}__lag-{lag}__order-{order}"
        / f"{func_id:03d}__func-{func}"
        / f"{sub_id:03d}__sub-{sub}.arrow"
    )
    if out_path.exists():
        logging.info("Output already exists; exiting.")
        return

    # Create skarf func.
    model, needs_groups = create_skarf_func(
        func, cv=LeaveOneGroupOut(), lag=lag, order=order
    )

    # We use leave one segment out cross-validation for some methods.
    if needs_groups:
        pooled_length = TS_LENGTH // pool
        assert pooled_length % NUM_CV_SPLITS == 0
        groups = np.repeat(np.arange(NUM_CV_SPLITS), pooled_length // NUM_CV_SPLITS)
        fit_params = {"segments": groups, "groups": groups}
    else:
        fit_params = {}

    # Load time series data.
    data_dir = project_root / "data/hcp_1200_rfmri_schaefer_timeseries"
    ts_dataset = hfds.load_from_disk(data_dir)
    ts_dataset = ts_dataset.select_columns(
        ["sub", "ses", "run", f"timeseries_n{parc_size}"]
    )
    ts_dataset = ts_dataset.filter(lambda sub_: sub_ == sub, input_columns="sub")
    ts_dataset.set_format("numpy")
    assert len(ts_dataset) == 4, "Expected 4 runs for each subject."

    schema = pa.schema(
        {
            "func_id": pa.int32(),
            "sub_id": pa.int32(),
            "func": pa.string(),
            "sub": pa.string(),
            "ses": pa.uint8(),
            "run": pa.uint8(),
            "success": pa.bool_(),
            "run_time": pa.float32(),
            "peak_mem_mb": pa.float32(),
            "err": pa.string(),
            "train_score": pa.float32(),
            "val_score": pa.float32(),
            "test_score": pa.float32(),
            "scores": pa.list_(pa.float32()),
            "mat": pa.list_(pa.float32()),
        }
    )
    records = []
    states = []

    tracemalloc.start()

    for ii, sample in enumerate(ts_dataset):
        X = np.asarray(sample[f"timeseries_n{parc_size}"], dtype=np.float64)
        # Pool consecutive time points to reduce effective TR
        X = pool_timeseries(X, pool=pool)
        # Standard scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Monitor memory and run time
        tracemalloc.clear_traces()
        tracemalloc.reset_peak()
        tic = time.perf_counter()

        model_i: BaseVAR = clone(model)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_i.fit(X, **fit_params)
            err = None
        except Exception as exc:
            model_i = None
            err = repr(exc)

        success = err is None
        run_time = time.perf_counter() - tic
        _, peak_mem = tracemalloc.get_traced_memory()
        peak_mem_mb = peak_mem / 1024**2

        if success:
            mat = get_skarf_coef(model_i)
            mat = mat.astype(np.float32).flatten()
        else:
            mat = None

        if success:
            train_score, val_score, test_score, scores = evaluate_model(
                model_i, ii, ts_dataset, scaler, parc_size=parc_size, pool=pool
            )
        else:
            train_score = val_score = test_score = np.nan
            scores = None

        result = {
            "func_id": func_id,
            "sub_id": sub_id,
            "func": func,
            "sub": sub,
            "ses": sample["ses"],
            "run": sample["run"],
            "success": success,
            "run_time": run_time,
            "peak_mem_mb": peak_mem_mb,
            "err": err,
            "train_score": train_score,
            "val_score": val_score,
            "test_score": test_score,
            "scores": scores,
            "mat": mat,
        }
        records.append(result)

        states.append(model_i)

    tracemalloc.stop()

    tab = pa.Table.from_pylist(records, schema=schema)

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(tab, out_path, compression="uncompressed")

    # Save model states
    out_state_path = out_path.with_suffix(".pkl")
    with out_state_path.open("wb") as f:
        pickle.dump(states, file=f)

    # Print summary
    success = float(tab["success"].to_numpy().mean())
    run_time = float(tab["run_time"].to_numpy().mean())
    peak_mem_mb = float(tab["peak_mem_mb"].to_numpy().mean())
    score = float(np.nanmean(tab["test_score"].to_numpy()))

    summary = {
        "func_id": func_id,
        "sub_id": sub_id,
        "func": func,
        "sub": sub,
        "success": success,
        "run_time": run_time,
        "peak_mem_mb": peak_mem_mb,
        "score": score,
    }

    logging.info(
        f"Done ({func_id:03d}, {sub_id:03d}): {func} {sub}\n{json.dumps(summary)}"
    )


def pool_timeseries(X: np.ndarray, pool: int) -> np.ndarray:
    N, D = X.shape
    length = pool * (N // pool)
    X = X[:length]
    X = X.reshape(length // pool, pool, D).mean(axis=1)
    return X


def evaluate_model(
    model: BaseVAR,
    train_idx: int,
    ts_dataset: hfds.Dataset,
    scaler: StandardScaler,
    parc_size: int,
    pool: int,
):
    """Evaluate the VAR model.

    We assume the dataset has 2 sessions with 2 runs each. The train score is computed
    on the training data itself, the val score on the other run from the same session,
    and the test score as the average of the two runs for the other session.
    """
    train_score = val_score = test_score = 0.0
    scores = []
    train_sample = ts_dataset[train_idx]

    for test_idx, test_sample in enumerate(ts_dataset):
        X_test = np.asarray(test_sample[f"timeseries_n{parc_size}"], dtype=np.float64)
        X_test = pool_timeseries(X_test, pool=pool)
        X_test = scaler.transform(X_test)
        score = model.score(X_test)
        scores.append(score)

        if test_idx == train_idx:
            train_score = score
        elif test_sample["ses"] == train_sample["ses"]:
            val_score = score
        else:
            test_score += 0.5 * score

    return train_score, val_score, test_score, scores


if __name__ == "__main__":
    typer.run(main)
