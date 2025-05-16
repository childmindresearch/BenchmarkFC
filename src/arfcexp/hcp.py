import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

HCP_PHENO_UNRESTRICTED_PATH = Path(os.environ["HCP_PHENO_UNRESTRICTED"])
HCP_PHENO_RESTRICTED_PATH = Path(os.environ["HCP_PHENO_RESTRICTED"])


def parse_hcp_metadata(path: Path) -> dict[str, str]:
    """Parse metadata from HCP file path."""
    sub = path.parents[3].name
    acq = path.parent.name
    if "7T" in acq:
        mod, task, mag, dir = acq.split("_")
    else:
        mod, task, dir = acq.split("_")
        mag = "3T"
    clean = "hp2000_clean" in path.name
    metadata = {
        "sub": sub,
        "mod": mod,
        "task": task,
        "mag": mag,
        "dir": dir,
        "clean": clean,
    }
    return metadata


def load_hcp_subject_list(
    subset: str = "hcp_complete_data_867",
) -> list[str]:
    sub_list_path = PROJECT_ROOT / f"resources/subject_lists/{subset}_subject_list.txt"
    sub_list = sub_list_path.read_text().strip().split()
    return sub_list


def load_hcp_pheno(restricted: bool = False) -> pd.DataFrame:
    hcp_pheno = pd.read_csv(
        HCP_PHENO_RESTRICTED_PATH if restricted else HCP_PHENO_UNRESTRICTED_PATH,
        dtype={"Subject": str},
    )
    hcp_pheno.set_index("Subject", inplace=True)
    return hcp_pheno


def load_hcp_behav_columns() -> list[str]:
    # Get 58 behavioral columns used in Yeo lab papers.
    hcp_behav_columns = (
        (PROJECT_ROOT / "resources/column_lists/58behaviors_age_sex.txt")
        .read_text()
        .splitlines()
    )
    # Drop age, sex.
    hcp_behav_columns = hcp_behav_columns[:58]
    return hcp_behav_columns


def load_hcp_behav() -> pd.DataFrame:
    hcp_behav_columns = load_hcp_behav_columns()
    hcp_pheno = load_hcp_pheno()
    hcp_behav = hcp_pheno.loc[:, hcp_behav_columns]
    return hcp_behav


def load_hcp_mean_fd() -> pd.DataFrame:
    hcp_fd_path = PROJECT_ROOT / "results/hcp_1200_rfmri_fd/hcp_1200_rfmri_fd.parquet"
    if not hcp_fd_path.exists():
        raise FileNotFoundError(
            f"HCP FD path {hcp_fd_path} does not exist; run compute_hcp_1200_rfmri_fd"
        )

    hcp_fd = pd.read_parquet(hcp_fd_path)

    # Only include 3T data and full runs.
    hcp_fd = hcp_fd.query("mag == '3T' and n_frames == 1200")

    hcp_mean_fd = hcp_fd.groupby("sub").agg({"mean_fd": "mean"})
    hcp_mean_fd.columns = ["Mean_FD"]
    return hcp_mean_fd


def load_hcp_covariates() -> pd.DataFrame:
    hcp_pheno = load_hcp_pheno()
    hcp_gender = hcp_pheno.loc[:, "Gender"]

    hcp_restricted_pheno = load_hcp_pheno(restricted=True)
    hcp_age_years = hcp_restricted_pheno.loc[:, "Age_in_Yrs"]

    hcp_mean_fd = load_hcp_mean_fd()

    covariates = pd.concat([hcp_gender, hcp_mean_fd, hcp_age_years], axis=1)
    return covariates


def load_hcp_family_groups() -> pd.Series:
    hcp_restricted_pheno = load_hcp_pheno(restricted=True)
    hcp_family_id = hcp_restricted_pheno.loc[:, "Pedigree_ID"]

    # Relabel to [0, N)
    _, hcp_family_groups = np.unique(hcp_family_id.values, return_inverse=True)
    hcp_family_groups = pd.Series(
        hcp_family_groups,
        index=hcp_family_id.index,
        name="Family_Group",
    )
    return hcp_family_groups


def load_hcp_behav_factors_topk():
    hcp_factor_topk_path = (
        PROJECT_ROOT / "results/hcp_1200_behav/hcp_1200_behav_factors_topk.json"
    )
    if not hcp_factor_topk_path.exists():
        raise FileNotFoundError(
            f"HCP factor top-k path {hcp_factor_topk_path} does not exist; "
            "run analyze_hcp_1200_behav"
        )

    with hcp_factor_topk_path.open() as f:
        hcp_factor_topk = json.load(f)
    return hcp_factor_topk
