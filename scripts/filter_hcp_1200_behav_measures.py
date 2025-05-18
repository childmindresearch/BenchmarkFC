"""
Some measures have very low variance. We will exclude these. Specifically, measures with
`IQR / (max - min) < 0.01`.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
HCP_PHENO_UNRESTRICTED_PATH = Path(os.environ["HCP_PHENO_UNRESTRICTED"])
IQR_NORM_CUTOFF = 0.01


def main():
    sub_list_path = (
        PROJECT_ROOT / "resources/subject_lists/hcp_complete_data_867_subject_list.txt"
    )
    sub_list = sub_list_path.read_text().strip().split()

    hcp_pheno = pd.read_csv(HCP_PHENO_UNRESTRICTED_PATH, dtype={"Subject": str})
    hcp_pheno.set_index("Subject", inplace=True)
    hcp_pheno = hcp_pheno.loc[sub_list]

    hcp_behav_columns = (
        (PROJECT_ROOT / "resources/column_lists/58behaviors.txt")
        .read_text()
        .splitlines()
    )
    hcp_behav = hcp_pheno.loc[:, hcp_behav_columns]

    q1, q3 = np.quantile(hcp_behav, [0.25, 0.75], axis=0)
    iqr = q3 - q1
    value_range = np.max(hcp_behav.values, axis=0) - np.min(hcp_behav.values, axis=0)
    iqr_norm = iqr / value_range

    print("Excluding the following measures:")
    (indices,) = np.where(iqr_norm < IQR_NORM_CUTOFF)
    for idx in indices:
        col = hcp_behav.columns[idx]
        print(f"\t{col} {iqr_norm[idx]:.4f}")

    hcp_behav_columns_with_var = [
        col
        for ii, col in enumerate(hcp_behav.columns)
        if iqr_norm[ii] >= IQR_NORM_CUTOFF
    ]
    num = len(hcp_behav_columns_with_var)

    column_list_path = (
        PROJECT_ROOT / f"resources/column_lists/{num}behaviors_with_var.txt"
    )
    with column_list_path.open("w") as f:
        print("\n".join(hcp_behav_columns_with_var), file=f)


if __name__ == "__main__":
    main()
