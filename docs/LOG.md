# ARFC experiments Logbook

## 2025-03-13

- Set up skeleton for experiments.
- Set up uv environment.
- Added `skarf` dependency.
- Trying to update `skarf` to include the `pyspi` dependency, but `uv sync -P skarf` is getting stuck.

```
  × Failed to download `torch==2.6.0`
  ├─▶ Failed to extract archive
  ╰─▶ Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT
      (current value: 30s).
  help: `torch` (v2.6.0) was included because `arfc-experiments` (v0.1.0) depends on
        `skarf[pyspi]` (v0.1.0a1.dev1+gef2b16e) which depends on `pyspi` (v1.1.1) which depends
        on `torch`
```


## 2025-03-14

- Added `skarf` as a submodule editable dependency.
- Added [`dvc`](https://dvc.org) and initialized dvc project. [Opted out of dvc usage tracking](https://dvc.org/doc/user-guide/analytics)
- Added a `dvc` local remote backup at `/ocean/projects/med220004p/clane2/.backup/dvcstore`.

- Script to download schaefer: [`download_schaefer_parcellations.sh`](../scripts/download_schaefer_parcellations.sh).

- Added dvc stage to download parcellations, fun

  ```bash
  dvc stage add -n download_schaefer \
    -d scripts/download_schaefer_parcellations.sh \
    --outs-persist-no-cache resources/schaefer_parcellations \
    bash scripts/download_schaefer_parcellations.sh
  ```

- Got rid of dvc... too much machinery, doesn't work well for long running slurm tasks (https://github.com/iterative/dvc/issues/7419).

- Also thought about `make`, same problem of asynchronous long-running slurm jobs.

- Added [`just`](https://github.com/casey/just), which is similar to `make`, but just for running and abbreviating commands.

- Looking into motion info for HCP. In [the manual](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf), page 96, it says:

  ```
  Motion parameters. Estimates of motion parameters are saved into two different files: Movement_Regressors.txt and Movement_Regressors_dt.txt. The first file (Movement_Regressors.txt) contains 12 variables. The first six variables are the motion parameters estimates from a rigid-body transformation to the SBRef image acquired at the start of each fMRI scan.
     trans_x (mm)
     trans_y (mm)
     trans_z (mm)
     rot_x (deg)
     rot_y (deg)
     rot_z (deg)
  ```

- Added function to compute FD.

- Downloaded He 2019 KRDNN 953 subject list for HCP and Li 2019 GSR HCP behavioral column lists.

## 2025-03-19

- Computed FD on all HCP subjects.
- Wrote notebook to analyze FD. Also used as a small test for how to do reproducible notebooks:
  - Loading mpl style from style sheet
  - Execute with `jupyter execute --inplace`
- Wrote notebook to filter hcp subjects:
  - complete 58 behavioral data
  - complete 3T rest data
  - mean FD < 0.3 (nb, FD outlier threshold is 0.28, but 0.3 is a standard threshold and close enough).
