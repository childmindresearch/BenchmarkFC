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


## 2025-03-20

- Added motion censoring following [Li et al., NeuroImage 2019](https://www.sciencedirect.com/science/article/abs/pii/S1053811919303027). Only difference is I used a FD spike threshold of 0.4 rather than 0.2. I'm computing FD using Power rather than Jenkinson method (which Li et al use), 0.2 threshold for this metric results in too much censoring, 0.4 is in the typical range of [0.2, 0.5], and most importantly, 0.4 is close to the actual (boxplot) outlier threshold for the data (0.375).

- Updated subject filtering using censoring, resulting in 867 subjects. Comparable with 835 from [Kong et al., NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120044).

- Implemented time series preprocessing following Li et al., 2019; Kong et al., 2023 (censoring interpolation, GSR, bandpass filter, z-scoring). Using `nilearn.signal.clean`. Also noticed that the default bandpass filtering in nilearn seems to introduce some boundary effects. So I addressed these by padding, interpolating, and trimming. Tested out on synthetic time series and it looked good.

## 2025-03-21

- Tested out the preprocessing on a few example subjects. There were a few issues:

  1) Large differences in the mean activity per ROI (vertical banding).
  2) Probably as a result, GSR is necessary to get meaningful time series (why ?).
  3) Masking interpolation doesn't always work well
  4) Boundary interpolation doesn't seem necessary

- Trying to get surface plotting infrastructure set up.
  - nilearn surface plotting has always been too slow. Going to see if I can get pyvista to work.
  - trying their [create poly notebook](https://docs.pyvista.org/examples/00-load/create-poly#sphx-glr-examples-00-load-create-poly-py)

  - kernel crash error, due to remote jupyter notebook execution

    ```
    The Kernel crashed while executing code in the current cell or a previous cell.  Please review the code in the cell(s) to identify a possible cause of the failure.  Click here for more info.  View Jupyter log for further details.
    ```

    I remember this also from brainspace. Maybe it's been since fixed. The [pyvista jupyter guide](https://docs.pyvista.org/user-guide/jupyter/) suggests installing `trame`.

  - Got issue setting backend to `trame`. [This issue](https://github.com/pyvista/pyvista/issues/5848) suggests installing `pyvista[jupyter]` to get all dependencies.

  - Import works now, but issue due to no display:

    ```
    /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/.venv/lib/python3.11/site-packages/pyvista/plotting/plotter.py:151: UserWarning:
    This system does not appear to be running an xserver.
    PyVista will likely segfault when rendering.

    Try starting a virtual frame buffer with xvfb, or using
      ``pyvista.start_xvfb()``
    ```

  - Now the example runs, but don't see anything.

  - Using `'static'` backend works, hooray 🎉.


## 2025-03-25

- Finished with implementing the pyvista based surface plotting utility. I think it will be pretty useful. Pushed it to its own project ([yaspy](https://github.com/childmindresearch/yaspy)).


## 2025-04-11

- Returning to this after some time away visiting family.

- First thing to resolve is what is going on with the time series preprocessing. I dug into it and identified a couple issues:

  1. The interpolation across censored frames doesn't work, introduces large overshoots (signal spikes). Overshoots do sometimes happen, esp with cubic spline. I also tried pchip interpolate, though that did not seem to resolve. Decided to just not interpolate. Motion spike related artifacts should largely be removed in any case thanks to ICA. Nb that in Yeo lab papers, they say they use lomb-scargle periodogram to interpolate censored frames. But I couldn't immediately find out how to apply this in python.

  2. Large Edge effects induced by the temporal filtering. Edge artifacts are common, and indicate an issue with padding. By default, reflect padding with odd symmetry is used. This makes the extension continuous with continuous derivative at the boundary, but introduces a large mean shift, which passes through the low pass filter. Even padding preserves the mean, at the expense of a discontinous derivative. But then, the signal is noisy so a discontinuous derivative doesn't matter too much.

- I updated the timeseries preprocessing util to reflect these changes and added notebooks to demo the results.

- Added script to extract filtered parcellated time series. Using huggingface datasets to generate and save data.


## 2025-04-14

- Visualized preprocessed parcellated time series. Look good. Noticed that there are clear periods of large activity. Feels vaguely related to something I've seen in Rick Betzel's papers. I wonder how much of functional connectivity is driven by these spikes.

- Testing PySPI SPIs (for the umpteenth time..).

## 2025-04-15

- Ran test of PySPI SPIs over night, 2 hour time limit, 4 GB memory. 6 did OOM, 53 timed out, and 16 otherwise failed.

- Debugging failed SPIs:

  ```
  cov_EllipticEnvelope
  cov_GraphicalLasso
  cov_GraphicalLassoCV
  cov_MinCovDet
  cov-sq_EllipticEnvelope
  cov-sq_GraphicalLasso
  cov-sq_GraphicalLassoCV
  cov-sq_MinCovDet
  prec_EllipticEnvelope
  prec_GraphicalLasso
  prec_GraphicalLassoCV
  prec_MinCovDet
  prec-sq_EllipticEnvelope
  prec-sq_GraphicalLasso
  prec-sq_GraphicalLassoCV
  prec-sq_MinCovDet
  ```

- These all crashed, no raised exception.

- I don't understand why no raised exception, but in separate testing, they often fail when `n_samples < n_features` due to [this regression bug](https://github.com/scikit-learn/scikit-learn/issues/30625).

- I can't easily downgrade sklearn, since skarf uses features from >= 1.6. E.g. in v1.5.2 there is [this metadata routing issue](https://github.com/scikit-learn/scikit-learn/pull/29634), which breaks tests in `LinearVAR`.

- Decided that downgrading sklearn is too much hassle for now, and will just accept that these methods fail for `n_samples < n_features`. More concerning to me is why I could not catch the exception.
