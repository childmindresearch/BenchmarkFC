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


## 2025-04-17

- Seems like these failing SPIs above crash (no exception) with seg fault only if I initialize the JVM. Odd, bc these don't require the JVM..

- Only the `infotheory` metrics require JVM/octave. I think we should only initialize these when creating an infotheory metric.

- Updated the pyspi interface in skarf to do this. Now the above failing SPIs run without crashing (though they still raise exception in some cases).

- Re-running the pyspi profile/test, with longer run time.


## 2025-04-21

- Second run of pyspi profile/test worked. 20 SPIs timed out, 6 hit OOM, none failed.

- Following SPIs had (caught) errors:

  ```
  cov-sq_GraphicalLasso
  cov_GraphicalLasso
  prec-sq_GraphicalLasso
  prec_GraphicalLasso
  sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-None
  sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-None
  sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-None
  sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-None
  sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-None
  sgc_parametric_mean_fs-1_fmin-0_fmax-0-5_order-None
  xme_gaussian_k10
  ```

  - GraphicalLasso fails bc of bad regularization strength, GraphicalLassoCV works fine though.

  - sgc failes bc of following error:

    ```
    ValueError: Model estimation order did not converge at max_order = 50
    ```

    But variants of sgc with order not equal to None work fine.

  - xme_gaussian_k10 fails with following error (rank deficient)

    ```
    infodynamics.utils.NonPositiveDefiniteMatrixException: infodynamics.utils.NonPositiveDefiniteMatrixException: CholeskyDecomposition is only performed on positive-definite matrices. Some reasons for non-positive-definite matrix are listed at http://www2.gsu.edu/~mkteer/npdmatri.html - note: a correlation matrix is non-positive-definite if you have more variables than observations. Failed row is 7
    ```

    But it works for `xme_gaussian_k1`.

- So all of the failures represent just failing param config (which is normal), rather than failing functions.

## 2025-04-24

Updates to skarf, see the git commit log for more details.

- Decomposition based linear VAR. This model is aimed at leveraging correlations between the features, which are known to interfere especially with sparse models. Specifically, we first fit some decomposition to the data (e.g. spatial ICA), and then constrain the linear model coefficients to the span of the decomposition basis (by multiplying the training data by the basis).

- Simplifications and improvements to `CovarianceVAR`. In particular, implemented a per-target poly fit option which makes more consistent with `LinearVAR`, and fits data much better. Intuitively, makes sense that each target might benefit from its own poly transform. (Also, note that this makes the Covariance VAR more similar to the decomposition linear VAR, where the VAR coefficients per target are constrained to a particular basis. In the Covariance VAR case, the basis consists of polynomial terms generated from the covariance vector for the target. Whereas in the linear VAR case, the basis is a precomputed decomposition shared across targets.)


## 2025-04-28

Wrapped up pyspi filtering notebook based on the profile/test results. Filtering based on completion success and run time with a liberal knee point cutoff (1200s).

Implemented script to compute SPI matrices with parallelism leveraging huggingface datasets.

Launched job to compute overnight.

## 2025-04-29

Job overnight got through 50 SPIs. Decided this was going to take too long/too much compute. Made the SPI filtering stricter:

- Exclude redundant square and precision SPIs (these can be computed later if needed).
- Use stricter run time knee point cutoff of 300s.

Also now running the job in batches using a job array.

## 2025-05-07

The ACCESS cluster was down last week due to severe weather. Back up now. (In the meantime, rewrote bids2table hah.)

Added a notebook for cleaning and analyzing the HCP behavioral data. Using the 58 behavioral measures that have been used across recent studies from the Yeo group.

Standard scaled each behavioral measure and clipped to +/- 3 sigma (to reduce the influence of outlier scores). (Standard scaling in this context makes sense because the measures have very different and mostly arbitrary units.)

Regressed gender, mean FD, and age from each behavioral measure. Nb, HCP restricted csv required for getting age. Mean FD computed from my previously computed FD data.

Did a factor analysis also following previous Yeo lab papers (Ooi et al., 2022; Kong et al., 2023). Found 4 factors: dissatisfaction, cognition, support, and emotion. Consistent with their results, except support is extra and not reported there.

Saved component scores to use as prediction targets.

Note that the factors are computed on the full set of subjects. Previous works computed factors on a held out set of subjects "to prevent leakage". But I think this is not necessary, since we are not interested in testing the generalization of the factors, but the prediction of the factor targets from brain data.

## 2025-5-08

Looking at outputs from batch pyspi job.

```
30925558_0   compute_p+         RM 1-00:00:00 2025-05-05T10:24:41            r043        128            128-00:40:32            1-00:00:19    TIMEOUT      0:0
30925558_1   compute_p+         RM 1-00:00:00 2025-05-05T10:24:41            r202        128            128-00:40:32            1-00:00:19    TIMEOUT      0:0
30925558_2   compute_p+         RM 1-00:00:00 2025-05-05T10:24:41            r381        128            7-01:40:16              01:19:32  COMPLETED      0:0
30925558_3   compute_p+         RM 1-00:00:00 2025-05-05T10:24:41            r413        128            44-14:41:04              08:21:53  COMPLETED      0:0
30925558_4   compute_p+         RM 1-00:00:00 2025-05-05T10:24:41            r052        128              01:06:08              00:00:31     FAILED      2:0
```

Two jobs completed successfully, one failed immediately, and two timed out.

The two time out jobs timed out for frustrating reasons...

```
[INFO 25-05-05 10:28:10]: Computing PySPI on HCP 1200 rfMRI Schaefer:
        spi_id=099 spi='je_kernel_W-0.5' parc_size=200 pool=3
[INFO 25-05-05 10:28:10]: Saving to: /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/data/hcp_1200_rfmri_schaefer_pyspi/parc-200__pool-3/099__spi-je_kernel_W-0.5
[INFO 25-05-05 10:28:10]: Running with 64 processes.
[INFO 25-05-05 10:28:10]: Loading SPI: je_kernel_W-0.5
[INFO 25-05-05 10:28:10]: Loading SPI config map from cache: /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/resources/spi_lists/spi_config_map_all.yaml
[INFO 25-05-05 10:34:09]: Loaded PySPI optional depedencies: {'octave': True, 'java': True}
[INFO 25-05-05 10:34:09]: SPICovariance(spi=<pyspi.statistics.infotheory.JointEntropy object at 0x14731c265fd0>)
[INFO 25-05-05 10:34:09]: Loading time series dataset:
        /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/data/hcp_1200_rfmri_schaefer_timeseries
[INFO 25-05-05 10:34:10]: Computing SPI matrices
Starting JVM with java class /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/.venv/lib/python3.11/site-packages/pyspi/lib/jidt/infodynamics.jar.
Map (num_proc=64):   0%|          | 0/3468 [00:00<?, ? examples/s]/ocean/projects/med220004p/clane2/miniconda3/pkgs/openjdk-8.0.412-h2b85faf_2/jre/lib/rt.jar: invalid LOC header (bad signature)
/ocean/projects/med220004p/clane2/miniconda3/pkgs/openjdk-8.0.412-h2b85faf_2/jre/lib/rt.jar: invalid LOC header (bad signature)
/ocean/projects/med220004p/clane2/miniconda3/pkgs/openjdk-8.0.412-h2b85faf_2/jre/lib/rt.jar: invalid LOC header (bad signature)
Map (num_proc=64):   0%|          | 1/3468 [00:00<08:56,  6.47 examples/s]/ocean/projects/med220004p/clane2/miniconda3/pkgs/openjdk-8.0.412-h2b85faf_2/jre/lib/rt.jar: invalid LOC header (bad signature)
Map (num_proc=64):   3%|▎         | 96/3468 [00:18<01:08, 49.05 examples/s]slurmstepd: error: *** JOB 30932723 ON r202 CANCELLED AT 2025-05-06T10:25:00 DUE TO TIME LIMIT ***
```

```
[INFO 25-05-05 13:20:03]: Computing PySPI on HCP 1200 rfMRI Schaefer:
        spi_id=097 spi='je_gaussian' parc_size=200 pool=3
[INFO 25-05-05 13:20:03]: Saving to: /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/data/hcp_1200_rfmri_schaefer_pyspi/parc-200__pool-3/097__spi-je_gaussian
[INFO 25-05-05 13:20:03]: Running with 64 processes.
[INFO 25-05-05 13:20:03]: Loading SPI: je_gaussian
[INFO 25-05-05 13:20:03]: Loading SPI config map from cache: /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/resources/spi_lists/spi_config_map_all.yaml
[INFO 25-05-05 13:21:21]: Loaded PySPI optional depedencies: {'octave': True, 'java': True}
[INFO 25-05-05 13:21:21]: SPICovariance(spi=<pyspi.statistics.infotheory.JointEntropy object at 0x14a317539650>)
[INFO 25-05-05 13:21:21]: Loading time series dataset:
        /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/data/hcp_1200_rfmri_schaefer_timeseries
[INFO 25-05-05 13:21:22]: Computing SPI matrices
Starting JVM with java class /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/.venv/lib/python3.11/site-packages/pyspi/lib/jidt/infodynamics.jar.
Map (num_proc=64):   0%|          | 0/3468 [00:00<?, ? examples/s]slurmstepd: error: *** JOB 30932722 ON r043 CANCELLED AT 2025-05-06T10:25:00 DUE TO TIME LIMIT ***
```

Both stalled for nearly 24 hours while computing a `je` SPI 🤬🤬.

It would be better if I had a way to detect this inactivity in a large job and at least cancel the job.

For now, it seems these java SPIs are suspect, at least when running in parallel. I also had the earlier issue where they interfere with other SPIs.

These are all the infotheory SPIs, IDs in range `[098, 140]`.

```
je_gaussian
je_kozachenko
je_kernel_W-0.5
ce_gaussian
ce_kozachenko
ce_kernel_W-0.5
mi_gaussian
mi_kraskov_NN-4_DCE
mi_kernel_W-0.25
tlmi_gaussian
tlmi_kraskov_NN-4
tlmi_kraskov_NN-4_DCE
tlmi_kernel_W-0.25
te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1
te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1
te_kernel_W-0.25_k-1
gc_gaussian_k-1_kt-1_l-1_lt-1
te_symbolic_k-1_kt-1_l-1_lt-1
te_symbolic_k-10_kt-1_l-1_lt-1
```

Testing out `je_gaussian`. Expected to take 80 sec per run from profiling. Computed only 16 matrices in one hour with 4 procs (despite claim of 10 examples/s).

```
[INFO 25-05-08 11:59:42]: Loading time series dataset:
        /ocean/projects/med220004p/clane2/ARFC/arfc-experiments/data/hcp_1200_rfmri_schaefer_timeseries
Loading dataset from disk: 100%|██████████████████████████████████| 157/157 [00:00<00:00, 22608.09it/s]
[INFO 25-05-08 11:59:43]: Computing SPI matrices
Map (num_proc=4):   0%|▏                                      | 16/3468 [00:19<05:37, 10.22 examples/s]
```

I think it must have an issue with multiprocessing.

Removed old pyspi outputs from `hcp_1200_rfmri_schaefer_pyspi_v1`, which were ran with the earlier spi list and only got through 50.


## 2025-05-09

To deal with this issue, I thought about just runing these SPIs separately. But I would ideally like to have a script that can run all of them cleanly.

Going to implement a new single threaded script, and then parallelize outside with gnu parallel (this tends to be a more robust approach anyway).

First want to check that these java SPIs can even run in parallel or if there is too much contention over the jar perhaps.

```sh
parallel uv run python scripts/test_profile_pyspi.py \
  {} '[400]' '[200]' --outdir tmp/java_profile_pyspi_parallel ::: {97..102}
```

Yes, these all ran fine, good. Must be an issue with python multiprocessing incompatibility with the jvm interface.


## 2025-05-12

Ran pyspi over the weekend. Successful run.

```
JobID           JobName  Partition  Timelimit               Start        NodeList      NCPUS     MaxRSS    CPUTime     AveCPU    Elapsed      State ExitCode
------------ ---------- ---------- ---------- ------------------- --------------- ---------- ---------- ---------- ---------- ---------- ---------- --------
31083735_0   compute_p+         RM 1-00:00:00 2025-05-10T08:33:43            r307        128            108-13:47:44              20:21:28 OUT_OF_ME+    0:125
31083735_0.+      batch                       2025-05-10T08:33:43            r307        128 244309668K 108-13:47:44 59-19:50:+   20:21:28 OUT_OF_ME+    0:125
31083735_0.+     extern                       2025-05-10T08:33:43            r307        128      4304K 108-13:47:44   00:00:27   20:21:28  COMPLETED      0:0
31083735_1   compute_p+         RM 1-00:00:00 2025-05-10T08:33:43            r313        128            111-01:10:24              20:49:18 OUT_OF_ME+    0:125
31083735_1.+      batch                       2025-05-10T08:33:43            r313        128 243627228K 111-01:10:24 60-18:18:+   20:49:18 OUT_OF_ME+    0:125
31083735_1.+     extern                       2025-05-10T08:33:43            r313        128      4064K 111-01:10:24   00:00:09   20:49:18  COMPLETED      0:0
31083735_2   compute_p+         RM 1-00:00:00 2025-05-10T08:33:43            r347        128            109-10:44:16              20:31:17 OUT_OF_ME+    0:125
31083735_2.+      batch                       2025-05-10T08:33:43            r347        128 244428552K 109-10:44:16 59-22:58:+   20:31:17 OUT_OF_ME+    0:125
31083735_2.+     extern                       2025-05-10T08:33:43            r347        128      1192K 109-10:44:16   00:00:07   20:31:17  COMPLETED      0:0
31083735_3   compute_p+         RM 1-00:00:00 2025-05-10T08:33:43            r402        128            111-12:28:48              20:54:36 OUT_OF_ME+    0:125
31083735_3.+      batch                       2025-05-10T08:33:43            r402        128 241987184K 111-12:28:48 61-00:09:+   20:54:36 OUT_OF_ME+    0:125
31083735_3.+     extern                       2025-05-10T08:33:43            r402        128      4092K 111-12:28:48   00:00:03   20:54:36  COMPLETED      0:0
31083735_4   compute_p+         RM 1-00:00:00 2025-05-10T08:33:43            r415        128            103-10:10:08              19:23:31 OUT_OF_ME+    0:125
31083735_4.+      batch                       2025-05-10T08:33:43            r415        128 244224532K 103-10:10:08 57-03:11:+   19:23:31 OUT_OF_ME+    0:125
31083735_4.+     extern                       2025-05-10T08:33:43            r415        128        12K 103-10:10:08   00:00:05   19:23:31  COMPLETED      0:0
```

All jobs finished in roughly 20 hours. Initial estimate was 5 hours per job, so a bit slower than expected. But utilization is ok (~50%). All jobs report going out of memory, but it seems they weren't cancelled (perhaps because they ran on full nodes).

All SPIs are complete except one:

```bash
output_dir="${PROJECT_ROOT}/data/hcp_1200_rfmri_schaefer_pyspi/parc-200__pool-3"
cd $output_dir

while read spi; do
  dname=$(echo *_spi-${spi})
  if [[ ! -d $dname ]]; then
    echo $dname 0
  else
    count=$(echo ${dname}/*.arrow | wc -w)
    if (( count != 867 )); then
      echo $dname $count
    fi
  fi
done < ${PROJECT_ROOT}/resources/spi_lists/spi_list_select_300s_142.txt
```

```
123__spi-tlmi_kraskov_NN-4 9
```

I don't know why only this one failed or what happened exactly.


## 2025-05-13

I have tried to get away with a target preprocessing scheme applied jointly to all behavioral targets, before train test splitting. But I decided I can't do this for two reasons:

1) Prior reference works apply fit the statistics for target preprocessing (scaling, nuisance regression) on train targets only, to avoid test data leakage.

2) GPT convinced me I really do have to do this.

Previously, I rationalized that this level of test set leakage was acceptable, since it is isolated from the training input data. But, I guess it is better to be fully rigorous and in agreement with prior methods.

To this end, I'm implementing sklearn style transformers that guard against test data leakage for the target preprocessing steps:

- scaling (ofc)
- nuisance regression

The one tricky case is the behavioral factors. I don't really want to refit factors to each train split separately, and then have to align them to some reference, and worry about replication. Rather, I think I will use the global factor analysis to cluster the 58 measures into 4 subsets, and just do plain average of the measures in each subset.

## 2025-05-14

Implemented an `HCPPhenoTargetTransform` and tested that it reproduces the results of the previous pipeline implemented in the `analyze_hcp_1200_pheno` notebook.

## 2025-05-15

Implemented `HCPPhenoRegressor` to encapsulate full preprocessing and model fit.


## 2025-05-16

Testing behavioral prediction pipeline. Renamed `HCPPheno*` -> `HCPBehav*`

Finished behavioral prediction script and ran job before signing off. Will check tomorrow.

## 2025-05-17

Looking at results of behavioral prediction. First off, I made a mistake by setting the wrong SPI list in the slurm job script 😅.

Rethinking how to preprocess the targets. I would like each target to be mean zero, stdev 1, so that mse is clearly interpretable as 1 - r2. I would also like to not be impacted by outliers and low variance measures.

Decided that some of the behavioral measures in the 58 list have too low variance to include. Excluding those with `IQR / (max - min) < 0.01`. This excludes:

```
Mars_Final 0.0057
Social_Task_Perc_TOM 0.0000
ER40HAP 0.0000
````

New target preprocessing pipeline:

- nuisance regression
- clip boxplot outliers
- standard scale

Histograms look even better with this pipeline, and mean zero unit variance is nice.


### Unstable pearson cross validation score

Looking at the results of the behavioral prediction, I noticed that the different splits are a bit all over the place. Wildly different scores, different alphas.

```
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 0, "alpha": 0.7, "train_score": 0.6421603364651384, "val_score": 0.006294984730124753, "test_score": 0.049982614331074204}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 1, "alpha": 0.7, "train_score": 0.6430477699318116, "val_score": 0.03497433831247067, "test_score": 0.17069091073426113}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 2, "alpha": 0.01, "train_score": 0.9867242390789055, "val_score": 0.07074390178972915, "test_score": -0.05293945926472552}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 3, "alpha": 0.01, "train_score": 0.9856713682718559, "val_score": 0.04628380588347194, "test_score": -0.007500372324638291}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 4, "alpha": 0.4, "train_score": 0.7013762715346581, "val_score": 0.042908665621448315, "test_score": 0.41036941340419747}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 5, "alpha": 0.01, "train_score": 0.9861410864568161, "val_score": 0.07701615653946206, "test_score": -0.09930342472959469}
```

I realized this is probably due the usage of Pearson correlation as the cross-validation metric. Crucially, pearson correlation rescales the predictions to be unit norm. This completely defeats the purpose of the shrinkage ridge regularization, and explains the instability of the CV.

[Yeo Lab uses Pearson as the CV metric.](https://github.com/ThomasYeoLab/CBIG/blob/v0.29.2-Kong2022_update/utilities/matlab/predictive_models/KernelRidgeRegression/CBIG_KRR_innerloop_cv.m#L219)

Instead, we will use MSE as the CV metric, which should behave better.

Indeed, testing it out in the same case as above, now we get full shrinkage to zero with consistent alpha=10 because the targets are not predictable...

```
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 0, "alpha": 10, "mse_train": 0.9611246216342486, "mse_val": 1.0180919079534982, "mse_test": 0.7745180700624525, "r2_train": 0.03887537836575139, "r2_test": -0.03562758869713534, "corr_train": 0.3971565620440809, "corr_test": 0.02492666684960681}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 1, "alpha": 10, "mse_train": 0.9610266408999235, "mse_val": 1.0082850325203854, "mse_test": 0.8623298339909494, "r2_train": 0.03897335910007671, "r2_test": 0.009632248040231106, "corr_train": 0.3919914253477742, "corr_test": 0.1419858124320163}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 2, "alpha": 10, "mse_train": 0.9600191177091754, "mse_val": 1.0105944022284774, "mse_test": 1.6128013937627186, "r2_train": 0.03998088229082464, "r2_test": -0.011360685443246643, "corr_train": 0.3743946095564662, "corr_test": -0.039428046353294326}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 3, "alpha": 10, "mse_train": 0.9621211490035952, "mse_val": 1.023441512967337, "mse_test": 0.8467339203776794, "r2_train": 0.03787885099640487, "r2_test": -0.12109222677269948, "corr_train": 0.42370800789429663, "corr_test": 0.14416719403204845}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 4, "alpha": 10, "mse_train": 0.9632325204941122, "mse_val": 1.0129599954959976, "mse_test": 1.3308963907659532, "r2_train": 0.03676747950588777, "r2_test": -0.03296702357648451, "corr_train": 0.39791043057665476, "corr_test": 0.43918166412960613}
{"spi": "cov_GraphicalLassoCV", "parc_size": 200, "pool": 3, "target": "Dissatisfaction", "seed": 2142, "split": 5, "alpha": 10, "mse_train": 0.960479340705597, "mse_val": 1.0100373474627873, "mse_test": 1.0633947982759717, "r2_train": 0.03952065929440263, "r2_test": -0.01437710350610466, "corr_train": 0.4048800844042982, "corr_test": -0.019794301259891005}
```

Now looking at a target where prediction is actually good (Cognition), I'm a little annoyed that the variance in the test metric is this high.

```
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 0, "alpha": 0.4, "mse_train": 0.4412719525576985, "mse_val": 0.8251346452792333, "mse_test": 0.8867604074436611, "r2_train": 0.5587280474423015, "r2_test": 0.19719826393842643, "corr_train": 0.8154691954195461, "corr_test": 0.4680292899183919}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 1, "alpha": 0.4, "mse_train": 0.4354918855684301, "mse_val": 0.8449765815744847, "mse_test": 0.6292856382852028, "r2_train": 0.5645081144315699, "r2_test": -0.03059304039084143, "corr_train": 0.8184052548865286, "corr_test": 0.2160045294821904}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 2, "alpha": 0.4, "mse_train": 0.4426833462853164, "mse_val": 0.8505979629822156, "mse_test": 0.4697452639967915, "r2_train": 0.5573166537146835, "r2_test": 0.30127402560509975, "corr_train": 0.8169863116630505, "corr_test": 0.5550368328539221}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 3, "alpha": 0.4, "mse_train": 0.43973320708982405, "mse_val": 0.82466104279595, "mse_test": 0.7286635202188009, "r2_train": 0.5602667929101759, "r2_test": 0.09283941674709784, "corr_train": 0.8163914421737204, "corr_test": 0.3339827610617342}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 4, "alpha": 0.4, "mse_train": 0.44542690111488026, "mse_val": 0.837328124638159, "mse_test": 0.7897997267580584, "r2_train": 0.5545730988851196, "r2_test": 0.261055304358221, "corr_train": 0.8130245405326928, "corr_test": 0.5753296296759542}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 5, "alpha": 0.4, "mse_train": 0.4478148986844923, "mse_val": 0.8727414165095275, "mse_test": 1.0144237185177405, "r2_train": 0.5521851013155077, "r2_test": 0.3107254568376149, "corr_train": 0.8132944875660526, "corr_test": 0.594690266802565}
```

I think the test splits are just too small (5%, ~44 samples). Trying out an outer loop CV strategy using group shuffle split with a fixed test size of 20%. This should be more stable.

```
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 0, "alpha": 0.4, "mse_train": 0.430620676946194, "mse_val": 0.8789212214707447, "mse_test": 0.6487049202014888, "r2_train": 0.569379323053806, "r2_test": 0.1597874143760476, "corr_train": 0.8290942523583859, "corr_test": 0.4031988599833575}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 1, "alpha": 0.4, "mse_train": 0.43854148846073054, "mse_val": 0.8787438339745526, "mse_test": 0.900018099526243, "r2_train": 0.5614585115392694, "r2_test": 0.20695963849381838, "corr_train": 0.8249629118347317, "corr_test": 0.4718427232478075}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 2, "alpha": 0.4, "mse_train": 0.4302699871261694, "mse_val": 0.8300194387297998, "mse_test": 0.8373165447010885, "r2_train": 0.5697300128738307, "r2_test": 0.12876414871426656, "corr_train": 0.8313448078987662, "corr_test": 0.3956113638242669}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 3, "alpha": 0.4, "mse_train": 0.4435186538811436, "mse_val": 0.8691957593685344, "mse_test": 0.6997907579108716, "r2_train": 0.5564813461188565, "r2_test": 0.24887262232444696, "corr_train": 0.8233509592942332, "corr_test": 0.517085519640998}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 4, "alpha": 0.4, "mse_train": 0.43974711907420594, "mse_val": 0.8626304071711859, "mse_test": 0.8469478640915832, "r2_train": 0.5602528809257941, "r2_test": 0.21279770998501168, "corr_train": 0.8253299839254685, "corr_test": 0.4704915831511565}
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 2142, "split": 5, "alpha": 0.4, "mse_train": 0.432177284988686, "mse_val": 0.833255714218285, "mse_test": 0.8878615790534914, "r2_train": 0.5678227150113138, "r2_test": 0.11851564557450822, "corr_train": 0.8272552054767326, "corr_test": 0.37040210818386465}
```

Definitely more stable, ok.

One question, why is the test MSE lower and more variable than val MSE? Two possible reasons I can think of: (1) the val MSE is an average over k (5) folds, the test MSE is a single estimate, and (2) the val MSE is results from fitting with a smaller training dataset size (0.64 vs 0.8).

## 2025-05-18

Analyzing results of behavioral prediction. Seems all jobs completed successfully, great.

Initial pass of results, looks like a large subset of SPIs can predict the cognition measures (none reliably better than emprical covariance, as we've seen before). Suprisingly though, it seems no SPI can predict any of the other four factors better than chance. Weird... Will be looking into this more.

## 2025-05-21

Ran a baseline behavioral prediction of all 55 targets + 4 factors using just empirical covariance features. Want to find out which targets are predictable and which aren't.

To test for which are reliably predictable, I want to also do a permutation test.

I've realized also that I'm perhaps spending too much time recomputing the kernel every iteration. I timed fitting a kernel ridge on this scale of data, it takes 500 ms for cosine kernel, and 15 ms for precomputed kernel. So I think probably have to do it.

Also, the implementation of the target transform is overly coupled with the regression model. One concrete challenge is if I shuffle the targets for a permutation test, it breaks the coupling between targets and covariates, which I don't want.

Decided to revise the implementation:

1. Compute kernel once and use sklearn `_safe_split` to correctly do data splitting (`GridSearchCV` already does this.)

2. Move the covariates into targets, and consolidate all target preprocessing in the behavioral target transform estimator.

## 2025-05-22

Testing the updated implementation and checking if I reproduce earlier results. Original implementation.

```
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "seed": 3293, "split": 0, "n_train": 684, "n_test": 183, "alpha": 0.4, "mse_train": 0.42551154966429866, "mse_val": 0.8061305437803726, "mse_test": 0.8511619934191628, "r2_train": 0.5744884503357013, "r2_test": 0.14118010572391992, "corr_train": 0.8252429869810961, "corr_test": 0.38107201487717346}
```

Updated implementation v1.

```
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "n_splits": 20, "perm_test": false, "seed": 3293, "split": 0, "n_train": 684, "n_test": 183, "alpha": 0.4, "mse_train": 0.4255110489727128, "mse_val": 0.7958018914730597, "mse_test": 0.8511616381915343, "r2_train": 0.5744889510272873, "r2_test": 0.14118046414752983, "corr_train": 0.8252433414826901, "corr_test": 0.3810724135667257}
```

The onle glaring difference is the validation loss. Realized I have the target transform outside the grid search. But this means we have leak during CV.

Moving target transform inside `GridSearchCV`.

```
{"spi": "cov_EmpiricalCovariance", "parc_size": 200, "pool": 3, "target": "Cognition", "n_splits": 20, "perm_test": false, "seed": 3293, "split": 0, "n_train": 684, "n_test": 183, "alpha": 0.4, "mse_train": 0.4255110489727128, "mse_val": 0.8061301002483316, "mse_test": 0.8511616381915343, "r2_train": 0.5744889510272873, "r2_test": 0.14118046414752983, "corr_train": 0.8252433414826901, "corr_test": 0.3810724135667257}
```

The results are now the same up to numerical precision. The one thing I've changed with the actual pipeline is now I do nuisance regression on only the required subset of the behaioral columns, rather than all, which is a waste.

I checked that indeed (surprisingly) having extra targets can change the numerical values of the regression coefficients, though only up to numerical precision. Matmuls I guess.

```python
X = np.random.randn(1000, 3)
y = np.random.randn(1000, 10)
ols_1 = LinearRegression().fit(X, y)
ols_2 = LinearRegression().fit(X, y)
ols_3 = LinearRegression().fit(X, y[:, :5])
np.all(ols_1.coef_ == ols_2.coef_)          # True
np.all(ols_1.coef_[:5] == ols_3.coef_)      # False
np.allclose(ols_1.coef_[:5], ols_3.coef_)   # True
```

Worth it, the new implementation is more than 10x faster.
