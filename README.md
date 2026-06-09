# BenchmarkFC

### Benchmarking the multiverse of methods: a case study in functional connectivity

> 🧠 **Presenting at OHBM 2026 ([poster](https://drive.google.com/file/d/1iqLxeIFOI9q-kucnyhsPhxxY-IViYMJe)).** Help us turn these experiments into a general method-selection tool for neuroimaging data! Feedback, issues, and contributions welcome.

## Overview

Contemporary neuroimaging offers many high-quality pipelines for functional connectivity (FC) analysis, but this methodological diversity can yield **divergent results** and **hinder reproducibility**. Researchers often lack a clear, data-driven process for selecting an FC method, relying instead on narrow comparisons or risking circular logic by aligning to external phenotypes.

**BenchmarkFC** is a benchmarking suite that evaluates and compares FC estimators against a broad battery of standardized benchmarks, providing **objective criteria for data-driven method selection**. Our long-term goal is to grow this research codebase into a general **method-selection tool for neuroimaging**.

We currently evaluate two families of FC estimators:

- [**PySPI**](https://github.com/DynamicsAndNeuralSystems/pyspi) — statistics of pairwise interaction (SPIs).
- [**skarf**](https://github.com/childmindresearch/skarf) — (auto)regressive functional connectivity estimators.

## Benchmarks

We group benchmarks into two complementary categories:

### Extrinsic — alignment with external data or established biological priors
- Behavioral Prediction
  - Cognition
    - Pearson's r
    - R-squared
- Demographic Prediction
  - Sex
    - Accuracy
  - Age
    - Pearson's r
    - MAE
- Homotopic FC (WIP)
  - Ranked Mean
- Weight Distance (WIP)
  - Spearman's rho
- Structure Function (WIP)
  - R-squared

### Intrinsic — inherent mathematical and statistical properties
- Structural Validity
  - Complexity
    - Single Value Entropy
    - Stable Rank
  - Topology
    - Small-worldness
    - Rich Club Coefficient
  - Efficiency
    - Traveling Salesman Problem Cost
  - Hierarchy
    - Trophic Incoherence
    - Core Depth
- Stability
  - Edgewise Reliability
    - ICC2
  - Network Structure
    - Gradient Similarity
  - Robustness (WIP)
    - Number of TRs
    - Width of TRs
  - Discriminability
    - Subject Identifiability Index
    - Discriminability Statistic

## Data

We use 3T resting-state fMRI from the minimally preprocessed **HCP S1200** release, parcellated with the **Schaefer 2018 (200-parcel)** atlas. 

## Project structure

The entrypoint for the project is the [`justfile`](justfile). It lists the full sequence of steps for reproducing the experiments and doubles as a table of contents for the project.

- `justfile`: list of steps for reproducing the analyses, to be used with [`just`](https://github.com/casey/just).
- `docs/`: project documentation
  - [`LOG.md`](docs/LOG.md): log of daily steps
  - [`METHODS.md`](docs/METHODS.md): writeup of methods
  - [`TODO.md`](docs/TODO.md): possible next steps
- `data/`: input and intermediate preprocessed data
- `results/`: output results and figures
- `scripts/`: high-level data processing scripts
- `src/`: small package of python utilities shared across scripts
- `submodules/`: external packages included as submodules
  - `skarf/`: skarf package submodule
- `notebooks/`: jupyter notebooks for analyzing results and making figures
- `resources/`: static resource files
  - `column_lists/`: lists of HCP phenotypic column subsets and HCP column dictionary
  - `schaefer_parcellations/`: downloaded Schaefer parcellations
  - `spi_lists/`: lists of PySPI SPI subsets
  - `subject_lists/`: lists of HCP subject subsets
- `logs/`: slurm job logs
- `.scratch/`: random misc junk files

## Reproducing

1. Clone the repository

```sh
git clone git@github.com:childmindresearch/BenchmarkFC.git
cd BenchmarkFC
git submodule init && git submodule update
```

2. Install the environment with [`uv`](https://docs.astral.sh/uv/)

```sh
uv sync
```

3. Run the commands in the justfile step by step

```sh
just download_schaefer
just download_hcp_1200
just download_misc_files
just compute_hcp_1200_rfmri_fd
...
```

We recommend reading the code to understand what's happening and what outputs are expected, and monitoring the output of each step to ensure everything runs correctly.

## Contributions

BenchmarkFC began as a series of experiments in functional connectivity. We are working toward a general, extensible **method-selection tool** for neuroimaging. We are still in the early stages of planning, so all contributions and suggestions are welcome! Please open an issue to start a discussion.