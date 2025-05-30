# skarf experiments

## Overview

This repository includes experiment code for evaluating various measures of functional connectivity. We evaluate two classes of functional connectivity:

- [PySPI](https://github.com/DynamicsAndNeuralSystems/pyspi) statistics of pairwise interaction (SPIs).
- [skarf](https://github.com/childmindresearch/skarf) autoregressive functional connectivity estimators.

We evaluate these functional connectivity measures using the following metrics:

- Behavioral trait prediction
- Test-retest reliability
- Inter-subject discriminability
- Autoregressive time-series modeling

The entrypoint for the project, i.e. first file to look at, is the [`justfile`](justfile). This file includes the full sequence of steps for reproducing the experiments. As a result, it can also be seen as a kind of table of contents. Each command should be meaningfully named and commented as to what it does.

## Project structure

- `justfile`: list of steps for reproducing the analyses, to be used with [`just`](https://github.com/casey/just). Can be also used as a table of contents for the project.
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
  - `subject_lists/`: lists of HCP subject subsets.
- `logs/`: slurm job logs
- `.scratch/`: random misc junk files

## Reproducing

1. Clone the repository

```sh
git clone git@github.com:childmindresearch/skarf-experiments.git
```

2. Install the environment with [`uv`](https://docs.astral.sh/uv/)

```sh
uv sync
```

3. Run the commands in the just file step by step

```sh
just download_schaefer
just download_hcp_1200
just download_misc_files
just compute_hcp_1200_rfmri_fd
...
```

I also recommend reading the code to understand what's happening and what outputs are expected, and monitoring the output of each step to ensure everything runs correctly.
