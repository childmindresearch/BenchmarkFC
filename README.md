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

- `justfile`: list of steps for reproducing the analyses, to be used with [`just`](https://github.com/casey/just).
- `docs/`: project documentation including a log of steps ([`docs/LOG.md`](docs/LOG.md))
- `resources/`: static resource files
  - `column_lists/`: lists of HCP phenotypic column subsets and HCP column dictionary
  - `schaefer_parcellations/`: downloaded Schaefer parcellations
  - `spi_lists/`: lists of PySPI SPI subsets
  - `subject_lists/`: lists of HCP subject subsets.
- `src/`: small package of python utilities
