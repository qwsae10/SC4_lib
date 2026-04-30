# scintkit

[WORK IN PROGRESS]

Simple tools for working with ScintPi and GNSS scintillation data.

This repository contains utilities and pipelines for:

- converting raw data to Parquet
- adding derived products
- running processing workflows

## Project Layout

- `src/scintkit/reading/` - binary file readers and data loading utilities
- `src/scintkit/preprocessing/` - data preprocessing and formatting utilities
- `src/scintkit/services/` - core processing functions (compute, conversion, phase detrending)
- `src/scintkit/utils/` - misc utility functions and helpers
- `src/scintkit/pipelines/` - end-to-end pipeline scripts for multi-level data processing on server
- `tests/` - test and example scripts

## Installation

Recommended (editable install for development):

Run from the repository root:

```bash
python -m pip install -e .
```

This makes the package importable as `scintkit` during development. If you do not want to install the package, you can add the `src` directory to `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Usage

You can import and call the compute functions directly. Example usage:

```python
import pandas as pd
from scintkit.services import compute

# load a full-rate dataframe (example)
df = pd.read_parquet("data/fullrate.parquet")

# compute derived products (adds columns like tec12, sigma_phi_1, s4_1, ...)
df_with_products = compute.add_products(df, verbose=True)
```

## Tutorial

'/tests/compare_oct11.ipynb' contains helpful instructions on how to process raw


