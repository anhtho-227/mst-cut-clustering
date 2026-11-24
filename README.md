# MST-Cut clustering - Step 1: data

Research code for a MILP model that cuts a given minimum spanning tree (MST) for clustering.
This repository currently contains **step 1 only: dataset access and selection**.

## Setup
```
git clone --depth 1 https://github.com/gagolews/clustering-data-v1.git data/clustering-data-v1
pip install -r requirements.txt
python scripts/make_catalog.py          # writes results/catalog.csv
```

## Contents
- `mstcut/data.py` - `load`, `catalog`, `tier`, `add_uniform_noise`, synthetic generators (S1–S3, `gen_bridge_k`).
- `scripts/make_catalog.py` - keeps datasets with n <= 1000, d <= 3 and flags their role
  (`rg1_noise`, `rg2_imbalance`, `control`) and tier.

Benchmark data: M. Gagolewski, *A framework for benchmarking clustering algorithms*, SoftwareX 20 (2022) 101270,
https://github.com/gagolews/clustering-data-v1 (not included here; cloned into `data/`).
