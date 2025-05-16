# Meta-Feature Selector

**Meta-Feature Selector** is a Python package for selecting a subset of the most informative features from a meta-dataset.

---
## Features

This package provides tools for:

- **Meta-Dataset Construction**
  - Use results from [TabZilla](https://github.com/naszilla/tabzilla) to build a meta-dataset.
  - See `ms/metadata_creation.ipynb` for an example.

- **Meta-Feature Selection**
  - Implemented in `ms/selection/`.
  - Includes a feature selection method based on causal analysis located at:
    - `ms/selection/selectors/causal`

- **Experiment Pipeline**
  - Ready-to-use pipeline located in:
    - `ms/pipeline/pipeline.py`

- **PILOT Algorithm**
  - Integrated implementation of the PILOT algorithm from [InstanceSpace](https://github.com/andremun/InstanceSpace).
---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/martilut/meta-select.git
cd meta-select
pip install -r requirements.txt
```

## Usage
To use examples:
1. Run the notebook `ms/metadata_creation.ipynb` to create a meta-dataset.
2. Run the notebook `ms/metadata_analysis.ipynb` to perform meta-feature selection and meta-learning.