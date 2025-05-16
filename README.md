# Meta-Feature Selector

**Meta-Feature Selector** is a Python package for selecting a subset of the most informative features from a meta-dataset. It is particularly useful in meta-learning settings where selecting relevant meta-features improves performance across a wide range of tasks or datasets.

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

- **Example Pipelines**
  - Ready-to-use examples located in:
    - `ms/pipeline/pipeline.py`

- **PILOT Algorithm**
  - Integrated implementation of the PILOT algorithm from [InstanceSpace](https://github.com/andremun/InstanceSpace).
---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/martilut/meta-select.git
cd meta-select
pip install -r requirements.txt
```
