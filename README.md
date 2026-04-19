# Brain Tumor Classification

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-111111.svg)](./LICENSE)
[![Tests](https://github.com/crisdanrodriguez/brain_tumor_classification/actions/workflows/tests.yml/badge.svg)](https://github.com/crisdanrodriguez/brain_tumor_classification/actions/workflows/tests.yml)

Handcrafted texture-feature extraction and stacking-based brain tumor classification on MRI images.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)
- [AI Assistance and Last Updated](#ai-assistance-and-last-updated)

## Overview

This repository is a machine learning and applied data science project focused on classifying brain tumor MRI images into four classes:

- `no_tumor`
- `glioma_tumor`
- `meningioma_tumor`
- `pituitary_tumor`

The project works with engineered image features instead of deep learning. It includes:

- A feature extraction pipeline based on first-order statistics and GLCM texture descriptors
- A stacking implementation built with `scikit-learn`
- A manual stacking implementation that recreates the ensemble logic from scratch
- An exploratory notebook for analysis of the engineered dataset

The repository currently contains the image dataset under [`data/`](./data), the derived feature table at [`data/brain_tumor_dataset.csv`](./data/brain_tumor_dataset.csv), the Python package under [`src/`](./src), complementary material under [`docs/`](./docs), and basic automated tests under [`tests/`](./tests).

## Installation

### Requirements

- Python `3.10+`
- `pip`

### Setup

```bash
git clone https://github.com/crisdanrodriguez/brain_tumor_classification.git
cd brain_tumor_classification

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Generate the engineered feature dataset

Rebuilds [`data/brain_tumor_dataset.csv`](./data/brain_tumor_dataset.csv) from the training images in [`data/Training`](./data/Training).

```bash
python -m brain_tumor_classification.feature_extraction
```

### Run the scikit-learn stacking benchmark

Evaluates the baseline models and the stacking ensemble with repeated stratified cross-validation.

```bash
python -m brain_tumor_classification.stacking_sklearn
```

### Run the manual stacking implementation

Runs the from-scratch ensemble workflow and opens an interactive prompt for predicting a new image path.

```bash
python -m brain_tumor_classification.stacking_from_scratch
```

### Open the exploratory notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

## Project Structure

```text
brain_tumor_classification/
├── .github/                         # CI workflow and collaboration templates
├── data/                            # MRI images and engineered feature dataset
├── docs/
│   ├── documentation/               # Technical report and reference material
│   └── results/                     # Presentation assets and project outputs
├── notebooks/                       # Exploratory analysis notebook
├── src/brain_tumor_classification/  # Main Python package
├── tests/                           # Basic smoke tests
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Results

The project compares three base learners, `KNN`, `Decision Tree`, and `Gaussian Naive Bayes`, against a stacking ensemble built on the same engineered feature space.

Relevant result material:

- [Project presentation](./docs/results/btc_presentation.pptx)

## Documentation

Additional technical and explanatory material is grouped under [`docs/`](./docs) and [`notebooks/`](./notebooks):

- [Technical report](./docs/documentation/brain_tumor_classification_with_stacking_method.pdf)
- [Implementation diagram](./docs/documentation/implementation_diagram.png)
- [EDA notebook](./notebooks/eda.ipynb)
- [Towards AI article](https://towardsai.net/p/l/stacking-ensemble-method-for-brain-tumor-classification-performance-analysis)

## Development

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Notes

- The repository keeps the current dataset inside version control because it is part of the project deliverable.
- Feature extraction is intentionally separated from module import time, so tests and CI can import the package safely.
- The `requirements.txt` file installs the local package plus notebook-oriented extras defined in [`pyproject.toml`](./pyproject.toml).

## License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## AI Assistance and Last Updated

This repository includes AI-assisted improvements focused on refactoring, documentation cleanup, test scaffolding, and repository presentation. The model-assisted work was used to improve maintainability and consistency, while keeping the project aligned with its existing scope.

Last updated: April 19, 2026.
