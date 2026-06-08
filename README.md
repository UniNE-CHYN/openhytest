# OpenHyTest

OpenHyTest is a Python toolbox for hydraulic test interpretation. It provides
data preparation utilities, analytical well-test models, nonlinear fitting,
diagnostic plots, and report generation for pumping, interference, multirate,
double-porosity, generalized radial flow, and slug-test analyses.

The project is inspired by functionality available in
[hytool](https://github.com/UniNE-CHYN/hytool).

## Functionality

- Prepare time-drawdown and time-pressure data with filtering, cleaning,
  selection, and diagnostic plotting tools.
- Calculate logarithmic derivatives with direct, spline, Bourdet, and
  Horne-style approaches.
- Transform variable-rate tests with the Birsoy and Summers equivalent-time
  method.
- Estimate initial model parameters, run trial curves, fit analytical models,
  and calculate hydraulic parameters such as transmissivity and storativity.
- Generate PDF, PNG, or SVG reports from fitted models.
- Explore complete workflows in the `notebooks/` demo files.

## Installation

OpenHyTest is currently developed as a local Python package. The recommended
workflow is to use a dedicated conda environment and install the package in
editable mode from the repository root.

```bash
conda create -n openhytest python=3.13
conda activate openhytest
cd /path/to/openhytest
python -m pip install -e .
```

For notebook use:

```bash
conda activate openhytest
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name openhytest --display-name "Python (openhytest)"
jupyter lab
```

Then open a notebook from `notebooks/` and select the `Python (openhytest)`
kernel.

Check the installation with:

```bash
python -c "import openhytest as ht; print(ht.Theis)"
```

## Analytical Models

- Theis radial flow (1935)
- Theis no-flow boundary
- Theis constant-head boundary (1941)
- Theis multirate tests, including constant-head boundary support
- Hantush-Jacob leaky aquifer (1955)
- Jacob-Lohman constant-head test (1952)
- Eden-Hazel step-test model
- Warren and Root double porosity (1963)
- Barker generalized radial flow (1988)
- Boulton delayed yield (1963)
- Papadopulos-Cooper large-diameter well (1967)
- Agarwal wellbore storage and skin (1970)
- Hvorslev slug test (1957)
- Cooper, Bredehoeft and Papadopulos slug/pulse test (1967)
- Neuzil modified pulse test (1982)
- CDM heat transport model

## Recent Improvements

- Updated package metadata so `python -m pip install -e .` installs the local
  `openhytest` package correctly.
- Added explicit runtime dependencies in `requirements.txt`.
- Added Plotly/Jupyter notebook support requirements to the installation
  documentation.
- Added a Sphinx functionality summary and refreshed installation instructions.
- Updated API documentation to include the current preprocessing class and a
  broader list of analytical model classes.
- Improved Matplotlib compatibility for type-curve plots on newer Matplotlib
  versions.
- Added bounds and validity checks to several fitting workflows to avoid
  non-finite residuals during optimization.
- Improved Barker/GRF and Boulton demo behavior in the current notebooks.

## Documentation

The Sphinx documentation lives in `doc/`.

```bash
conda activate openhytest
cd doc
make html
```

The generated HTML pages are written to `doc/_build/html/`.

## Developer

OpenHyTest is developed by Nathan Dutler.
