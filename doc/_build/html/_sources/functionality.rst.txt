Functionality
=============

OpenHyTest is a Python toolbox for interpreting hydraulic tests. It combines
data preparation tools, analytical well-test models, model fitting, diagnostic
plots, and report generation in a single package.

Data preparation
----------------

The ``preprocessing`` tools help prepare time-drawdown or time-pressure data
before model fitting. The main capabilities are:

* identifying the time and observation columns in a data frame;
* calculating logarithmic derivatives using direct, spline, Bourdet, or
  Horne-style approaches;
* plotting diagnostic drawdown and derivative curves;
* selecting, filtering, and cleaning hydraulic test data;
* calculating equivalent time for variable-rate tests using the Birsoy and
  Summers method;
* transforming recovery data to an equivalent pumping-test representation.

Analytical models
-----------------

The model library includes classical pumping-test, interference-test,
variable-rate, double-porosity, generalized radial flow, wellbore-storage,
skin-effect, and slug-test solutions. Available model families include:

* Theis radial flow;
* Theis models with no-flow and constant-head boundaries;
* Theis multirate models, including a constant-head boundary variant;
* Hantush-Jacob leaky aquifer model;
* Jacob-Lohman constant-head test model;
* Eden-Hazel step-test model;
* Warren and Root double-porosity model;
* Barker generalized radial flow model;
* Boulton delayed-yield model;
* Papadopulos-Cooper large-diameter well model;
* Agarwal wellbore-storage and skin model;
* Hvorslev, Neuzil, and Cooper slug-test models;
* CDM heat transport model.

Fitting and diagnostics
-----------------------

Most analytical models provide a common workflow:

* estimate initial parameters with ``guess_params()``;
* inspect the model with ``trial()``;
* fit parameters with ``fit()`` using SciPy least-squares methods;
* calculate hydraulic parameters such as transmissivity and storativity;
* compare measured drawdown and derivative data with the fitted model;
* generate PDF, PNG, or SVG reports with ``rpt()``.

Examples
--------

The ``notebooks/`` folder contains demonstration notebooks for the main model
families. These notebooks are the best starting point for learning the practical
workflow, including preprocessing, parameter estimation, model fitting, and
report generation.
