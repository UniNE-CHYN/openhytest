Installation
============

OpenHyTest is currently developed as a local Python package. The most reliable
way to use it is to clone the repository, create a dedicated Python environment,
and install the package in editable mode.

Conda environment
-----------------

Create and activate an environment:

.. code-block:: bash

   conda create -n openhytest python=3.13
   conda activate openhytest

Install OpenHyTest from the repository root:

.. code-block:: bash

   cd /path/to/openhytest
   python -m pip install -e .

The editable install keeps the Python package linked to the local source tree.
Changes made in ``openhytest/`` are therefore available immediately after
restarting Python or the notebook kernel.

Notebook support
----------------

The demo notebooks use Plotly, Matplotlib, and a Jupyter kernel. To run the
notebooks from the same environment, install Jupyter support and register the
kernel:

.. code-block:: bash

   conda activate openhytest
   python -m pip install jupyterlab ipykernel
   python -m ipykernel install --user --name openhytest --display-name "Python (openhytest)"

Then start Jupyter from the repository root:

.. code-block:: bash

   jupyter lab

Open a notebook from the ``notebooks/`` folder and select the
``Python (openhytest)`` kernel.

Quick import check
------------------

After installation, verify that the package imports:

.. code-block:: bash

   python -c "import openhytest as ht; print(ht.Theis)"

If this command prints the ``Theis`` class, the package is available in the
active environment.
