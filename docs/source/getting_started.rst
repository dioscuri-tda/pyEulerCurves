Getting Started with pyEulerCurves
===================================

Welcome to **pyEulerCurves** – a fast and parallel tool for computing Euler Characteristic Curves (ECC) from point clouds and bitmap images.

This guide will help you get started using the library in just a few steps.

Installation
------------

To install the latest version of pyEulerCurves from PyPI:

.. code-block:: bash

    pip install pyEulerCurves

Or, if you're developing locally, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/dioscuri-tda/pyEulerCurves.git
    cd pyEulerCurves
    pip install -e .

Basic Usage
-----------

pyEulerCurves can compute Euler characteristic curves from both **point cloud** data and **bitmap images**.

Point Cloud Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyEulerCurves.pointcloud import ECC_from_pointcloud

    # Create a simple 2D point cloud
    X = np.random.rand(100, 2)

    # Initialize ECC transformer
    ecc = ECC_from_pointcloud(epsilon=0.1)

    # Compute ECCs
    ecc_curves = ecc.fit_transform(X)

    # ecc_curve is a list of [filtration, EC] pairs

Bitmap Image Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyEulerCurves.bitmap import ECC_from_bitmap

    # Create a binary image
    bitmap = np.random.randint(0, 2, size=(50, 50))

    # Initialize ECC transformer for cubical complex
    ecc_bitmap = ECC_from_bitmap(periodic_boundary=False)

    # Compute ECC
    ecc_curve = ecc_bitmap.fit_transform(bitmap)

Visualizing the ECC
--------------------

You can plot the Euler Characteristic Curve using the helper function:

.. code-block:: python

    from pyEulerCurves.utils import plot_euler_curve

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_euler_curve(ecc_curve[0], this_ax=ax, with_lines=True)
    plt.show()

Next Steps
----------

- Check out the :doc:`API <API>` reference for a full list of functions and classes.
- Explore the :doc:`Examples <examples/index>`.

Need help? Feel free to open an issue on the GitHub repository or consult the accompanying paper for theoretical background:

*Paweł Dłotko and Davide Gurnari. "Euler characteristic curves and profiles: a stable shape invariant for big data problems." GigaScience 12 (2023).* https://doi.org/10.1093/gigascience/giad094
