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
    from pyEulerCurves import ECC_from_pointcloud

    # Create a simple 2D point cloud
    X = np.random.rand(100, 2)

    # Initialize ECC transformer
    trans = ECC_from_pointcloud(epsilon=0.2)

    # Compute ECCs
    ecc = trans.fit_transform(X)

    # ecc_curve is a list of [filtration, EC] pairs

Bitmap Image Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyEulerCurves import ECC_from_bitmap

    # Create random grayscale image
    bitmap = np.random.randint(0, 256, size=(50, 50))

    # Initialize ECC transformer for cubical complex
    trans = ECC_from_bitmap(periodic_boundary=False)

    # Compute ECC
    ecc = trans.fit_transform(bitmap)

Filtered Graph Example
~~~~~~~~~~~~~~~~~~~~~~

pyEulerCurves can also compute the Euler Characteristic Profile (ECP) of the
flag complex of a (multi-)filtered graph. A scalar filtration is represented
as a single float; multifiltrations use tuples of floats.

.. code-block:: python

    from pyEulerCurves import ECP_from_filtered_graph, FilteredGraph

    graph = FilteredGraph(
        vertex_filtrations=[(0.0, 0.0), (0.0, 0.0), (0.1, 0.0)],
        edges=[(0, 1), (1, 2), (0, 2)],
        edge_filtrations=[(0.2, 0.4), (0.5, 0.6), (0.7, 0.8)],
    )

    ecp = ECP_from_filtered_graph().fit_transform(graph)

If NetworkX is installed, the same transformer accepts a ``networkx.Graph``
whose nodes and edges define a ``"filtration"`` attribute.

.. code-block:: python

    import networkx as nx
    from pyEulerCurves import ECP_from_filtered_graph

    graph = nx.Graph()
    graph.add_node("a", filtration=(0.0, 0.0))
    graph.add_node("b", filtration=(0.0, 0.0))
    graph.add_edge("a", "b", filtration=(0.25, 0.5))

    ecp = ECP_from_filtered_graph().fit_transform(graph)

For large graphs, many distinct floating-point filtration values can produce a
large ECP. If you want to compute on a coarser finite filtration poset, coarsen the
vertex and edge filtration values before constructing the ``FilteredGraph``.

Visualizing the ECC
--------------------

You can plot the Euler Characteristic Curve using the helper function:

.. code-block:: python

    from pyEulerCurves import plot_euler_curve

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_euler_curve(ecc, this_ax=ax, with_lines=True)
    plt.show()

Next Steps
----------

- Check out the :doc:`API <API>` reference for a full list of functions and classes.
- Explore the :doc:`Examples <examples/index>`.

Need help? Feel free to open an issue on the GitHub repository or consult the accompanying paper for theoretical background:

*Paweł Dłotko and Davide Gurnari. "Euler characteristic curves and profiles: a stable shape invariant for big data problems." GigaScience 12 (2023).* https://doi.org/10.1093/gigascience/giad094
