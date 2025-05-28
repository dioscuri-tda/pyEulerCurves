# pyEulerCurves
[![PyPI version](https://img.shields.io/pypi/v/pyEulerCurves.svg?color=blue)](https://pypi.org/project/pyEulerCurves)
[![Documentation Status](https://readthedocs.org/projects/pyeulercurves/badge/?version=latest)](https://pyeulercurves.readthedocs.io/en/latest/?badge=latest)

**pyEulerCurves** is a Python package for computing **Euler Characteristic Curves (ECC)** from point cloud or image data. It provides an intuitive, à la scikit-learn interface and supports fast, parallel computations thanks to its C++ backend. ECCs are powerful topological signatures useful in shape analysis, computer vision, and machine learning.


## 📦 Installation

Install from PyPI:

```bash
pip install pyeulercurves
````

Or install from source:

```bash
git clone https://github.com/dioscuri-tda/pyEulerCurves.git
pip install ./pyEulerCurves
```

## 📘 Documentation

Full documentation is available on [ReadTheDocs](https://pyeulercurves.readthedocs.io).


## 🧪 Examples

Interactive Jupyter notebooks are available in the [examples folder](https://github.com/dioscuri-tda/pyEulerCurves).



## 📚 Citation

If you use **pyEulerCurves** in your research, please cite  [our paper](https://doi.org/10.1093/gigascience/giad094)

```bibtex
@article{10.1093/gigascience/giad094,
    author = {Dłotko, Paweł and Gurnari, Davide},
    title = {Euler characteristic curves and profiles: a stable shape invariant for big data problems},
    journal = {GigaScience},
    volume = {12},
    pages = {giad094},
    year = {2023},
    doi = {10.1093/gigascience/giad094}
}
```