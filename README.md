# pyEulerCurves  
A python package to compute Euler Characteristic Curves of point-cloud or image data with an interface à la scikit-learn. 

## Prerequisites

* A compiler with C++11 support
* CMake >= 3.4 or Pip 10+
* Ninja or Pip 10+

## Installation

Just clone this repository and pip install. Note the `--recursive` option which is
needed for the pybind11 submodule:

```bash
git clone --recursive https://github.com/dgurnari/pyEulerCurves.git
pip install ./pyEulerCurves
```

With the `setup.py` file included in this example, the `pip install` command will
invoke CMake and build the pybind11 module as specified in `CMakeLists.txt`.


## Examples
Example notebooks can be found in `examples/` subfolder.
