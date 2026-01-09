from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "pyEulerCurves._compute_local_EC_cubical",
        ["pyEulerCurves/src/compute_local_EC_cubical.cpp"],
        include_dirs=["pyEulerCurves/src"],
    ),
    Pybind11Extension(
        "pyEulerCurves._compute_local_EC_VR",
        ["pyEulerCurves/src/compute_local_EC_VR.cpp"],
        include_dirs=["pyEulerCurves/src"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
