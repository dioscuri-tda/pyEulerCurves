cmake_minimum_required(VERSION 2.8.12)
project(pyEulerCurves)

add_subdirectory(pybind11)
pybind11_add_module(_compute_local_EC_VR src/compute_local_EC_VR.cpp)
pybind11_add_module(_compute_local_EC_cubical src/compute_local_EC_cubical.cpp)
