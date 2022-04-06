cmake_minimum_required(VERSION 3.0.0)
project(cvpr VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11/bin/nvcc")

add_subdirectory(core/pcl)
add_subdirectory(core/pybind11)
pybind11_add_module(cvpr core/cvpr.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
