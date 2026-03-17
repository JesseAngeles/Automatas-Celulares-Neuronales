cmake_minimum_required(VERSION 3.15)
project(CA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Buscamos archivos .cpp y .cu
file(GLOB_RECURSE SOURCES
    "./ca.cu"
    "bindings/python_module.cpp"
)

# Creamos el módulo de python
pybind11_add_module(CA ${SOURCES})

# Enlazar librerias de CUDA
target_link_libraries(CA PRIVATE CUDA::cudart)