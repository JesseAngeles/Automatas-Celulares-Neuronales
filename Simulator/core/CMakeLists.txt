cmake_minimum_required(VERSION 3.15)

# GCC 16 no es soportado por CUDA 13.2; usar GCC 15 como host compiler
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(GCC15 NAMES g++-15)
    if(GCC15)
        set(CMAKE_CUDA_HOST_COMPILER "${GCC15}" CACHE STRING "" FORCE)
    endif()
endif()

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
    "./nca.cu"
    "bindings/python_module.cpp"
)

# Creamos el módulo de python
pybind11_add_module(CA ${SOURCES})

# Enlazar librerias de CUDA
target_link_libraries(CA PRIVATE CUDA::cudart)