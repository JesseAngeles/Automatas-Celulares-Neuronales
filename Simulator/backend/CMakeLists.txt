cmake_minimum_required(VERSION 3.15)
project(ca_backend)

set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_dir})

find_package(pybind11 REQUIRED)

file(GLOB_RECURSE SOURCES
    core/*.cpp
    engine/*.cpp
    rules/*.cpp
    bindings/*.cpp
)

pybind11_add_module(ca_backend ${SOURCES})