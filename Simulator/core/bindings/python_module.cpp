#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "life.h"

namespace py = pybind11;

PYBIND11_MODULE(core, m)
{
    py::class_<Life>(m, "Life")
        .def(py::init<int, int, std::vector<std::vector<int>>>())
        
        .def("step", &Life::step)
        .def("print", &Life::print)
        .def("get_space", &Life::getSpace)
        .def("get_width", &Life::getWidth)
        .def("get_height", &Life::getHeight)
        .def("get_cell", &Life::getCell)
        .def("set_cell", &Life::setCell)
        .def("set_space", &Life::setSpace);
}