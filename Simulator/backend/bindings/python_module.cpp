#include <pybind11/pybind11.h>

#include "../core/Grid.h"
#include "../engine/Automaton.h"
#include "../rules/LifeRule.h"

namespace py = pybind11;

PYBIND11_MODULE(ca_backend, m)
{
    py::class_<Grid>(m, "Grid")
        .def("get_width", &Grid::getWidth)
        .def("get_height", &Grid::getHeight)
        .def("get", &Grid::get)
        .def("set", &Grid::set);

    py::class_<Automaton>(m, "Automaton")
        .def(py::init([](int w, int h) {
            return new Automaton(
                w,
                h,
                std::make_unique<LifeRule>()
            );
        }))
        .def("step", &Automaton::Step)
        .def("get_grid", &Automaton::getGrid,
             py::return_value_policy::reference);
}