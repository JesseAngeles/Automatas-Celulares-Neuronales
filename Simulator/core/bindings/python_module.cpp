#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ca.h"

namespace py = pybind11;

PYBIND11_MODULE(CA, m)
{
    py::class_<CA>(m, "CA")
        // Constructor
        .def(py::init([](
                          int width, int height,
                          py::array_t<int, py::array::c_style | py::array::forcecast> space,
                          int states,
                          py::array_t<int, py::array::c_style | py::array::forcecast> neighbourhood,
                          int n_size,
                          py::array_t<int, py::array::c_style | py::array::forcecast> rule,
                          int r_size)
                      { return new CA(
                            width, height,
                            space.mutable_data(),
                            states,
                            neighbourhood.mutable_data(),
                            n_size,
                            rule.mutable_data(),
                            r_size); }))

        .def("step", &CA::step)

        .def("run", [](CA &self, int steps)
             {
        int *history = self.run(steps);

        py::capsule free_when_done(history, [](void *ptr)
                                   { delete[] reinterpret_cast<int *>(ptr); });

        return py::array_t<int>(
            {steps, self.getHeight(), self.getWidth() },
        {
            self.getHeight() * self.getWidth() * (ssize_t)sizeof(int),
            self.getWidth() * (ssize_t)sizeof(int),
            (ssize_t)sizeof(int)
        },
        history,
        free_when_done
        ); })

        .def("print", &CA::print)

        // Getters
        .def("get_space", [](CA &self)
             {
            auto shape = std::vector<ssize_t>{self.getHeight(), self.getWidth()};
            auto strides = std::vector<ssize_t>{self.getWidth() * (ssize_t)sizeof(int), (ssize_t)sizeof(int)};

            return py::array_t<int>(
                {self.getHeight(), self.getWidth()},
                {self.getWidth() * (ssize_t)sizeof(int), (ssize_t)sizeof(int)},
                self.getSpace(),
                py::cast(self)); })

        .def("get_width", &CA::getWidth)
        .def("get_height", &CA::getHeight)
        .def("get_states", &CA::getStates)
        .def("get_n_size", &CA::getNSize)
        .def("get_r_size", &CA::getRSize)

        .def("get_neighbourhood", [](CA &self)
             { return py::array_t<int>(
                   {self.getNSize(), 2},
                   {2 * (ssize_t)sizeof(int), (ssize_t)sizeof(int)},
                   self.getNeighbourhood(),
                   py::cast(self)); })

        .def("get_rule", [](CA &self)
             { return py::array_t<int>(
                   {self.getRSize()},
                   {(ssize_t)sizeof(int)},
                   self.getRule(),
                   py::cast(self)); })

        .def("get_cell", &CA::getCell)

        // Setters
        .def("set_space", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_space)
             {
            if (new_space.ndim() != 2)
                throw std::runtime_error("set_space: 2D array expected (height, width)");

            if (new_space.shape(0) != self.getHeight() || new_space.shape(1) != self.getWidth())
                throw std::runtime_error("set_space: wrong dimensions");
            
            self.setSpace(const_cast<int*>(new_space.data())); })

        .def("set_states", &CA::setStates)

        .def("set_neighbourhood", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_neighbourhood)
             {
            if (new_neighbourhood.ndim() != 2)
                throw std::runtime_error("set_neighbourhood: 2D array expected (n_size, 2)");

            if (new_neighbourhood.shape(0) != self.getNSize() || new_neighbourhood.shape(1) != 2)
                throw std::runtime_error("set_neighbourhood: wrong dimensions");

            self.setNeighbourhood(const_cast<int*>(new_neighbourhood.data()), new_neighbourhood.shape(0)); })

        .def("set_rule", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_rule)
             {
            if (new_rule.ndim() != 1)
                throw std::runtime_error("set_rule: 1D array expected");
           
            self.setRule(const_cast<int * >(new_rule.data())); })

        .def("set_cell", &CA::setCell);
}