#include "bind/lib/python3.14/site-packages/pybind11/include/pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ca.h"

namespace py = pybind11;

PYBIND11_MODULE(CA, m)
{
    py::class_<CA>(m, "CA")
        // Constructor
        .def(py::init([](int width, int height, py::array_t<int> space, int states,
                         py::array_t<int> neighbourhood, int n_size,
                         py::array_t<int> rule, int r_size)
                      { return new CA(width, height,
                                      static_cast<int *>(space.mutable_data()),
                                      states,
                                      static_cast<int *>(neighbourhood.mutable_data()),
                                      n_size,
                                      static_cast<int *>(rule.mutable_data()),
                                      r_size); }))

        .def("step", &CA::step)
        .def("print", &CA::print)

        // Getters
        .def("get_space", [](CA &self)
             {
            auto shape = std::vector<ssize_t>{ self.getHeight(), self.getWidth() };
            auto strides = std::vector<ssize_t>{ self.getWidth() * (ssize_t)sizeof(int), (ssize_t) sizeof(int) };

            return py::array_t<int>(shape, strides, self.getSpace(), py::cast(self)); })

        .def("get_width", &CA::getWidth)
        .def("get_height", &CA::getHeight)
        .def("get_states", &CA::getStates)

        .def("get_neighbourhood", [](CA &self)
             {
            auto shape = std::vector<ssize_t>{ self.getNSize(), 2 };
            auto strides = std::vector<ssize_t>{ 2 * (ssize_t)sizeof(int), (ssize_t) sizeof(int) };

            return py::array_t<int>(shape, strides, self.getNeighbourhood(), py::cast(self)); })

        .def("get_rule", [](CA &self)
             {
            auto shape = std::vector<ssize_t>{ self.getRSize() };
            auto strides = std::vector<ssize_t>{ (ssize_t)sizeof(int) };

            return py::array_t<int>(shape, strides, self.getRule(), py::cast(self)); })

        .def("get_cell", &CA::getCell)

        // Setters
        .def("set_space", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_space) {
            if (new_space.ndim() != 2)
                throw std::runtime_error("set_space");
            
            if (new_space.shape(0) != self.getHeight() || new_space.shape(1) != self.getWidth())
                throw std::runtime_error("set_space");
            
            std::memcpy(self.getSpace(), new_space.data(), self.getWidth() * self.getHeight() * sizeof(int));
        })

        .def("set_states", &CA::setStates)

        .def("set_neighbourhood", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_neighbourhood) {
            if (new_neighbourhood.ndim() != 2)
                throw std::runtime_error("set_neighbourhood");
            
            if (new_neighbourhood.shape(0) != self.getNSize() || new_neighbourhood.shape(1) != 2)
                throw std::runtime_error("set_neighbourhood");
            
            std::memcpy(self.getNeighbourhood(), new_neighbourhood.data(), self.getNSize() * 2 * sizeof(int));
        })

        .def("set_rule", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_rule) {
            if (new_rule.ndim() != 1)
                throw std::runtime_error("set_rule");
            
            if (new_rule.shape(0) != self.getRSize())
                throw std::runtime_error("set_rule");
            
            std::memcpy(self.getRule(), new_rule.data(), self.getRSize() * sizeof(int));
        })

        .def("set_cell", &CA::setCell);
}