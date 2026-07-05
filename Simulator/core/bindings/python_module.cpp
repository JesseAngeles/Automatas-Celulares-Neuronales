#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "ca.h"
#include "nca.h"

namespace py = pybind11;

PYBIND11_MODULE(CA, m)
{
    py::class_<CA>(m, "CA")
        // Constructor
        .def(py::init([](
                          int width, int height,
                          py::array_t<uint8_t, py::array::c_style | py::array::forcecast> space,
                          int states,
                          py::array_t<int, py::array::c_style | py::array::forcecast> neighbourhood,
                          int n_size,
                          py::array_t<int, py::array::c_style | py::array::forcecast> group_sizes,
                          int n_groups,
                          py::array_t<int, py::array::c_style | py::array::forcecast> rule,
                          int r_size)
                      { return new CA(
                            width, height,
                            space.mutable_data(),
                            states,
                            neighbourhood.mutable_data(), n_size,
                            group_sizes.mutable_data(), n_groups,
                            rule.mutable_data(), r_size); }))

        // Sin GIL durante el kernel: el prefetch corre en otro hilo sin bloquear
        // el event loop de la GUI. CA serializa el acceso a GPU con su mutex.
        .def("step", &CA::step, py::call_guard<py::gil_scoped_release>())

        .def("run", [](CA &self, int steps)
             {
        if (steps <= 0)
            throw std::runtime_error("run: steps must be > 0");

        // El array destino lo posee numpy; run() drena su staging pinned
        // directamente aquí (sin capsule ni transferencia de ownership).
        auto result = py::array_t<uint8_t>({steps, self.getHeight(), self.getWidth()});
        uint8_t *out = result.mutable_data();
        {
            py::gil_scoped_release release;
            self.run(steps, out);
        }
        return result; })

        .def("print", &CA::print)

        // Getters
        .def("get_space", [](CA &self)
             {
            return py::array_t<uint8_t>(
                {self.getHeight(), self.getWidth()},
                {self.getWidth() * (ssize_t)sizeof(uint8_t), (ssize_t)sizeof(uint8_t)},
                self.getSpace(),
                py::cast(self)); })

        .def("get_width", &CA::getWidth)
        .def("get_height", &CA::getHeight)
        .def("get_states", &CA::getStates)
        .def("get_n_size", &CA::getNSize)
        .def("get_n_groups", &CA::getNGroups)
        .def("get_r_size", &CA::getRSize)

        .def("get_neighbourhood", [](CA &self)
             { return py::array_t<int>(
                   {self.getNSize(), 2},
                   {2 * (ssize_t)sizeof(int), (ssize_t)sizeof(int)},
                   self.getNeighbourhood(),
                   py::cast(self)); })

        .def("get_group_sizes", [](CA &self)
             { return py::array_t<int>(
                   {self.getNGroups()},
                   {(ssize_t)sizeof(int)},
                   self.getGroupSizes(),
                   py::cast(self)); })

        .def("get_rule", [](CA &self)
             { return py::array_t<int>(
                   {self.getRSize()},
                   {(ssize_t)sizeof(int)},
                   self.getRule(),
                   py::cast(self)); })

        .def("get_cell", &CA::getCell)

        // Setters
        .def("set_space", [](CA &self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> new_space)
             {
            if (new_space.ndim() != 2)
                throw std::runtime_error("set_space: 2D array expected (height, width)");

            if (new_space.shape(0) != self.getHeight() || new_space.shape(1) != self.getWidth())
                throw std::runtime_error("set_space: wrong dimensions");

            self.setSpace(new_space.data()); })

        .def("set_states", &CA::setStates)

        .def("set_neighbourhood", [](CA &self,
                                     py::array_t<int, py::array::c_style | py::array::forcecast> new_neighbourhood,
                                     py::array_t<int, py::array::c_style | py::array::forcecast> new_group_sizes)
             {
            if (new_neighbourhood.ndim() != 2 || new_neighbourhood.shape(1) != 2)
                throw std::runtime_error("set_neighbourhood: 2D array expected (n_size, 2)");
            if (new_group_sizes.ndim() != 1)
                throw std::runtime_error("set_neighbourhood: group_sizes must be 1D");

            self.setNeighbourhood(
                const_cast<int*>(new_neighbourhood.data()), new_neighbourhood.shape(0),
                const_cast<int*>(new_group_sizes.data()), new_group_sizes.shape(0)); })

        .def("set_rule", [](CA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_rule)
             {
            if (new_rule.ndim() != 1)
                throw std::runtime_error("set_rule: 1D array expected");

            self.setRule(const_cast<int * >(new_rule.data())); })

        .def("set_cell", &CA::setCell)

        .def("set_boundary", &CA::setBoundary)
        .def("get_boundary", &CA::getBoundary);

    py::class_<NCA>(m, "NCA")
        // Constructor
        .def(py::init([](
                          int width, int height,
                          py::array_t<float, py::array::c_style | py::array::forcecast> space,
                          py::array_t<int, py::array::c_style | py::array::forcecast> neighbourhood,
                          int n_size)
                      { return new NCA(
                            width, height,
                            space.mutable_data(),
                            neighbourhood.mutable_data(),
                            n_size); }))

        .def("step", &NCA::step, py::call_guard<py::gil_scoped_release>())

        .def("run", [](NCA &self, int steps)
             {
        if (steps <= 0)
            throw std::runtime_error("run: steps must be > 0");

        // float32 de extremo a extremo: sin truncar el estado continuo
        auto result = py::array_t<float>({steps, self.getHeight(), self.getWidth()});
        float *out = result.mutable_data();
        {
            py::gil_scoped_release release;
            self.run(steps, out);
        }
        return result; })

        .def("print", &NCA::print)

        // Getters
        .def("get_space", [](NCA &self)
             { return py::array_t<float>(
                   {self.getHeight(), self.getWidth()},
                   {self.getWidth() * (ssize_t)sizeof(float), (ssize_t)sizeof(float)},
                   self.getSpace(),
                   py::cast(self)); })

        .def("get_width", &NCA::getWidth)
        .def("get_height", &NCA::getHeight)
        .def("get_n_size", &NCA::getNSize)
        .def("get_cell", &NCA::getCell)

        .def("get_neighbourhood", [](NCA &self)
             { return py::array_t<int>(
                   {self.getNSize(), 2},
                   {2 * (ssize_t)sizeof(int), (ssize_t)sizeof(int)},
                   self.getNeighbourhood(),
                   py::cast(self)); })

        // Setters
        .def("set_space", [](NCA &self, py::array_t<float, py::array::c_style | py::array::forcecast> new_space)
             {
            if (new_space.ndim() != 2)
                throw std::runtime_error("set_space: 2D array expected (height, width)");
            if (new_space.shape(0) != self.getHeight() || new_space.shape(1) != self.getWidth())
                throw std::runtime_error("set_space: wrong dimensions");
            self.setSpace(new_space.mutable_data()); })

        .def("set_cell", &NCA::setCell)

        .def("set_neighbourhood", [](NCA &self, py::array_t<int, py::array::c_style | py::array::forcecast> new_neighbourhood)
             {
            if (new_neighbourhood.ndim() != 2 || new_neighbourhood.shape(1) != 2)
                throw std::runtime_error("set_neighbourhood: 2D array expected (n_size, 2)");
            self.setNeighbourhood(new_neighbourhood.mutable_data(), new_neighbourhood.shape(0)); })

        .def("set_weights", [](NCA &self,
                               std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> weights,
                               std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> biases,
                               std::vector<int> layer_in,
                               std::vector<int> layer_out,
                               std::vector<int> activations)
             {
            int n = weights.size();
            std::vector<float *> w_ptrs(n), b_ptrs(n);
            for (int i = 0; i < n; i++) {
                w_ptrs[i] = weights[i].mutable_data();
                b_ptrs[i] = biases[i].mutable_data();
            }
            self.setWeights(w_ptrs.data(), b_ptrs.data(),
                            layer_in.data(), layer_out.data(),
                            activations.data(), n); });
}
