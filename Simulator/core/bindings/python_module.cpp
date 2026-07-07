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
        .def("get_boundary", &CA::getBoundary)

        // Modo estocástico: p = 1.0 mantiene el comportamiento determinista
        .def("set_update_prob", &CA::setUpdateProb)
        .def("get_update_prob", &CA::getUpdateProb)
        .def("set_seed", &CA::setSeed);

    py::class_<NCA>(m, "NCA")
        // Constructor: space (H, W) con channels=1, o (H, W, C) con channels>1.
        // El core almacena el estado en layout SoA channel-major (C, H, W);
        // el binding reempaqueta HWC → CHW en la frontera.
        .def(py::init([](
                          int width, int height,
                          py::array_t<float, py::array::c_style | py::array::forcecast> space,
                          py::array_t<int, py::array::c_style | py::array::forcecast> neighbourhood,
                          int n_size,
                          int channels)
                      {
            if (channels == 1)
                return new NCA(width, height, space.mutable_data(),
                               neighbourhood.mutable_data(), n_size, 1);

            if (space.ndim() != 3 || space.shape(2) != channels)
                throw std::runtime_error("NCA: 3D array expected (height, width, channels)");
            const float *src = space.data();
            const size_t n_cells = (size_t)width * height;
            std::vector<float> chw((size_t)channels * n_cells);
            for (size_t cell = 0; cell < n_cells; cell++)
                for (int c = 0; c < channels; c++)
                    chw[(size_t)c * n_cells + cell] = src[cell * channels + c];
            return new NCA(width, height, chw.data(),
                           neighbourhood.mutable_data(), n_size, channels); }),
             py::arg("width"), py::arg("height"), py::arg("space"),
             py::arg("neighbourhood"), py::arg("n_size"), py::arg("channels") = 1)

        .def("step", &NCA::step, py::call_guard<py::gil_scoped_release>())

        .def("run", [](NCA &self, int steps) -> py::object
             {
        if (steps <= 0)
            throw std::runtime_error("run: steps must be > 0");

        const int H = self.getHeight(), W = self.getWidth(), C = self.getChannels();

        // float32 de extremo a extremo: sin truncar el estado continuo.
        // El core escribe frames contiguos (C, H, W); con C > 1 se devuelve
        // una vista transpuesta (steps, H, W, C) sin copiar.
        if (C == 1)
        {
            auto result = py::array_t<float>({steps, H, W});
            float *out = result.mutable_data();
            {
                py::gil_scoped_release release;
                self.run(steps, out);
            }
            return result;
        }
        auto result = py::array_t<float>({steps, C, H, W});
        float *out = result.mutable_data();
        {
            py::gil_scoped_release release;
            self.run(steps, out);
        }
        return result.attr("transpose")(0, 2, 3, 1); })

        .def("print", &NCA::print)

        // Getters
        .def("get_space", [](NCA &self) -> py::object
             {
        const int H = self.getHeight(), W = self.getWidth(), C = self.getChannels();
        if (C == 1)
            return py::array_t<float>(
                {H, W},
                {W * (ssize_t)sizeof(float), (ssize_t)sizeof(float)},
                self.getSpace(),
                py::cast(self));
        // Vista (H, W, C) con strides sobre el buffer CHW del host (sin copia)
        return py::array_t<float>(
            {H, W, C},
            {W * (ssize_t)sizeof(float), (ssize_t)sizeof(float),
             (ssize_t)H * W * (ssize_t)sizeof(float)},
            self.getSpace(),
            py::cast(self)); })

        .def("get_width", &NCA::getWidth)
        .def("get_height", &NCA::getHeight)
        .def("get_channels", &NCA::getChannels)
        .def("get_n_size", &NCA::getNSize)
        .def("get_cell", &NCA::getCell,
             py::arg("x"), py::arg("y"), py::arg("channel") = 0)

        .def("get_neighbourhood", [](NCA &self)
             { return py::array_t<int>(
                   {self.getNSize(), 2},
                   {2 * (ssize_t)sizeof(int), (ssize_t)sizeof(int)},
                   self.getNeighbourhood(),
                   py::cast(self)); })

        // Setters
        .def("set_space", [](NCA &self, py::array_t<float, py::array::c_style | py::array::forcecast> new_space)
             {
            const int H = self.getHeight(), W = self.getWidth(), C = self.getChannels();
            if (C == 1)
            {
                if (new_space.ndim() != 2)
                    throw std::runtime_error("set_space: 2D array expected (height, width)");
                if (new_space.shape(0) != H || new_space.shape(1) != W)
                    throw std::runtime_error("set_space: wrong dimensions");
                self.setSpace(new_space.mutable_data());
                return;
            }
            if (new_space.ndim() != 3)
                throw std::runtime_error("set_space: 3D array expected (height, width, channels)");
            if (new_space.shape(0) != H || new_space.shape(1) != W || new_space.shape(2) != C)
                throw std::runtime_error("set_space: wrong dimensions");
            const float *src = new_space.data();
            const size_t n_cells = (size_t)W * H;
            std::vector<float> chw((size_t)C * n_cells);
            for (size_t cell = 0; cell < n_cells; cell++)
                for (int c = 0; c < C; c++)
                    chw[(size_t)c * n_cells + cell] = src[cell * C + c];
            self.setSpace(chw.data()); })

        .def("set_cell", &NCA::setCell,
             py::arg("x"), py::arg("y"), py::arg("value"), py::arg("channel") = 0)

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
                            activations.data(), n); })

        // Modo estocástico: p = 1.0 mantiene el comportamiento determinista
        .def("set_update_prob", &NCA::setUpdateProb)
        .def("get_update_prob", &NCA::getUpdateProb)
        .def("set_seed", &NCA::setSeed);
}
