#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  

#include <vector>

namespace py = pybind11;

class CASimulator {
public:
    CASimulator(int size)
        : grid(size, std::vector<int>(size, 0))
    {
        // Inicializa con un patrón simple
        if (size >= 3) {
            grid[size / 2][size / 2] = 1;
            grid[size / 2 - 1][size / 2] = 1;
            grid[size / 2][size / 2 - 1] = 1;
        }
    }

    void step() {
        // Ejemplo tonto: invierte las celdas
        for (auto& row : grid)
            for (auto& c : row)
                c = 1 - c;
    }

    std::vector<std::vector<int>> get_grid() const {
        return grid;
    }

private:
    std::vector<std::vector<int>> grid;
    int states;
    std::vector<std::vector<std::vector<int>>> neighbourhood;
};

PYBIND11_MODULE(CASimulatorCore, m) {
    py::class_<CASimulator>(m, "CASimulator")
        .def(py::init<int>())
        .def("step", &CASimulator::step)
        .def("get_grid", &CASimulator::get_grid);
}
