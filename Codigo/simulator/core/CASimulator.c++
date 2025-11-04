#include <pybind11/pybind11.h> 
#include <pybind11/stl.h>  

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <utility>
#include <algorithm>

std::vector<std::vector<std::vector<int>>> life = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}},
    {{0, 0}}};

namespace py = pybind11;

class CASimulator
{
public:
    CASimulator(int grid_size,
                std::vector<std::vector<std::vector<int>>> neighbourhoods = life,
                int states = 2,
                int rule_number = 6152)
        : grid(grid_size, std::vector<int>(grid_size, 0))
    {
        this->grid_size = grid_size;
        this->neighbourhoods = neighbourhoods;
        this->states = states;
        this->rule_number = rule_number;
        calculateLength();
        setRule(this->rule_number);
    }

    void step()
    {

        std::vector<std::vector<int>> grid = this->grid;

        for (int i = 0; i < grid_size; i++)
        {
            for (int j = 0; j < grid_size; j++)
            {
                std::vector<int> neighbour_pattern;

                // Obtener la vecindad
                for (auto &neighbourhood : neighbourhoods)
                {
                    int total = 0;

                    for (auto &neighbour : neighbourhood)
                    {
                        int nx = i + neighbour[0];
                        int ny = j + neighbour[1];

                        // todo define frontier behaviour
                        if (nx < 0)
                            nx = grid.size() + nx;
                        else if (nx >= grid.size())
                            nx = nx - grid.size();

                        if (ny < 0)
                            ny = grid[nx].size() + ny;
                        else if (ny >= grid.size())
                            ny = ny - grid[nx].size();

                        total += grid[nx][ny];
                    }
                    neighbour_pattern.push_back(total);
                }

                std::reverse(neighbour_pattern.begin(), neighbour_pattern.end());

                this->grid[i][j] = rule[neighbour_pattern];
            }
        }
    }

    void printGrid()
    {
        for (auto &row : grid)
        {
            for (auto &cell : row)
                printf("%d ", cell);
            std::cout << std::endl;
        }
        std::cout << std::endl
                  << std::endl;
    }

    void printNeighbourhood()
    {
        for (auto &&neighbourhood : this->neighbourhoods)
        {
            for (auto &&positions : neighbourhood)
            {
                std::cout << "(";
                for (auto &&pos : positions)
                    std::cout << pos << " ";
                std::cout << ")";
            }
            std::cout << std::endl;
        }
    }

    void printInfo()
    {
        std::cout << "NEIGHBOURHOODS: \n";
        printNeighbourhood();
        std::cout << "STATES: " << this->states << "\n";
        std::cout << "RULE VECTOR: \n";
        for (const auto &[key, val] : rule)
        {
            std::cout << "{";
            for (size_t i = 0; i < key.size(); ++i)
            {
                std::cout << key[i];
                if (i + 1 < key.size())
                    std::cout << ",";
            }
            std::cout << "} -> " << val << "\n";
        }
        std::cout << "\nTotal rules: " << total_rules << "\n";
    }

    // GETTERS Y SETTERS

    std::vector<std::vector<int>> getGrid() const { return grid; }

    void setGrid(std::vector<std::vector<int>> grid) { this->grid = grid; }

    void setRule(int rule_number)
    {
        std::vector<int> digits;

        // Conversión correcta a base 'states'
        int temp = rule_number;
        while (temp > 0)
        {
            digits.insert(digits.begin(), temp % states);
            temp /= states;
        }

        // Rellenar con ceros si falta
        while (digits.size() < rule_size)
            digits.insert(digits.begin(), 0);

        // Cortar si sobra
        if (digits.size() > rule_size)
            digits = std::vector<int>(digits.end() - rule_size, digits.end());

        // Obtener tamaños de vecindades e invertir
        std::vector<int> sizes;
        for (auto &neighbourhood : neighbourhoods)
            sizes.push_back(neighbourhood.size() + 1);
        std::reverse(sizes.begin(), sizes.end());

        // Generar todas las combinaciones
        std::vector<std::vector<int>> combinations;
        std::vector<int> indices(sizes.size(), 0);
        bool done = false;

        while (!done)
        {
            combinations.push_back(indices);
            for (int i = sizes.size() - 1; i >= 0; --i)
            {
                indices[i]++;
                if (indices[i] < sizes[i])
                    break;
                indices[i] = 0;
                if (i == 0)
                    done = true;
            }
        }

        std::reverse(combinations.begin(), combinations.end());

        // Crear mapa regla
        this->rule.clear();
        for (size_t i = 0; i < combinations.size() && i < digits.size(); ++i)
            this->rule[combinations[i]] = digits[i];
        std::cout << "length: " << rule.size() << "\n";
    }

    void setNeighbourhood(std::vector<std::vector<std::vector<int>>> neighbourhoods)
    {
        this->neighbourhoods = neighbourhoods;
        calculateLength();
        setRule(this->rule_number);
    }

    void setStates(int states)
    {
        this->states = states;
        calculateLength();
        setRule(this->rule_number);
    }

private:
    int grid_size;
    std::vector<std::vector<int>> grid;
    int states;
    int rule_number;
    std::map<std::vector<int>, int> rule;
    int rule_size;
    int total_rules;
    std::vector<std::vector<std::vector<int>>> neighbourhoods;

    void calculateLength()
    {
        rule_size = 1;
        for (auto &neighbourhood : neighbourhoods)
            rule_size *= (states - 1) * (neighbourhood.size()) + 1;

        total_rules = std::pow(states, rule_size);
    }
};

PYBIND11_MODULE(CASimulatorCore, m)
{
    py::class_<CASimulator>(m, "CASimulator")
        .def(py::init<int,
                      std::vector<std::vector<std::vector<int>>>,
                      int,
                      int>(),
             py::arg("grid_size"),
             py::arg("neighbourhoods") = std::vector<std::vector<std::vector<int>>>{
                 {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}},
                 {{0, 0}}},
             py::arg("states") = 2,
             py::arg("rule_number") = 6152)
        .def("step", &CASimulator::step)
        .def("getGrid", &CASimulator::getGrid)
        .def("setGrid", &CASimulator::setGrid)
        .def("setRule", &CASimulator::setRule)
        .def("setStates", &CASimulator::setStates)
        .def("setNeighbourhood", &CASimulator::setNeighbourhood)
        .def("printInfo", &CASimulator::printInfo)
        .def("printGrid", &CASimulator::printGrid)
        .def("printNeighbourhood", &CASimulator::printNeighbourhood);
}