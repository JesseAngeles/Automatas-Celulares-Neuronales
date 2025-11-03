#include <iostream>
#include <vector>

std::vector<std::vector<std::vector<int>>> life = {{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}, {{0, 0}}};

class CASimulator
{
public:
    CASimulator(int size,
                std::vector<std::vector<std::vector<int>>> neighbourhoods = life,
                int states = 2)
        : grid(size, std::vector<int>(size, 0))
    {
        // Asignacion de variables
        this->neighbourhoods = neighbourhoods;
        this->states = states;
    }

    void step()
    {
        this->grid[3][3] = 1 - grid[1][1];
        std::vector<int> current_pos = {3, 3};
        std::vector<int> neighbour_pattern;
        
        // Define neighbour_pattern
        for (auto &neighbourhood : neighbourhoods)
        {
            int total = 0; 

            for (auto &neighbour : neighbourhood)
            {
                int nx = current_pos[0] + neighbour[0]; 
                int ny = current_pos[1] + neighbour[1]; 

                // Verifica que esté dentro del rango válido del grid
                if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[nx].size())
                {
                    total += grid[nx][ny];
                    int neighbour_value = grid[nx][ny];
                    std::cout << "(" << nx << "," << ny << ")=" << neighbour_value << " ";
                }
                else
                {
                    std::cout << "(" << nx << "," << ny << ") fuera de rango ";
                }
            }
            neighbour_pattern.push_back(total);
        }

        // Calculate new value
        
    }

    void printGrid()
    {
        for (auto &row : grid)
        {
            for (auto &cell : row)
                printf("%d ", cell);
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void printNeighbourhood()
    {
        for (auto &&neighbourhood : this->neighbourhoods)
        {
            for (auto &&positions : neighbourhood)
            {
                std::cout << "(";
                for (auto &&pos : positions)
                {
                    std::cout << pos << " ";
                }
                std::cout << ")";
            }
            std::cout << std::endl;
        }
    }

    // Getters y Setters
    std::vector<std::vector<int>> getGrid() const
    {
        return grid;
    }

    int getStates() const
    {
        return states;
    }

    std::vector<std::vector<std::vector<int>>> getNeighbourhoods() const
    {
        return neighbourhoods;
    }

private:
    std::vector<std::vector<int>> grid;
    int states;
    // conjuntos totalisticos // vecindades // posiciones relativas
    std::vector<std::vector<std::vector<int>>> neighbourhoods;
};

int main()
{
    CASimulator ca(10);

    ca.printGrid();
    ca.step();
    ca.printGrid();

    ca.printNeighbourhood();

    return 0;
}