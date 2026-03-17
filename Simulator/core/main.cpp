#include <iostream>
#include "life.h"

int main()
{
    int width = 8;
    int height = 8;

    std::vector<std::vector<int>> space = {
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,1,0,0,0},
        {0,0,0,0,1,0,0,0},
        {0,0,0,1,1,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0}
    };

    Life life(width, height, space);
    life.setCell(2,3,1);
    life.print();
    life.step();
    life.print();
    std::cout << life.getCell(2,3);
    std::cout << life.getCell(3,2);

    return 0;
}