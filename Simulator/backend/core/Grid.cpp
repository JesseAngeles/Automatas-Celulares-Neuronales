// core.Grid

#include "Grid.h"

Grid::Grid(int w, int h): width(w), height(h), cells(w * h, 0) {}

int Grid::getWidth() const {
    return width;
}

int Grid::getHeight() const {
    return height;
}

int Grid::get(int x, int y) const
{
    if(x < 0 || x >= width || y < 0 || y >= height)
        return 0;

    return cells[y * width + x];
}

void Grid::set(int x, int y, int value) {
    cells[y * width + x] = value;
}