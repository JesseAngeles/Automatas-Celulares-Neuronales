// core.grid.h

#pragma once
#include <vector>

class Grid
{
public:
    Grid(int width, int height);

    int getWidth() const;
    int getHeight() const;

    int get(int x, int y) const;
    void set(int x, int y, int value);

    const std::vector<int> &data() const;

private:
    int width;
    int height;

    std::vector<int> cells;
};