#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#pragma once

class Life
{
public:
    Life(int, int, std::vector<std::vector<int>>);
    ~Life();
    void step();
    void print();

    // Getters
    std::vector<std::vector<int>> getSpace() const;
    int getWidth() const;
    int getHeight() const;
    int getCell(int x, int y) const;

    // Setters
    void setSpace(std::vector<std::vector<int>>);
    void setCell(int x, int y, int);

private:
    int *space;

    size_t size;
    int *A_d, *B_d;
    int width;
    int height;

    dim3 threads;
    dim3 blocks;
};