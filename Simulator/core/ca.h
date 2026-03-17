// ca.h

#pragma once

#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

class CA
{

public:
    CA(
        int width,
        int height,
        int *space,
        int states,
        int *neighbourhood, int n_size,
        int *rule, int r_size);
    ~CA();

    void step();
    int *run(int steps);
    void print();

    // Getters
    int *getSpace() const;
    int getWidth() const;
    int getHeight() const;
    int getStates() const;
    int *getNeighbourhood() const;
    int *getRule() const;
    int getCell(int x, int y) const;

    int getNSize() const;
    int getRSize() const;

    // Setters
    void setSpace(int *);
    void setStates(int);
    void setNeighbourhood(int *);
    void setRule(int *);
    void setCell(int, int, int);

private:
    int *space;
    int width;
    int height;

    int states;

    int *neighbourhood;
    int n_size;

    int *rule;
    int r_size;

    size_t size;
    int *A_d, *B_d;
    dim3 threads;
    dim3 blocks;
};