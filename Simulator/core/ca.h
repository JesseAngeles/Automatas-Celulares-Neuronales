// ca.h

#pragma once

#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

enum BoundaryMode {
    BOUNDARY_OPEN       = 0,
    BOUNDARY_PERIODIC   = 1,
    BOUNDARY_REFLECTIVE = 2,
    BOUNDARY_REFLEXIVE  = 3
};

class CA
{

public:
    CA(
        int width,
        int height,
        int *space,
        int states,
        int *neighbourhood, int n_size,
        int *group_sizes, int n_groups,
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
    int getNSize() const;
    int *getGroupSizes() const;
    int getNGroups() const;
    int *getRule() const;
    int getRSize() const;
    int getCell(int x, int y) const;
    int getBoundary() const;

    // Setters
    void setSpace(int *);
    void setStates(int);
    void setNeighbourhood(int *, int, int *, int);
    void setRule(int *);
    void setCell(int, int, int);
    void setBoundary(int mode);

private:
    int *space;
    int width;
    int height;

    int states;

    int *neighbourhood;
    int n_size;

    int *group_sizes;
    int n_groups;

    int *rule;
    int r_size;

    int boundary_mode;

    size_t size;
    int *A_d, *B_d;
    dim3 threads;
    dim3 blocks;

    void fixRuleSize();
};
