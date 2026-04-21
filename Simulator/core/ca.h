// ca.h

#pragma once

#include <cstdint>
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
        uint8_t *space,
        int states,
        int *neighbourhood, int n_size,
        int *group_sizes, int n_groups,
        int *rule, int r_size);
    ~CA();

    void step();
    uint8_t *run(int steps);
    void print();

    // Getters
    uint8_t *getSpace();          // no-const: sincroniza device→host si hace falta
    int getWidth() const;
    int getHeight() const;
    int getStates() const;
    int *getNeighbourhood() const;
    int getNSize() const;
    int *getGroupSizes() const;
    int getNGroups() const;
    int *getRule() const;
    int getRSize() const;
    uint8_t getCell(int x, int y);
    int getBoundary() const;

    // Setters
    void setSpace(const uint8_t *);
    void setStates(int);
    void setNeighbourhood(int *, int, int *, int);
    void setRule(int *);
    void setCell(int, int, int);
    void setBoundary(int mode);

private:
    uint8_t *space;
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

    // host_dirty = true cuando el buffer del host está desfasado frente a A_d
    bool host_dirty;

    size_t size;
    uint8_t *A_d, *B_d;
    cudaStream_t stream;
    dim3 threads;
    dim3 blocks;

    void fixRuleSize();
    void syncHost();   // device → host si host_dirty
    void launch();     // lanza el kernel con boundary especializado en compile-time
};
