// nca.h

#pragma once

#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

struct MLPlayer
{
    int in_size;
    int out_size;
    int activation; 
};

class NCA
{
public:
    NCA(
        int width,
        int height,
        float *space,
        int *neighbourhood, int n_size);

    ~NCA();

    void step();
    void print();

    void setWeights(
        float **weights,
        float **biases,
        int *layer_in,
        int *layer_out,
        int *activations,
        int n_layers
    );

    // Getters
    int getWidth() const;
    int getHeight() const;
    
    float *getSpace() const;
    float getCell(int x, int y) const;
    int *getNeighbourhood() const;
    int getNSize() const;

    // Setters
    void setSpace(float *);
    void setCell(int, int, float);
    void setNeighbourhood(int *, int);

private:
    int width;
    int height;
    float *space;

    int *neighbourhood;
    int n_size;

    // CUDA
    float *weights_d;
    float *biases_d;
    int   *w_offsets_d;
    int   *b_offsets_d;
    int *layer_in_d;
    int *layer_out_d;
    int *activation_d;
    int n_layers;
    int max_layer_size;
    bool nn_ready;

    float *temp_d;
    int   out_size;

    size_t size;
    float *A_d, *B_d;
    dim3 threads;
    dim3 blocks;

    void freeNN();
};