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
    int getBoundary() const;
    int getOneHot() const;

    // Setters
    void setSpace(float *);
    void setCell(int, int, float);
    void setNeighbourhood(int *, int);
    void setBoundary(int mode);   // 0=abierta, 1=periódica, 2=reflectiva, 3=reflexiva (igual que CA)
    void setOneHot(int states);   // 0 = entrada cruda; > 0 = cada vecino en one-hot con ese número de estados

private:
    int width;
    int height;
    float *space;

    int *neighbourhood;
    int n_size;
    int boundary_mode;
    int one_hot_states;
    int first_layer_in;

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
    void launch();   // lanza el kernel con la frontera especializada en compile-time
};