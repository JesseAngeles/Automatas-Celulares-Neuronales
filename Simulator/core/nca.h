// nca.h

#pragma once

#include <iostream>
#include <cstring>
#include <mutex>
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
    void run(int steps, float *out);  // escribe (steps, H, W) en out; evolución continua
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

    // Modo estocástico: cada celda se actualiza con probabilidad p por paso.
    // p = 1.0 reproduce el comportamiento determinista.
    void setUpdateProb(float p);
    float getUpdateProb() const;
    void setSeed(unsigned long long seed);  // reinicia el contador de pasos del RNG

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
    int total_weights;
    int total_biases;
    bool use_smem;      // pesos+biases caben en shared memory del bloque
    bool nn_ready;

    float *temp_d;
    int   out_size;

    // Modo estocástico: probabilidad de actualización por celda; semilla y
    // contador de pasos (offset del RNG counter-based, un valor por frame)
    float update_prob;
    unsigned long long rng_seed;
    unsigned long long step_count;

    size_t size;
    float *A_d, *B_d;
    cudaStream_t stream;
    dim3 threads;
    dim3 blocks;

    // run(): mismo pipeline a profundidad 2 que CA::run (staging pinned
    // reutilizable de 2 chunks + stream de copia con eventos)
    static constexpr int RUN_CHUNK = 8;
    float *stage_h;
    size_t stage_cap;
    cudaStream_t copy_stream;
    cudaEvent_t k_done[2 * RUN_CHUNK];
    cudaEvent_t c_done[2 * RUN_CHUNK];
    cudaEvent_t chunk_done[2];

    // step()/run() corren sin GIL: serializar el acceso a GPU
    std::mutex gpu_mutex;

    void freeNN();
    void launchNN();  // lanza NCA_GPU sobre `stream` con la config vigente
};