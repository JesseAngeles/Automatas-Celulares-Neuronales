// nca.cu

#include "nca.h"

// Constant memory
__constant__ int c_neighbourhood[128]; // Hasta 64 vecinos

__device__ float activate(float x, int type)
{
    if (type == 0)
        return fmaxf(0.0f, x); // relu
    if (type == 1)
        return 1.0f / (1.0f + expf(-x)); // sigmoids
    if (type == 2)
        return tanh(x); // tanh
    return x;           // linear
}

__global__ void NCA_GPU(
    const float *__restrict__ A,
    float *__restrict__ B,
    int width,
    int height,
    int n_size,
    const float *weights,
    const float *biases,
    const int *w_offsets,
    const int *b_offsets,
    const int *layer_in,
    const int *layer_out,
    const int *activations,
    int n_layers,
    float *temp,
    int max_layer_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int cell_id = idy * width + idx;
    float *cur = &temp[cell_id * 2 * max_layer_size];
    float *nxt = cur + max_layer_size;

    // Valores del vecindario
    for (int i = 0; i < n_size; i++)
    {
        int nx = (idx + c_neighbourhood[i * 2] + width) % width;
        int ny = (idy + c_neighbourhood[i * 2 + 1] + height) % height;
        cur[i] = A[ny * width + nx];
    }

    // Forward pass
    for (int l = 0; l < n_layers; l++)
    {
        const float *W = &weights[w_offsets[l]];
        const float *b = &biases[b_offsets[l]];

        int in = layer_in[l];
        int out = layer_out[l];

        for (int j = 0; j < out; j++)
        {
            float sum = b[j];
            for (int i = 0; i < in; i++)
                sum += W[j * in + i] * cur[i];
            nxt[j] = activate(sum, activations[l]);
        }

        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    B[cell_id] = cur[0];
}

NCA::NCA(
    int in_width,
    int in_height,
    float *in_space,
    int *in_neighbourhood, int in_n_size)
    : width(in_width),
      height(in_height),
      n_size(in_n_size),
      n_layers(0),
      max_layer_size(0),
      nn_ready(false),
      weights_d(nullptr),
      biases_d(nullptr),
      w_offsets_d(nullptr),
      b_offsets_d(nullptr),
      layer_in_d(nullptr),
      layer_out_d(nullptr),
      activation_d(nullptr),
      temp_d(nullptr)
{
    size = width * height * sizeof(float);

    threads = dim3(16, 16);
    blocks = dim3((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

    space = new float[in_width * in_height];
    std::memcpy(space, in_space, size);

    neighbourhood = new int[in_n_size * 2];
    std::memcpy(neighbourhood, in_neighbourhood, in_n_size * 2 * sizeof(int));

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);

    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, in_n_size * 2 * sizeof(int));
}

NCA::~NCA()
{
    cudaFree(A_d);
    cudaFree(B_d);

    delete[] space;
    delete[] neighbourhood;

    freeNN();
}

void NCA::step()
{
    if (!nn_ready)
        throw std::runtime_error("Weights not loaded. Call setWeights() first.");

    NCA_GPU<<<blocks, threads>>>(
        A_d, B_d, width, height, n_size,
        weights_d, biases_d,
        w_offsets_d, b_offsets_d,
        layer_in_d, layer_out_d,
        activation_d, n_layers,
        temp_d, max_layer_size);

    cudaMemcpy(space, B_d, size, cudaMemcpyDeviceToHost);
    float *tmp = A_d;
    A_d = B_d;
    B_d = tmp;
}

void NCA::print()
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            std::cout << space[y * width + x] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

void NCA::setWeights(
    float **weights,
    float **biases,
    int *layer_in,
    int *layer_out,
    int *activations,
    int n_layers)
{
    freeNN();
    this->n_layers = n_layers;
    max_layer_size = 0;

    // Calcular offsets y tamaños totales
    int *w_offsets = new int[n_layers];
    int *b_offsets = new int[n_layers];
    int total_weights = 0;
    int total_biases = 0;

    for (int i = 0; i < n_layers; i++)
    {
        w_offsets[i] = total_weights;
        b_offsets[i] = total_biases;
        total_weights += layer_in[i] * layer_out[i];
        total_biases += layer_out[i];
        max_layer_size = max(max_layer_size, max(layer_in[i], layer_out[i]));
    }

    // Aplanar biases y pesos en arreglos del host
    float *flat_weights = new float[total_weights];
    float *flat_biases = new float[total_biases];

    for (int i = 0; i < n_layers; i++)
    {
        memcpy(flat_weights + w_offsets[i], weights[i], layer_in[i] * layer_out[i] * sizeof(float));
        memcpy(flat_biases + b_offsets[i], biases[i], layer_out[i] * sizeof(float));
    }

    // Subir a la GPU
    cudaMalloc(&weights_d,    total_weights * sizeof(float));
    cudaMalloc(&biases_d,     total_biases  * sizeof(float));
    cudaMalloc(&w_offsets_d,  n_layers * sizeof(int));
    cudaMalloc(&b_offsets_d,  n_layers * sizeof(int));
    cudaMalloc(&layer_in_d,   n_layers * sizeof(int));
    cudaMalloc(&layer_out_d,  n_layers * sizeof(int));
    cudaMalloc(&activation_d, n_layers * sizeof(int));

    cudaMemcpy(weights_d,    flat_weights, total_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases_d,     flat_biases,  total_biases  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_offsets_d,  w_offsets,    n_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_offsets_d,  b_offsets,    n_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(layer_in_d,   layer_in,     n_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(layer_out_d,  layer_out,    n_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(activation_d, activations,  n_layers * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&temp_d, width * height * 2 * max_layer_size * sizeof(float));

    delete[] flat_weights;
    delete[] flat_biases;
    delete[] w_offsets;
    delete[] b_offsets;

    nn_ready = true;
}

// Getters
int NCA::getWidth() const { return width; }
int NCA::getHeight() const { return height; }
float *NCA::getSpace() const { return space; }
int *NCA::getNeighbourhood() const { return neighbourhood; }
int NCA::getNSize() const { return n_size; }
float NCA::getCell(int x, int y) const { return space[y * width + x]; }

// Setters
void NCA::setSpace(float *new_space)
{
    std::memcpy(space, new_space, size);
    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
}

void NCA::setCell(int x, int y, float value)
{
    space[y * width + x] = value;
    cudaMemcpy(&A_d[y * width + x], &space[y * width + x], sizeof(float), cudaMemcpyHostToDevice);
}

void NCA::setNeighbourhood(int *new_neighbourhood, int new_n_size)
{
    if (new_n_size != n_size)
    {
        delete[] neighbourhood;
        neighbourhood = new int[new_n_size * 2];
        n_size = new_n_size;
    }
    std::memcpy(neighbourhood, new_neighbourhood, n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, n_size * 2 * sizeof(int));
}

void NCA::freeNN()
{
    if (!nn_ready)
        return;

    cudaFree(weights_d);
    cudaFree(biases_d);
    cudaFree(w_offsets_d);
    cudaFree(b_offsets_d);
    cudaFree(layer_in_d);
    cudaFree(layer_out_d);
    cudaFree(activation_d);
    cudaFree(temp_d);

    nn_ready = false;
}