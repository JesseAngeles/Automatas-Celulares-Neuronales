// nca.cu

#include "nca.h"
#include <stdexcept>
#include <string>

// Constant memory
constexpr int MAX_VECINOS = 128;                      // 9 x 9 = 81 cabe; 11 x 11 = 121 también
__constant__ int c_neighbourhood[MAX_VECINOS * 2];    // pares (dx, dy)

static void validarVecindad(int n)
{
    if (n < 1 || n > MAX_VECINOS)
        throw std::invalid_argument("La vecindad debe tener entre 1 y " + std::to_string(MAX_VECINOS) +
                                    " vecinos (tiene " + std::to_string(n) + ")");
}

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

template <int BM>
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
    int max_layer_size,
    int out_size,
    int one_hot)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int cell_id = idy * width + idx;
    float *cur = &temp[cell_id * 2 * max_layer_size];
    float *nxt = cur + max_layer_size;

    // Valores del vecindario, con la misma frontera que CA_GPU_T
    for (int i = 0; i < n_size; i++)
    {
        int raw_nx = idx + c_neighbourhood[i * 2];
        int raw_ny = idy + c_neighbourhood[i * 2 + 1];
        bool fuera = raw_nx < 0 || raw_nx >= width || raw_ny < 0 || raw_ny >= height;

        if constexpr (BM == 1) {
            // Periódica (toroidal)
            int nx = ((raw_nx % width)  + width)  % width;
            int ny = ((raw_ny % height) + height) % height;
            cur[i] = A[ny * width + nx];
        } else if constexpr (BM == 2) {
            // Reflectiva: fuera → celda origen
            cur[i] = fuera ? A[idy * width + idx] : A[raw_ny * width + raw_nx];
        } else if constexpr (BM == 3) {
            // Reflexiva: fuera → clamp
            int nx = max(0, min(raw_nx, width  - 1));
            int ny = max(0, min(raw_ny, height - 1));
            cur[i] = A[ny * width + nx];
        } else {
            // Abierta: fuera → 0
            cur[i] = fuera ? 0.0f : A[raw_ny * width + raw_nx];
        }
    }

    // Con one-hot la primera capa es una tabla: la entrada del vecino i con estado s es i * one_hot + s,
    // así que cada neurona suma la columna de su estado en vez de multiplicar las n_size * one_hot entradas
    int l0 = 0;
    if (one_hot > 0)
    {
        int in = layer_in[0];
        int out = layer_out[0];
        for (int j = 0; j < out; j++)
        {
            float sum = biases[j];
            for (int i = 0; i < n_size; i++)
            {
                int s = __float2int_rn(cur[i]);
                if (s >= 0 && s < one_hot)
                    sum += weights[j * in + i * one_hot + s];
            }
            nxt[j] = activate(sum, activations[0]);
        }
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
        l0 = 1;
    }

    // Forward pass
    for (int l = l0; l < n_layers; l++)
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

    if (out_size == 1) {
        B[cell_id] = cur[0];
    } else {
        int best = 0;
        float best_val = cur[0];
        for (int j = 1; j < out_size; j++) {
            if (cur[j] > best_val) {
                best_val = cur[j];
                best     = j;
            }
        }
        B[cell_id] = (float)best;
    }
}

NCA::NCA(
    int in_width,
    int in_height,
    float *in_space,
    int *in_neighbourhood, int in_n_size)
    : width(in_width),
      height(in_height),
      n_size(in_n_size),
      boundary_mode(1),
      one_hot_states(0),
      first_layer_in(0),
      n_layers(0),
      max_layer_size(0),
      out_size(1),
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
    validarVecindad(in_n_size);
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
    int esperado = one_hot_states > 0 ? n_size * one_hot_states : n_size;
    if (first_layer_in != esperado)
        throw std::runtime_error("La primera capa tiene " + std::to_string(first_layer_in) + " entradas y se esperaban " +
                                 std::to_string(esperado) + " (" + std::to_string(n_size) + " vecinos" +
                                 (one_hot_states > 0 ? " x " + std::to_string(one_hot_states) + " estados one-hot)" : ")"));

    launch();

    cudaMemcpy(space, B_d, size, cudaMemcpyDeviceToHost);
    float *tmp = A_d;
    A_d = B_d;
    B_d = tmp;
}

void NCA::launch()
{
#define NCA_LAUNCH(BM) NCA_GPU<BM><<<blocks, threads>>>(                    \
        A_d, B_d, width, height, n_size, weights_d, biases_d,               \
        w_offsets_d, b_offsets_d, layer_in_d, layer_out_d, activation_d,    \
        n_layers, temp_d, max_layer_size, out_size, one_hot_states)
    switch (boundary_mode)
    {
        case 0:  NCA_LAUNCH(0); break;
        case 2:  NCA_LAUNCH(2); break;
        case 3:  NCA_LAUNCH(3); break;
        default: NCA_LAUNCH(1); break;
    }
#undef NCA_LAUNCH
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
    this->out_size = layer_out[n_layers - 1];
    first_layer_in = layer_in[0];
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
int NCA::getBoundary() const { return boundary_mode; }
int NCA::getOneHot() const { return one_hot_states; }
float NCA::getCell(int x, int y) const { return space[y * width + x]; }

// Setters
void NCA::setSpace(float *new_space)
{
    std::memcpy(space, new_space, size);
    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
}

void NCA::setBoundary(int mode) { boundary_mode = mode; }
void NCA::setOneHot(int states) { one_hot_states = states; }

void NCA::setCell(int x, int y, float value)
{
    space[y * width + x] = value;
    cudaMemcpy(&A_d[y * width + x], &space[y * width + x], sizeof(float), cudaMemcpyHostToDevice);
}

void NCA::setNeighbourhood(int *new_neighbourhood, int new_n_size)
{
    validarVecindad(new_n_size);
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