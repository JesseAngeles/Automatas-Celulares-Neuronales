// ca.cu

#include "ca.h"

__global__ void CA_GPU(
    int *A,
    int *B,
    int width,
    int height,
    int states,
    int *neighbourhood, int n_size,
    int *rule, int r_size);

CA::CA(
    int in_width,
    int in_height,
    int *in_space,
    int in_states,
    int *in_neighbourhood, int in_n_size,
    int *in_rule, int in_r_size)
    : width(in_width),
      height(in_height),
      states(in_states),
      n_size(in_n_size),
      r_size(in_r_size)
{
    size = width * height * sizeof(int);
    threads = dim3(16, 16);
    blocks = dim3((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

    space = new int[in_width * in_height];
    std::memcpy(space, in_space, size);

    neighbourhood = new int[in_n_size * 2];
    std::memcpy(neighbourhood, in_neighbourhood, in_n_size * 2 * sizeof(int));

    rule = new int[in_r_size];
    std::memcpy(rule, in_rule, in_r_size * sizeof(int));

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&N_d, n_size * 2 * sizeof(int));
    cudaMalloc(&R_d, r_size * sizeof(int));
    cudaMemcpy(N_d, neighbourhood, n_size * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(R_d, rule, r_size * sizeof(int), cudaMemcpyHostToDevice);
}

CA::~CA()
{
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(N_d);
    cudaFree(R_d);

    delete[] space;
    delete[] neighbourhood;
    delete[] rule;
}

void CA::step()
{
    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);

    CA_GPU<<<blocks, threads>>>(A_d, B_d, width, height, states, N_d, n_size, R_d, r_size);
    cudaDeviceSynchronize();

    cudaMemcpy(space, B_d, size, cudaMemcpyDeviceToHost);
}

__global__ void CA_GPU(
    int *A,
    int *B,
    int width,
    int height,
    int states,
    int *neighbourhood, int n_size,
    int *rule, int r_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int total = 0;
    int p = 1;

    // Recorremos los vecinos (n_size es la cantidad de parejas, ej: 3)
    for (int i = n_size - 1; i >= 0; i--)
    {
        // Extraemos los offsets x e y del arreglo plano
        int offset_x = neighbourhood[i * 2];
        int offset_y = neighbourhood[i * 2 + 1];

        // Calculamos la posición del vecino con condiciones de borde
        int nx = (idx + offset_x + width) % width;
        int ny = (idy + offset_y + height) % height;

        // Accedemos a la memoria usando el índice lineal 2D
        total += A[ny * width + nx] * p;
        p *= states;
    }

    // Save result
    B[idy * width + idx] = rule[total];
}
void CA::print()
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            std::cout << space[y * width + x] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Getters
int *CA::getSpace() const { return space; }
int CA::getWidth() const { return width; }
int CA::getHeight() const { return height; }
int CA::getStates() const { return states; }
int *CA::getNeighbourhood() const { return neighbourhood; }
int *CA::getRule() const { return rule; }
int CA::getCell(int x, int y) const { 
    return space[y * width + x];
}

int CA::getNSize() const { return n_size; }
int CA::getRSize() const { return r_size; }

// Setters
void CA::setSpace(int *new_space) { std::memcpy(space, new_space, size); }
void CA::setStates(int new_states) { states = new_states; }
void CA::setNeighbourhood(int *new_neighbourhood) { neighbourhood = new_neighbourhood; }
void CA::setRule(int *new_rule) { rule = new_rule; }
void CA::setCell(int x, int y, int value) { 
    space[y * width + x] = value;
 }