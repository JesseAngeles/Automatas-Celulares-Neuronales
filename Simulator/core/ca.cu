// ca.cu

#include "ca.h"

// Constant memory
__constant__ int c_neighbourhood[128]; // Hasta 64 vecinos
__constant__ int c_rule[4096];

__global__ void CA_GPU(
    const int *__restrict__ A,
    int *__restrict__ B,
    int width,
    int height,
    int states,
    int n_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int total = 0;
    int p = 1;

    for (int i = n_size - 1; i >= 0; i--)
    {
        int nx = (idx + c_neighbourhood[i * 2] + width) % width;
        int ny = (idy + c_neighbourhood[i * 2 + 1] + height) % height;

        total += A[ny * width + nx] * p;
        p *= states;
    }

    B[idy * width + idx] = c_rule[total];
}

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

    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, in_n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_rule, rule, in_r_size * sizeof(int));
}

CA::~CA()
{
    cudaFree(A_d);
    cudaFree(B_d);

    delete[] space;
    delete[] neighbourhood;
    delete[] rule;
}

void CA::step()
{
    CA_GPU<<<blocks, threads>>>(A_d, B_d, width, height, states, n_size);
    cudaMemcpy(space, B_d, size, cudaMemcpyDeviceToHost);
    int *tmp = A_d;
    A_d = B_d;
    B_d = tmp;
}

int *CA::run(int steps)
{
    int *history = new int[steps * height * width];
    size_t frame_size = height * width;

    for (int s = 0; s < steps; s++)
    {
        CA_GPU<<<blocks, threads>>>(A_d, B_d, width, height, states, n_size);

        cudaMemcpy(&history[s * frame_size], B_d, size, cudaMemcpyDeviceToHost);

        int *tmp = A_d;
        A_d = B_d;
        B_d = tmp;
    }

    std::memcpy(space, &history[(steps - 1) * frame_size], size);

    return history;
}

void CA::print()
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            std::cout << space[y * width + x] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Getters
int *CA::getSpace() const { return space; }
int CA::getWidth() const { return width; }
int CA::getHeight() const { return height; }
int CA::getStates() const { return states; }
int *CA::getNeighbourhood() const { return neighbourhood; }
int *CA::getRule() const { return rule; }
int CA::getNSize() const { return n_size; }
int CA::getRSize() const { return r_size; }
int CA::getCell(int x, int y) const
{
    return space[y * width + x];
}

// Setters
void CA::setStates(int new_states)
{
    states = new_states;
    fixRuleRize();
}
void CA::setNeighbourhood(int *new_neighbourhood, int new_n_size)
{
    if (new_n_size > 64)
        throw std::runtime_error("n_size exceed memory (64)");

    // El tamaño cambiar
    if (new_n_size != n_size)
    {
        n_size = new_n_size;
        delete[] neighbourhood;
        neighbourhood = new int[n_size * 2];
        fixRuleRize();
    }
    std::memcpy(neighbourhood, new_neighbourhood, n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, n_size * 2 * sizeof(int));
}

void CA::setRule(int *new_rule)
{
    std::memcpy(rule, new_rule, r_size * sizeof(int));
    for (int i = 0; i < r_size; i++)
        rule[i] %= states;
    cudaMemcpyToSymbol(c_rule, rule, r_size * sizeof(int));
}

void CA::setSpace(int *new_space)
{
    std::memcpy(space, new_space, size);
    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
}
void CA::setCell(int x, int y, int value)
{
    space[y * width + x] = value;
    cudaMemcpy(&A_d[y * width + x], &space[y * width + x], sizeof(int), cudaMemcpyHostToDevice);
}

void CA::fixRuleRize()
{
    int new_r_size = (int)pow(states, n_size);
    if (new_r_size > 4096)
        throw std::runtime_error("r_size exceed the memory (4096)");

    int *new_rule = new int[new_r_size];

    int elements = min(r_size, new_r_size);
    for (int i = 0; i < elements; i++)
        new_rule[i] = rule[i];

    if (new_r_size > r_size)
        for (int i = r_size; i < new_r_size; i++)
            new_rule[i] = 0;

    delete[] rule;

    r_size = new_r_size;
    rule = new_rule;
    for (int i = 0; i < r_size; i++)
        rule[i] %= states;
    cudaMemcpyToSymbol(c_rule, rule, r_size * sizeof(int));
}