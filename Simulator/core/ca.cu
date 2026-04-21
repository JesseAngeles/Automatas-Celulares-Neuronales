// ca.cu

#include "ca.h"
#include <algorithm>

// Constant memory
__constant__ int c_neighbourhood[128]; // Hasta 64 vecinos (dx,dy pares)
__constant__ int c_group_sizes[64];    // Tamaño de cada grupo
__constant__ int c_rule[4096];

template <int BM>
__global__ void CA_GPU_T(
    const uint8_t *__restrict__ A,
    uint8_t *__restrict__ B,
    int width,
    int height,
    int states,
    int n_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int sums[16];   // hasta 16 grupos — suficiente en la práctica
    int offset = 0;

    for (int g = 0; g < n_groups; g++)
    {
        int sum = 0;
        int gs = c_group_sizes[g];
        for (int k = 0; k < gs; k++)
        {
            int raw_nx = idx + c_neighbourhood[(offset + k) * 2];
            int raw_ny = idy + c_neighbourhood[(offset + k) * 2 + 1];
            int cell_val;

            if constexpr (BM == 1) {
                // Periódica (toroidal)
                int nx = ((raw_nx % width)  + width)  % width;
                int ny = ((raw_ny % height) + height) % height;
                cell_val = A[ny * width + nx];
            } else if constexpr (BM == 2) {
                // Reflectiva: fuera → celda origen
                if (raw_nx < 0 || raw_nx >= width || raw_ny < 0 || raw_ny >= height)
                    cell_val = A[idy * width + idx];
                else
                    cell_val = A[raw_ny * width + raw_nx];
            } else if constexpr (BM == 3) {
                // Reflexiva: fuera → clamp
                int nx = max(0, min(raw_nx, width  - 1));
                int ny = max(0, min(raw_ny, height - 1));
                cell_val = A[ny * width + nx];
            } else {
                // Abierta: fuera → 0
                if (raw_nx < 0 || raw_nx >= width || raw_ny < 0 || raw_ny >= height)
                    cell_val = 0;
                else
                    cell_val = A[raw_ny * width + raw_nx];
            }
            sum += cell_val;
        }
        sums[g] = sum;
        offset += gs;
    }

    int rule_index = 0;
    int multiplier = 1;
    for (int g = n_groups - 1; g >= 0; g--)
    {
        rule_index += sums[g] * multiplier;
        multiplier *= (states - 1) * c_group_sizes[g] + 1;
    }

    B[idy * width + idx] = (uint8_t)c_rule[rule_index];
}

void CA::launch()
{
    switch (boundary_mode)
    {
        case 0: CA_GPU_T<0><<<blocks, threads, 0, stream>>>(A_d, B_d, width, height, states, n_groups); break;
        case 1: CA_GPU_T<1><<<blocks, threads, 0, stream>>>(A_d, B_d, width, height, states, n_groups); break;
        case 2: CA_GPU_T<2><<<blocks, threads, 0, stream>>>(A_d, B_d, width, height, states, n_groups); break;
        case 3: CA_GPU_T<3><<<blocks, threads, 0, stream>>>(A_d, B_d, width, height, states, n_groups); break;
        default: CA_GPU_T<1><<<blocks, threads, 0, stream>>>(A_d, B_d, width, height, states, n_groups); break;
    }
}

CA::CA(
    int in_width,
    int in_height,
    uint8_t *in_space,
    int in_states,
    int *in_neighbourhood, int in_n_size,
    int *in_group_sizes, int in_n_groups,
    int *in_rule, int in_r_size)
    : width(in_width),
      height(in_height),
      states(in_states),
      n_size(in_n_size),
      n_groups(in_n_groups),
      r_size(in_r_size),
      boundary_mode(BOUNDARY_PERIODIC),
      host_dirty(false)
{
    if (in_n_size > 64)
        throw std::runtime_error("n_size exceeds memory limit (64)");
    if (in_n_groups > 16)
        throw std::runtime_error("n_groups exceeds memory limit (16)");
    if (in_r_size > 4096)
        throw std::runtime_error("r_size exceeds memory limit (4096)");

    size = (size_t)width * height * sizeof(uint8_t);

    threads = dim3(16, 16);
    blocks = dim3((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

    space = new uint8_t[(size_t)width * height];
    std::memcpy(space, in_space, size);

    neighbourhood = new int[in_n_size * 2];
    std::memcpy(neighbourhood, in_neighbourhood, in_n_size * 2 * sizeof(int));

    group_sizes = new int[in_n_groups];
    std::memcpy(group_sizes, in_group_sizes, in_n_groups * sizeof(int));

    rule = new int[in_r_size];
    std::memcpy(rule, in_rule, in_r_size * sizeof(int));

    cudaStreamCreate(&stream);
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);

    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, in_n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_group_sizes, group_sizes, in_n_groups * sizeof(int));
    cudaMemcpyToSymbol(c_rule, rule, in_r_size * sizeof(int));
}

CA::~CA()
{
    cudaStreamSynchronize(stream);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaStreamDestroy(stream);

    delete[] space;
    delete[] neighbourhood;
    delete[] group_sizes;
    delete[] rule;
}

void CA::step()
{
    launch();
    std::swap(A_d, B_d);
    host_dirty = true;
}

uint8_t *CA::run(int steps)
{
    uint8_t *history = nullptr;
    // Memoria pinned para habilitar cudaMemcpyAsync real
    cudaHostAlloc((void **)&history, (size_t)steps * height * width, cudaHostAllocDefault);

    size_t frame_size = (size_t)height * width;

    for (int s = 0; s < steps; s++)
    {
        launch();
        cudaMemcpyAsync(&history[s * frame_size], B_d, size, cudaMemcpyDeviceToHost, stream);
        std::swap(A_d, B_d);
    }
    cudaStreamSynchronize(stream);

    // Espacio en host = último frame del historial
    std::memcpy(space, &history[(steps - 1) * frame_size], size);
    host_dirty = false;

    return history;
}

void CA::syncHost()
{
    if (!host_dirty) return;
    cudaMemcpyAsync(space, A_d, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    host_dirty = false;
}

void CA::print()
{
    syncHost();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            std::cout << (int)space[y * width + x] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Getters
uint8_t *CA::getSpace()             { syncHost(); return space; }
int CA::getWidth() const            { return width; }
int CA::getHeight() const           { return height; }
int CA::getStates() const           { return states; }
int *CA::getNeighbourhood() const   { return neighbourhood; }
int CA::getNSize() const            { return n_size; }
int *CA::getGroupSizes() const      { return group_sizes; }
int CA::getNGroups() const          { return n_groups; }
int *CA::getRule() const            { return rule; }
int CA::getRSize() const            { return r_size; }
uint8_t CA::getCell(int x, int y)   { syncHost(); return space[y * width + x]; }

// Setters
void CA::setStates(int new_states)
{
    states = new_states;
    fixRuleSize();
}

void CA::setNeighbourhood(int *new_neighbourhood, int new_n_size, int *new_group_sizes, int new_n_groups)
{
    if (new_n_size > 64)
        throw std::runtime_error("n_size exceeds memory limit (64)");
    if (new_n_groups > 16)
        throw std::runtime_error("n_groups exceeds memory limit (16)");

    if (new_n_size != n_size)
    {
        delete[] neighbourhood;
        neighbourhood = new int[new_n_size * 2];
        n_size = new_n_size;
    }
    std::memcpy(neighbourhood, new_neighbourhood, n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, n_size * 2 * sizeof(int));

    if (new_n_groups != n_groups)
    {
        delete[] group_sizes;
        group_sizes = new int[new_n_groups];
        n_groups = new_n_groups;
    }
    std::memcpy(group_sizes, new_group_sizes, n_groups * sizeof(int));
    cudaMemcpyToSymbol(c_group_sizes, group_sizes, n_groups * sizeof(int));

    fixRuleSize();
}

void CA::setRule(int *new_rule)
{
    std::memcpy(rule, new_rule, r_size * sizeof(int));
    for (int i = 0; i < r_size; i++)
        rule[i] %= states;
    cudaMemcpyToSymbol(c_rule, rule, r_size * sizeof(int));
}

void CA::setSpace(const uint8_t *new_space)
{
    syncHost();  // descarta datos dirty del device; vamos a sobrescribir A_d
    std::memcpy(space, new_space, size);
    cudaMemcpyAsync(A_d, space, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    host_dirty = false;
}

void CA::setCell(int x, int y, int value)
{
    syncHost();
    space[y * width + x] = (uint8_t)value;
    cudaMemcpyAsync(&A_d[y * width + x], &space[y * width + x],
                    sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

void CA::setBoundary(int mode) { boundary_mode = mode; }
int  CA::getBoundary() const   { return boundary_mode; }

void CA::fixRuleSize()
{
    int new_r_size = 1;
    for (int g = 0; g < n_groups; g++)
        new_r_size *= (states - 1) * group_sizes[g] + 1;

    if (new_r_size > 4096)
        throw std::runtime_error("r_size exceeds memory limit (4096)");

    int *new_rule = new int[new_r_size];

    int elements = (r_size < new_r_size) ? r_size : new_r_size;
    for (int i = 0; i < elements; i++)
        new_rule[i] = rule[i] % states;
    for (int i = r_size; i < new_r_size; i++)
        new_rule[i] = 0;

    delete[] rule;
    rule = new_rule;
    r_size = new_r_size;
    cudaMemcpyToSymbol(c_rule, rule, r_size * sizeof(int));
}
