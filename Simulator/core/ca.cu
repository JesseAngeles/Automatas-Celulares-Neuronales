// ca.cu

#include "ca.h"

// Constant memory
__constant__ int c_neighbourhood[128]; // Hasta 64 vecinos (dx,dy pares)
__constant__ int c_group_sizes[64];    // Tamaño de cada grupo
__constant__ int c_rule[4096];

__global__ void CA_GPU(
    const int *__restrict__ A,
    int *__restrict__ B,
    int width,
    int height,
    int states,
    int n_groups,
    int boundary_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    // Calcular suma de cada grupo
    int sums[64];
    int offset = 0;

    for (int g = 0; g < n_groups; g++)
    {
        int sum = 0;
        for (int k = 0; k < c_group_sizes[g]; k++)
        {
            int raw_nx = idx + c_neighbourhood[(offset + k) * 2];
            int raw_ny = idy + c_neighbourhood[(offset + k) * 2 + 1];
            int cell_val;

            if (boundary_mode == 1) {
                // Periódica: toroidal
                int nx = ((raw_nx % width)  + width)  % width;
                int ny = ((raw_ny % height) + height) % height;
                cell_val = A[ny * width + nx];
            } else if (boundary_mode == 2) {
                // Reflectiva: fuera de rango → valor de la celda origen
                if (raw_nx < 0 || raw_nx >= width || raw_ny < 0 || raw_ny >= height)
                    cell_val = A[idy * width + idx];
                else
                    cell_val = A[raw_ny * width + raw_nx];
            } else if (boundary_mode == 3) {
                // Reflexiva: fuera de rango → vecino más cercano (clamp)
                int nx = max(0, min(raw_nx, width  - 1));
                int ny = max(0, min(raw_ny, height - 1));
                cell_val = A[ny * width + nx];
            } else {
                // Abierta (0 y cualquier otro): fuera de rango → 0
                if (raw_nx < 0 || raw_nx >= width || raw_ny < 0 || raw_ny >= height)
                    cell_val = 0;
                else
                    cell_val = A[raw_ny * width + raw_nx];
            }
            sum += cell_val;
        }
        sums[g] = sum;
        offset += c_group_sizes[g];
    }

    // Índice lineal: primer grupo es el más significativo
    int rule_index = 0;
    int multiplier = 1;
    for (int g = n_groups - 1; g >= 0; g--)
    {
        rule_index += sums[g] * multiplier;
        multiplier *= (states - 1) * c_group_sizes[g] + 1;
    }

    B[idy * width + idx] = c_rule[rule_index];
}

CA::CA(
    int in_width,
    int in_height,
    int *in_space,
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
      boundary_mode(BOUNDARY_PERIODIC)
{
    size = width * height * sizeof(int);

    threads = dim3(16, 16);
    blocks = dim3((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

    space = new int[in_width * in_height];
    std::memcpy(space, in_space, size);

    neighbourhood = new int[in_n_size * 2];
    std::memcpy(neighbourhood, in_neighbourhood, in_n_size * 2 * sizeof(int));

    group_sizes = new int[in_n_groups];
    std::memcpy(group_sizes, in_group_sizes, in_n_groups * sizeof(int));

    rule = new int[in_r_size];
    std::memcpy(rule, in_rule, in_r_size * sizeof(int));

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);

    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, in_n_size * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_group_sizes, group_sizes, in_n_groups * sizeof(int));
    cudaMemcpyToSymbol(c_rule, rule, in_r_size * sizeof(int));
}

CA::~CA()
{
    cudaFree(A_d);
    cudaFree(B_d);

    delete[] space;
    delete[] neighbourhood;
    delete[] group_sizes;
    delete[] rule;
}

void CA::step()
{
    CA_GPU<<<blocks, threads>>>(A_d, B_d, width, height, states, n_groups, boundary_mode);
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
        CA_GPU<<<blocks, threads>>>(A_d, B_d, width, height, states, n_groups, boundary_mode);

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
int CA::getNSize() const { return n_size; }
int *CA::getGroupSizes() const { return group_sizes; }
int CA::getNGroups() const { return n_groups; }
int *CA::getRule() const { return rule; }
int CA::getRSize() const { return r_size; }
int CA::getCell(int x, int y) const { return space[y * width + x]; }

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
    if (new_n_groups > 64)
        throw std::runtime_error("n_groups exceeds memory limit (64)");

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
