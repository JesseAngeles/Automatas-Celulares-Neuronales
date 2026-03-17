#include "life.h"

__global__ void lifeGPU(int *A, int *B, int widtth, int height);

Life::Life(int width, int height, std::vector<std::vector<int>> in_space)
    : width(width), height(height)
{
    space = new int[width * height];
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            space[i * width + j] = in_space[i][j];

    size = width * height * sizeof(int);
    threads = dim3(16, 16);
    blocks = dim3(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
}

Life::~Life()
{
    cudaFree(A_d);
    cudaFree(B_d);
    delete[] space;
}

void Life::step()
{
    cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice);

    lifeGPU<<<blocks, threads>>>(A_d, B_d, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(space, B_d, size, cudaMemcpyDeviceToHost);
}

void Life::print()
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            std::cout << space[i * width + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__global__ void lifeGPU(int *A, int *B, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    int sum = 0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
        {
            if (x == 0 && y == 0)
                continue;

            int r = row + x;
            int c = col + y;

            if (r >= 0 && r < height && c >= 0 && c < width)
                sum += A[r * width + c];
        }

    if (
        (A[row * width + col] == 0 && sum == 3) ||
        (A[row * width + col] == 1 && sum <= 3 && sum >= 2))
        B[row * width + col] = 1;
    else
        B[row * width + col] = 0;
}

// Getters
std::vector<std::vector<int>> Life::getSpace() const
{
    std::vector<std::vector<int>> return_space(height, std::vector<int>(width));
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            return_space[i][j] = space[i * width + j];

    return return_space;
}

int Life::getWidth() const { return width; }

int Life::getHeight() const { return height; }

int Life::getCell(int x, int y) const { 
    return space[y * width + x]; 
}

// Setters
void Life::setSpace(std::vector<std::vector<int>> new_space) { 
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            space[i * width + j] = new_space[i][j];
}

void Life::setCell(int x, int y, int val) {
    space[y * width + x] = val;
}