// ca.cu

#include "ca.h"
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error in " #call ": ") + cudaGetErrorString(_err)); \
} while(0)

// Constant memory
__constant__ int c_neighbourhood[128]; // Hasta 64 vecinos (dx,dy pares)
__constant__ int c_group_sizes[64];    // Tamaño de cada grupo
__constant__ int c_multipliers[64];    // Multiplicador posicional de cada grupo en el índice de regla

// TILED=true: el bloque carga su tile (blockDim + halo 2*rx × 2*ry) de A en
// shared memory y todas las lecturas de vecinos salen de ahí. La frontera se
// resuelve al llenar el halo, salvo REFLECTIVE (sustituye por el valor de la
// celda lectora, que depende del lector y no de la posición del vecino).
template <int BM, bool TILED>
__global__ void CA_GPU_T(
    const uint8_t *__restrict__ A,
    uint8_t *__restrict__ B,
    int width,
    int height,
    int n_groups,
    const int *__restrict__ rule,
    int rx,
    int ry)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ uint8_t tile[];
    int tw = 0, base_t = 0;

    if constexpr (TILED)
    {
        tw = blockDim.x + 2 * rx;
        const int th = blockDim.y + 2 * ry;
        const int x0 = blockIdx.x * blockDim.x - rx;
        const int y0 = blockIdx.y * blockDim.y - ry;
        const int nthreads = blockDim.x * blockDim.y;

        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tw * th; i += nthreads)
        {
            int gx = x0 + i % tw;
            int gy = y0 + i / tw;
            uint8_t v;
            if constexpr (BM == 1) {
                gx = ((gx % width)  + width)  % width;
                gy = ((gy % height) + height) % height;
                v = A[gy * width + gx];
            } else if constexpr (BM == 3) {
                gx = max(0, min(gx, width  - 1));
                gy = max(0, min(gy, height - 1));
                v = A[gy * width + gx];
            } else {
                // OPEN: fuera → 0. Para REFLECTIVE el lector nunca usa estos slots.
                v = (gx < 0 || gx >= width || gy < 0 || gy >= height)
                        ? 0 : A[gy * width + gx];
            }
            tile[i] = v;
        }
        __syncthreads();  // todos los hilos participan de la carga antes de salir

        base_t = (threadIdx.y + ry) * tw + (threadIdx.x + rx);
    }

    if (idx >= width || idy >= height)
        return;

    int rule_index = 0;
    int offset = 0;

    for (int g = 0; g < n_groups; g++)
    {
        int sum = 0;
        int gs = c_group_sizes[g];
        for (int k = 0; k < gs; k++)
        {
            int dx = c_neighbourhood[(offset + k) * 2];
            int dy = c_neighbourhood[(offset + k) * 2 + 1];
            int cell_val;

            if constexpr (TILED) {
                if constexpr (BM == 2) {
                    int nx = idx + dx;
                    int ny = idy + dy;
                    cell_val = (nx < 0 || nx >= width || ny < 0 || ny >= height)
                                   ? tile[base_t]
                                   : tile[base_t + dy * tw + dx];
                } else {
                    cell_val = tile[base_t + dy * tw + dx];
                }
            } else {
                int raw_nx = idx + dx;
                int raw_ny = idy + dy;

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
            }
            sum += cell_val;
        }
        rule_index += sum * c_multipliers[g];
        offset += gs;
    }

    B[idy * width + idx] = (uint8_t)rule[rule_index];
}

void CA::launch()
{
#define CA_LAUNCH(BM) do { \
    if (tiled) CA_GPU_T<BM, true ><<<blocks, threads, tile_bytes, stream>>>(A_d, B_d, width, height, n_groups, rule_d, rx, ry); \
    else       CA_GPU_T<BM, false><<<blocks, threads, 0,          stream>>>(A_d, B_d, width, height, n_groups, rule_d, rx, ry); \
} while (0)

    switch (boundary_mode)
    {
        case 0:  CA_LAUNCH(0); break;
        case 1:  CA_LAUNCH(1); break;
        case 2:  CA_LAUNCH(2); break;
        case 3:  CA_LAUNCH(3); break;
        default: CA_LAUNCH(1); break;
    }
#undef CA_LAUNCH
}

void CA::updateTileMeta()
{
    rx = 0;
    ry = 0;
    for (int i = 0; i < n_size; i++)
    {
        rx = std::max(rx, std::abs(neighbourhood[i * 2]));
        ry = std::max(ry, std::abs(neighbourhood[i * 2 + 1]));
    }
    tile_bytes = (size_t)(threads.x + 2 * rx) * (threads.y + 2 * ry);
    // Vecindades muy anchas: el tile deja de caber/rendir; usar el kernel global
    tiled = (tile_bytes <= 16384);
}

void CA::uploadMultipliers()
{
    int mult[64];
    mult[n_groups - 1] = 1;
    for (int g = n_groups - 2; g >= 0; g--)
        mult[g] = mult[g + 1] * ((states - 1) * group_sizes[g + 1] + 1);
    CUDA_CHECK(cudaMemcpyToSymbol(c_multipliers, mult, n_groups * sizeof(int)));
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
      host_dirty(false),
      stage_h(nullptr),
      stage_cap(0)
{
    if (in_n_size > 64)
        throw std::runtime_error("n_size exceeds memory limit (64)");
    if (in_n_groups > 64)
        throw std::runtime_error("n_groups exceeds memory limit (64)");
    if (in_r_size > (1 << 24))
        throw std::runtime_error("r_size exceeds memory limit (16777216)");

    size = (size_t)width * height * sizeof(uint8_t);

    // 32×8: cada warp cubre una fila completa de 32 celdas contiguas
    threads = dim3(32, 8);
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

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));
    for (int i = 0; i < 2 * RUN_CHUNK; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&k_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&c_done[i], cudaEventDisableTiming));
    }
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_done[i], cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc(&A_d, size));
    CUDA_CHECK(cudaMalloc(&B_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, space, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, in_n_size * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_group_sizes, group_sizes, in_n_groups * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&rule_d, (size_t)r_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rule_d, rule, (size_t)r_size * sizeof(int), cudaMemcpyHostToDevice));

    updateTileMeta();
    uploadMultipliers();
}

CA::~CA()
{
    cudaStreamSynchronize(stream);  // best-effort in destructor, don't throw
    cudaStreamSynchronize(copy_stream);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(rule_d);
    if (stage_h)
        cudaFreeHost(stage_h);
    for (int i = 0; i < 2 * RUN_CHUNK; i++)
    {
        cudaEventDestroy(k_done[i]);
        cudaEventDestroy(c_done[i]);
    }
    for (int i = 0; i < 2; i++)
        cudaEventDestroy(chunk_done[i]);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(copy_stream);

    delete[] space;
    delete[] neighbourhood;
    delete[] group_sizes;
    delete[] rule;
}

void CA::step()
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::swap(A_d, B_d);
    host_dirty = true;
}

// Pipeline a profundidad 2: el kernel s+1 se solapa con la copia D2H del frame s
// (streams separados), y la CPU drena el chunk c-1 del staging pinned hacia `out`
// mientras la GPU procesa el chunk c. El staging (2 chunks) se reutiliza entre
// llamadas: evita el cudaHostAlloc por llamada, que dominaba el costo por frame.
void CA::run(int steps, uint8_t *out)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);

    const size_t frame = (size_t)width * height;
    const int R2 = 2 * RUN_CHUNK;

    size_t need = (size_t)R2 * frame;
    if (stage_cap < need)
    {
        if (stage_h)
            CUDA_CHECK(cudaFreeHost(stage_h));
        CUDA_CHECK(cudaHostAlloc((void **)&stage_h, need, cudaHostAllocDefault));
        stage_cap = need;
    }

    const int n_chunks = (steps + RUN_CHUNK - 1) / RUN_CHUNK;

    for (int c = 0; c < n_chunks; c++)
    {
        const int half = c & 1;
        const int first = c * RUN_CHUNK;
        const int count = std::min(RUN_CHUNK, steps - first);

        for (int j = 0; j < count; j++)
        {
            const int s = first + j;
            // WAR: el kernel s escribe el búfer que la copia s-2 estuvo leyendo
            if (s >= 2)
                CUDA_CHECK(cudaStreamWaitEvent(stream, c_done[(s - 2) % R2], 0));

            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(k_done[s % R2], stream));

            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, k_done[s % R2], 0));
            CUDA_CHECK(cudaMemcpyAsync(stage_h + ((size_t)half * RUN_CHUNK + j) * frame,
                                       B_d, size, cudaMemcpyDeviceToHost, copy_stream));
            CUDA_CHECK(cudaEventRecord(c_done[s % R2], copy_stream));

            std::swap(A_d, B_d);
        }
        CUDA_CHECK(cudaEventRecord(chunk_done[half], copy_stream));

        if (c > 0)
        {
            const int pc = c - 1;
            const int ph = pc & 1;
            const int pfirst = pc * RUN_CHUNK;
            const int pcount = std::min(RUN_CHUNK, steps - pfirst);
            CUDA_CHECK(cudaEventSynchronize(chunk_done[ph]));
            std::memcpy(out + (size_t)pfirst * frame,
                        stage_h + (size_t)ph * RUN_CHUNK * frame,
                        (size_t)pcount * frame);
        }
    }

    const int lc = n_chunks - 1;
    const int lh = lc & 1;
    const int lfirst = lc * RUN_CHUNK;
    CUDA_CHECK(cudaEventSynchronize(chunk_done[lh]));
    std::memcpy(out + (size_t)lfirst * frame,
                stage_h + (size_t)lh * RUN_CHUNK * frame,
                (size_t)(steps - lfirst) * frame);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));

    // Espacio en host = último frame del historial
    std::memcpy(space, out + (size_t)(steps - 1) * frame, size);
    host_dirty = false;
}

void CA::syncHost()
{
    if (!host_dirty) return;
    CUDA_CHECK(cudaMemcpyAsync(space, A_d, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    host_dirty = false;
}

void CA::print()
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
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
uint8_t *CA::getSpace()             { std::lock_guard<std::mutex> lock(gpu_mutex); syncHost(); return space; }
int CA::getWidth() const            { return width; }
int CA::getHeight() const           { return height; }
int CA::getStates() const           { return states; }
int *CA::getNeighbourhood() const   { return neighbourhood; }
int CA::getNSize() const            { return n_size; }
int *CA::getGroupSizes() const      { return group_sizes; }
int CA::getNGroups() const          { return n_groups; }
int *CA::getRule() const            { return rule; }
int CA::getRSize() const            { return r_size; }
uint8_t CA::getCell(int x, int y)   { std::lock_guard<std::mutex> lock(gpu_mutex); syncHost(); return space[y * width + x]; }

// Setters
void CA::setStates(int new_states)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);

    // Validar r_size ANTES de modificar estados
    int64_t new_r_size_64 = 1;
    for (int g = 0; g < n_groups; g++)
        new_r_size_64 *= (int64_t)(new_states - 1) * group_sizes[g] + 1;
    if (new_r_size_64 > (1 << 24))
        throw std::runtime_error("r_size exceeds memory limit (16777216)");

    states = new_states;
    fixRuleSize();
}

void CA::setNeighbourhood(int *new_neighbourhood, int new_n_size, int *new_group_sizes, int new_n_groups)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);

    if (new_n_size > 64)
        throw std::runtime_error("n_size exceeds memory limit (64)");
    if (new_n_groups > 64)
        throw std::runtime_error("n_groups exceeds memory limit (64)");

    // Validar r_size ANTES de modificar cualquier estado o memoria GPU
    int64_t new_r_size_64 = 1;
    for (int g = 0; g < new_n_groups; g++)
        new_r_size_64 *= (int64_t)(states - 1) * new_group_sizes[g] + 1;
    if (new_r_size_64 > (1 << 24))
        throw std::runtime_error("r_size exceeds memory limit (16777216)");

    if (new_n_size != n_size)
    {
        delete[] neighbourhood;
        neighbourhood = new int[new_n_size * 2];
        n_size = new_n_size;
    }
    std::memcpy(neighbourhood, new_neighbourhood, n_size * 2 * sizeof(int));
    CUDA_CHECK(cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, n_size * 2 * sizeof(int)));

    if (new_n_groups != n_groups)
    {
        delete[] group_sizes;
        group_sizes = new int[new_n_groups];
        n_groups = new_n_groups;
    }
    std::memcpy(group_sizes, new_group_sizes, n_groups * sizeof(int));
    CUDA_CHECK(cudaMemcpyToSymbol(c_group_sizes, group_sizes, n_groups * sizeof(int)));

    updateTileMeta();
    fixRuleSize();
}

void CA::setRule(int *new_rule)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    std::memcpy(rule, new_rule, r_size * sizeof(int));
    for (int i = 0; i < r_size; i++)
        rule[i] %= states;
    CUDA_CHECK(cudaMemcpy(rule_d, rule, (size_t)r_size * sizeof(int), cudaMemcpyHostToDevice));
}

void CA::setSpace(const uint8_t *new_space)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    // No hace falta syncHost: host y device se sobrescriben completos
    std::memcpy(space, new_space, size);
    CUDA_CHECK(cudaMemcpyAsync(A_d, space, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    host_dirty = false;
}

void CA::setCell(int x, int y, int value)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    syncHost();
    space[y * width + x] = (uint8_t)value;
    CUDA_CHECK(cudaMemcpyAsync(&A_d[y * width + x], &space[y * width + x],
                    sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CA::setBoundary(int mode)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    boundary_mode = mode;
}

int CA::getBoundary() const { return boundary_mode; }

void CA::fixRuleSize()
{
    int64_t new_r_size_64 = 1;
    for (int g = 0; g < n_groups; g++)
        new_r_size_64 *= (int64_t)(states - 1) * group_sizes[g] + 1;

    if (new_r_size_64 > (1 << 24))
        throw std::runtime_error("r_size exceeds memory limit (16777216)");

    int new_r_size = (int)new_r_size_64;

    int *new_rule = new int[new_r_size];

    int elements = (r_size < new_r_size) ? r_size : new_r_size;
    for (int i = 0; i < elements; i++)
        new_rule[i] = rule[i] % states;
    for (int i = r_size; i < new_r_size; i++)
        new_rule[i] = 0;

    delete[] rule;
    rule = new_rule;
    r_size = new_r_size;

    cudaFree(rule_d);
    CUDA_CHECK(cudaMalloc(&rule_d, (size_t)r_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rule_d, rule, (size_t)r_size * sizeof(int), cudaMemcpyHostToDevice));

    uploadMultipliers();
}
