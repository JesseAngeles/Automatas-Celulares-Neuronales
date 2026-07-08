// nca.cu

#include "nca.h"
#include <algorithm>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error in " #call ": ") + cudaGetErrorString(_err)); \
} while(0)

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

// REG=true: las activaciones del forward pass viven en un buffer por-hilo
// (local/registros, respaldado por L1) en lugar del scratch global temp[].
// Elimina el tráfico a DRAM del caso dominante (la capa l relee su entrada
// layer_out[l] veces por celda). Requiere max_layer_size <= NCA_REG_BUF.
constexpr int NCA_REG_BUF = 64;

template <bool STOCH, bool REG>
__global__ void NCA_GPU(
    const float *__restrict__ A,
    float *__restrict__ B,
    int width,
    int height,
    int n_size,
    const float *__restrict__ weights,
    const float *__restrict__ biases,
    const int *__restrict__ w_offsets,
    const int *__restrict__ b_offsets,
    const int *__restrict__ layer_in,
    const int *__restrict__ layer_out,
    const int *__restrict__ activations,
    int n_layers,
    float *temp,
    int max_layer_size,
    int out_size,
    int total_weights,
    int total_biases,
    int use_smem,
    int channels,
    float p,
    unsigned long long seed,
    unsigned long long step_idx)
{
    // Pesos y biases del MLP a shared memory (todos los hilos leen los mismos)
    extern __shared__ float smem[];
    const float *Wall = weights;
    const float *ball = biases;
    if (use_smem)
    {
        const int nt  = blockDim.x * blockDim.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = tid; i < total_weights; i += nt)
            smem[i] = weights[i];
        for (int i = tid; i < total_biases; i += nt)
            smem[total_weights + i] = biases[i];
        __syncthreads();  // antes del return de los hilos fuera de la grilla
        Wall = smem;
        ball = smem + total_weights;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height)
        return;

    int cell_id = idy * width + idx;
    const size_t n_cells = (size_t)width * height;

    // Modo estocástico: la celda solo se actualiza con probabilidad p.
    // Philox counter-based: mismo (seed, celda, paso) → mismo valor. STOCH
    // es parámetro de plantilla: la instancia determinista no contiene
    // código de cuRAND (sin costo de registros cuando p = 1.0).
    if constexpr (STOCH)
    {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, (unsigned long long)cell_id, step_idx, &rng);
        if (curand_uniform(&rng) > p)
        {
            for (int c = 0; c < channels; c++)
                B[(size_t)c * n_cells + cell_id] = A[(size_t)c * n_cells + cell_id];
            return;
        }
    }

    // Scratch del forward pass:
    //  - REG=true: buffer por-hilo (local, L1) con stride 1 — sin tráfico
    //    global entre capas.
    //  - REG=false (fallback max_layer_size > NCA_REG_BUF): SoA en temp[],
    //    temp[k * n_cells + cell_id], hilos consecutivos → accesos coalescidos.
    // stride se pliega a constante 1 en la instancia REG (parámetro de plantilla).
    float rbuf[REG ? 2 * NCA_REG_BUF : 1];
    float *cur, *nxt;
    const size_t stride = REG ? 1 : n_cells;
    if constexpr (REG)
    {
        cur = rbuf;
        nxt = rbuf + NCA_REG_BUF;
    }
    else
    {
        cur = temp + cell_id;
        nxt = temp + (size_t)max_layer_size * n_cells + cell_id;
    }

    // Valores del vecindario: los C canales de cada vecino, consecutivos,
    // en el orden de la vecindad (entrada del MLP = n_size * channels)
    for (int i = 0; i < n_size; i++)
    {
        int nx = (idx + c_neighbourhood[i * 2] + width) % width;
        int ny = (idy + c_neighbourhood[i * 2 + 1] + height) % height;
        int nid = ny * width + nx;
        for (int c = 0; c < channels; c++)
            cur[(size_t)(i * channels + c) * stride] = A[(size_t)c * n_cells + nid];
    }

    // Forward pass
    for (int l = 0; l < n_layers; l++)
    {
        const float *W = &Wall[w_offsets[l]];
        const float *b = &ball[b_offsets[l]];

        int in = layer_in[l];
        int out = layer_out[l];

        for (int j = 0; j < out; j++)
        {
            float sum = b[j];
            for (int i = 0; i < in; i++)
                sum += W[j * in + i] * cur[(size_t)i * stride];
            nxt[(size_t)j * stride] = activate(sum, activations[l]);
        }

        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    if (channels > 1) {
        // Multi-canal: la salida del MLP es el vector de C canales de la celda
        for (int c = 0; c < channels; c++)
            B[(size_t)c * n_cells + cell_id] = cur[(size_t)c * stride];
    } else if (out_size == 1) {
        B[cell_id] = cur[0];
    } else {
        int best = 0;
        float best_val = cur[0];
        for (int j = 1; j < out_size; j++) {
            float v = cur[(size_t)j * stride];
            if (v > best_val) {
                best_val = v;
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
    int *in_neighbourhood, int in_n_size,
    int in_channels)
    : width(in_width),
      height(in_height),
      channels(in_channels),
      n_size(in_n_size),
      n_layers(0),
      max_layer_size(0),
      total_weights(0),
      total_biases(0),
      use_smem(false),
      use_reg(false),
      out_size(1),
      update_prob(1.0f),
      rng_seed(0),
      step_count(0),
      nn_ready(false),
      weights_d(nullptr),
      biases_d(nullptr),
      w_offsets_d(nullptr),
      b_offsets_d(nullptr),
      layer_in_d(nullptr),
      layer_out_d(nullptr),
      activation_d(nullptr),
      temp_d(nullptr),
      stage_h(nullptr),
      stage_cap(0)
{
    if (in_channels < 1)
        throw std::runtime_error("channels must be >= 1");

    size = (size_t)width * height * channels * sizeof(float);

    // 32×8: cada warp cubre una fila completa de 32 celdas contiguas
    threads = dim3(32, 8);
    blocks = dim3((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

    space = new float[(size_t)in_width * in_height * channels];
    std::memcpy(space, in_space, size);

    neighbourhood = new int[in_n_size * 2];
    std::memcpy(neighbourhood, in_neighbourhood, in_n_size * 2 * sizeof(int));

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
}

NCA::~NCA()
{
    cudaStreamSynchronize(stream);  // best-effort in destructor, don't throw
    cudaStreamSynchronize(copy_stream);
    cudaFree(A_d);
    cudaFree(B_d);
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

    freeNN();
}

void NCA::launchNN()
{
    size_t smem_bytes = use_smem ? (size_t)(total_weights + total_biases) * sizeof(float) : 0;
#define NCA_ARGS A_d, B_d, width, height, n_size, weights_d, biases_d, \
    w_offsets_d, b_offsets_d, layer_in_d, layer_out_d, activation_d, n_layers, \
    temp_d, max_layer_size, out_size, total_weights, total_biases, \
    use_smem ? 1 : 0, channels, update_prob, rng_seed, step_count
    if (update_prob < 1.0f)
    {
        if (use_reg) NCA_GPU<true,  true ><<<blocks, threads, smem_bytes, stream>>>(NCA_ARGS);
        else         NCA_GPU<true,  false><<<blocks, threads, smem_bytes, stream>>>(NCA_ARGS);
    }
    else
    {
        if (use_reg) NCA_GPU<false, true ><<<blocks, threads, smem_bytes, stream>>>(NCA_ARGS);
        else         NCA_GPU<false, false><<<blocks, threads, smem_bytes, stream>>>(NCA_ARGS);
    }
#undef NCA_ARGS
    step_count++;
}

void NCA::step()
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    if (!nn_ready)
        throw std::runtime_error("Weights not loaded. Call setWeights() first.");

    launchNN();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(space, B_d, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::swap(A_d, B_d);
}

// Mismo pipeline a profundidad 2 que CA::run: kernel s+1 se solapa con la copia
// D2H del frame s y la CPU drena el chunk anterior mientras la GPU procesa el
// siguiente. El estado del device queda continuo (sin re-inyección discreta).
void NCA::run(int steps, float *out)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    if (!nn_ready)
        throw std::runtime_error("Weights not loaded. Call setWeights() first.");

    const size_t frame = (size_t)width * height * channels;  // elementos float por frame
    const int R2 = 2 * RUN_CHUNK;

    size_t need = (size_t)R2 * frame * sizeof(float);
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

            launchNN();
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
                        (size_t)pcount * frame * sizeof(float));
        }
    }

    const int lc = n_chunks - 1;
    const int lh = lc & 1;
    const int lfirst = lc * RUN_CHUNK;
    CUDA_CHECK(cudaEventSynchronize(chunk_done[lh]));
    std::memcpy(out + (size_t)lfirst * frame,
                stage_h + (size_t)lh * RUN_CHUNK * frame,
                (size_t)(steps - lfirst) * frame * sizeof(float));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));

    // Espacio en host = último frame del historial
    std::memcpy(space, out + (size_t)(steps - 1) * frame, size);
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
    std::lock_guard<std::mutex> lock(gpu_mutex);

    if (layer_in[0] != n_size * channels)
        throw std::runtime_error("layer_in[0] must equal n_size * channels");
    if (channels > 1 && layer_out[n_layers - 1] != channels)
        throw std::runtime_error("output layer size must equal channels");

    freeNN();
    this->n_layers = n_layers;
    this->out_size = layer_out[n_layers - 1];
    max_layer_size = 0;

    // Calcular offsets y tamaños totales
    int *w_offsets = new int[n_layers];
    int *b_offsets = new int[n_layers];
    total_weights = 0;
    total_biases = 0;

    for (int i = 0; i < n_layers; i++)
    {
        w_offsets[i] = total_weights;
        b_offsets[i] = total_biases;
        total_weights += layer_in[i] * layer_out[i];
        total_biases += layer_out[i];
        max_layer_size = max(max_layer_size, max(layer_in[i], layer_out[i]));
    }

    // El kernel copia pesos+biases a shared memory si caben (límite por bloque)
    use_smem = ((size_t)(total_weights + total_biases) * sizeof(float) <= 48 * 1024);
    // Forward pass en buffer por-hilo si las capas caben (evita el scratch global)
    use_reg = (max_layer_size <= NCA_REG_BUF);

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

    // El scratch global solo se usa en el fallback (max_layer_size > NCA_REG_BUF)
    if (use_reg)
        temp_d = nullptr;
    else
        cudaMalloc(&temp_d, (size_t)width * height * 2 * max_layer_size * sizeof(float));

    delete[] flat_weights;
    delete[] flat_biases;
    delete[] w_offsets;
    delete[] b_offsets;

    nn_ready = true;
}

// Getters
int NCA::getWidth() const { return width; }
int NCA::getHeight() const { return height; }
int NCA::getChannels() const { return channels; }
float *NCA::getSpace() const { return space; }
int *NCA::getNeighbourhood() const { return neighbourhood; }
int NCA::getNSize() const { return n_size; }
float NCA::getCell(int x, int y, int channel) const
{
    return space[(size_t)channel * width * height + y * width + x];
}

// Setters
void NCA::setSpace(float *new_space)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    std::memcpy(space, new_space, size);
    CUDA_CHECK(cudaMemcpyAsync(A_d, space, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NCA::setCell(int x, int y, float value, int channel)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    size_t idx = (size_t)channel * width * height + y * width + x;
    space[idx] = value;
    CUDA_CHECK(cudaMemcpyAsync(&A_d[idx], &space[idx],
                               sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NCA::setUpdateProb(float p)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    if (!(p > 0.0f && p <= 1.0f))
        throw std::runtime_error("update_prob must be in (0, 1]");
    update_prob = p;
}

float NCA::getUpdateProb() const { return update_prob; }

void NCA::setSeed(unsigned long long seed)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    rng_seed = seed;
    step_count = 0;
}

void NCA::setNeighbourhood(int *new_neighbourhood, int new_n_size)
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    if (new_n_size != n_size)
    {
        delete[] neighbourhood;
        neighbourhood = new int[new_n_size * 2];
        n_size = new_n_size;
    }
    std::memcpy(neighbourhood, new_neighbourhood, n_size * 2 * sizeof(int));
    CUDA_CHECK(cudaMemcpyToSymbol(c_neighbourhood, neighbourhood, n_size * 2 * sizeof(int)));
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