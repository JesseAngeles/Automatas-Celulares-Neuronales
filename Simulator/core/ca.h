// ca.h

#pragma once

#include <cstdint>
#include <iostream>
#include <cstring>
#include <mutex>
#include <cuda_runtime.h>

enum BoundaryMode {
    BOUNDARY_OPEN       = 0,
    BOUNDARY_PERIODIC   = 1,
    BOUNDARY_REFLECTIVE = 2,
    BOUNDARY_REFLEXIVE  = 3
};

class CA
{

public:
    CA(
        int width,
        int height,
        uint8_t *space,
        int states,
        int *neighbourhood, int n_size,
        int *group_sizes, int n_groups,
        int *rule, int r_size);
    ~CA();

    void step();
    void run(int steps, uint8_t *out);  // escribe (steps, H, W) en out (host)
    void print();

    // Getters
    uint8_t *getSpace();          // no-const: sincroniza device→host si hace falta
    int getWidth() const;
    int getHeight() const;
    int getStates() const;
    int *getNeighbourhood() const;
    int getNSize() const;
    int *getGroupSizes() const;
    int getNGroups() const;
    int *getRule() const;
    int getRSize() const;
    uint8_t getCell(int x, int y);
    int getBoundary() const;

    // Setters
    void setSpace(const uint8_t *);
    void setStates(int);
    void setNeighbourhood(int *, int, int *, int);
    void setRule(int *);
    void setCell(int, int, int);
    void setBoundary(int mode);

private:
    // Frames por chunk en el pipeline de run(); el staging pinned aloja 2 chunks
    static constexpr int RUN_CHUNK = 8;

    uint8_t *space;
    int width;
    int height;

    int states;

    int *neighbourhood;
    int n_size;

    int *group_sizes;
    int n_groups;

    int *rule;
    int r_size;
    int *rule_d;    // device copy of rule (global memory, no size limit)

    int boundary_mode;

    // host_dirty = true cuando el buffer del host está desfasado frente a A_d
    bool host_dirty;

    // Radio máximo |dx|,|dy| de la vecindad; habilita el kernel con tile en smem
    int rx, ry;
    bool tiled;
    size_t tile_bytes;

    size_t size;
    uint8_t *A_d, *B_d;
    cudaStream_t stream;
    dim3 threads;
    dim3 blocks;

    // run(): staging pinned reutilizable (doble búfer) + stream/eventos de copia
    uint8_t *stage_h;
    size_t stage_cap;
    cudaStream_t copy_stream;
    cudaEvent_t k_done[2 * RUN_CHUNK];
    cudaEvent_t c_done[2 * RUN_CHUNK];
    cudaEvent_t chunk_done[2];

    // run() corre sin GIL desde el hilo de prefetch: serializar el acceso a GPU
    std::mutex gpu_mutex;

    void fixRuleSize();
    void syncHost();           // device → host si host_dirty (llamar con lock tomado)
    void launch();             // lanza el kernel con boundary especializado en compile-time
    void updateTileMeta();     // recalcula rx/ry/tiled tras cambiar la vecindad
    void uploadMultipliers();  // multiplicadores del índice de regla a constant memory
};
