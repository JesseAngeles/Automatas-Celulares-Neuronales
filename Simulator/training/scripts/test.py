import numpy as np


# ── Forward pass (replica del kernel NCA en numpy) ────────────────────────────

def nca_step(grid, W0, b0, W1, b1):
    # Vecindad 3×3 en el mismo orden que el entrenamiento:
    # for x in range(-1,2): for y in range(-1,2): frame[i+x][j+y]
    offsets = [(x, y) for x in range(-1, 2) for y in range(-1, 2)]
    patches = np.stack([
        np.roll(np.roll(grid, -x, axis=0), -y, axis=1)
        for x, y in offsets
    ], axis=-1)                                    # (H, W, 9)

    h = patches.reshape(-1, 9) @ W0.T + b0        # (H*W, 64)
    h = np.maximum(0, h)                           # ReLU
    out = h @ W1.T + b1                            # (H*W, 1)
    out = 1 / (1 + np.exp(-out))                   # Sigmoid
    return out.reshape(grid.shape)


# ── Regla de Life (referencia exacta) ─────────────────────────────────────────

def life_step(grid):
    neighbors = sum(
        np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        for dy in [-1, 0, 1] for dx in [-1, 0, 1]
        if not (dy == 0 and dx == 0)
    )
    alive = grid == 1
    return ((neighbors == 3) | (alive & (neighbors == 2))).astype(np.float32)


def print_grid(grid, threshold=0.5):
    for row in grid:
        print("".join("█" if v > threshold else "·" for v in row))


# ── Cargar pesos ───────────────────────────────────────────────────────────────

data = np.load("life_weights.npz")
W0 = data["weights_0"]   # (64, 9)
b0 = data["biases_0"]    # (64,)
W1 = data["weights_1"]   # (1, 64)
b1 = data["biases_1"]    # (1,)

print(f"Pesos cargados — capas: {W0.shape}, {W1.shape}")

# ── Grid inicial: glider ───────────────────────────────────────────────────────

grid = np.zeros((20, 20), dtype=np.float32)
grid[1, 2] = 1
grid[2, 3] = 1
grid[3, 1] = 1
grid[3, 2] = 1
grid[3, 3] = 1

life = grid.copy()
nca  = grid.copy()

# ── Comparación paso a paso ────────────────────────────────────────────────────

STEPS = 8
print(f"\n{'Paso':>5}  {'Celdas distintas':>18}  {'Match':>5}")
print("─" * 35)

errors_final = 0
for step in range(STEPS):
    life     = life_step(life)
    nca_raw  = nca_step(nca, W0, b0, W1, b1)
    nca      = (nca_raw > 0.5).astype(np.float32)

    errors = int(np.sum(np.abs(life - nca)))
    errors_final = errors
    match  = "✓" if errors == 0 else "✗"
    print(f"{step+1:>5}  {errors:>18}  {match:>5}")

print()
print("=== Life (paso final) ===")
print_grid(life)
print()
print("=== NCA  (paso final) ===")
print_grid(nca)

total = 20 * 20
print(f"\nPrecisión paso {STEPS}: {(1 - errors_final/total)*100:.2f}%")
