"""Regla del dado oculto (2 bits a, b por célula, dado = 2a + b), con frontera abierta (fuera = 0):
    a' = a_O XOR (a_C OR a_E) XOR b_C        (regla 30 en x, acoplada con b)
    b' = b_N XOR (b_C OR b_S) XOR a_C        (regla 30 en y, acoplada con a)
O = (-1, 0), E = (+1, 0), N = (0, -1), S = (0, +1) en (dx, dy).  python regla_dado.py → estadísticas."""
import numpy as np

DADOS = 4


def vecino(X, dx, dy):
    """X[..., y + dy, x + dx] con frontera abierta (0 fuera)."""
    H, W = X.shape[-2:]
    P = np.pad(X, [(0, 0)] * (X.ndim - 2) + [(1, 1), (1, 1)])
    return P[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


def paso(D):
    a, b = D >> 1, D & 1
    a2 = vecino(a, -1, 0) ^ (a | vecino(a, 1, 0)) ^ b
    b2 = vecino(b, 0, -1) ^ (b | vecino(b, 0, 1)) ^ a
    return ((a2 << 1) | b2).astype(np.int8)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D = rng.integers(0, 4, (160, 160)).astype(np.int8)
    hist = [D]
    for t in range(1, 400):
        D = paso(D)
        hist.append(D)
    H = np.stack(hist).astype(int)
    for t in (0, 50, 200, 399):
        f = np.bincount(H[t].ravel(), minlength=4) / H[t].size
        print(t, "frecuencias", np.round(f, 3),
              "corr x", round(np.corrcoef(H[t][:, :-1].ravel(), H[t][:, 1:].ravel())[0, 1], 3),
              "corr y", round(np.corrcoef(H[t][:-1].ravel(), H[t][1:].ravel())[0, 1], 3),
              "corr t", round(np.corrcoef(H[t - 1 if t else 1].ravel(), H[t].ravel())[0, 1], 3))
    # ¿se repite? estados iguales en todo el tablero
    vistos = {}
    for t, X in enumerate(hist):
        k = X.tobytes()
        if k in vistos:
            print("ciclo", vistos[k], "→", t); break
        vistos[k] = t
    else:
        print("sin ciclo en 400 pasos")
    # bordes: frecuencias en las 3 columnas de cada lado tras 200 pasos
    print("bordes x0", np.round(np.bincount(H[200:, :, :3].ravel(), minlength=4) / H[200:, :, :3].size, 3),
          "x-1", np.round(np.bincount(H[200:, :, -3:].ravel(), minlength=4) / H[200:, :, -3:].size, 3))
    D0 = np.zeros((160, 160), np.int8)
    for t in range(300):
        D0 = paso(D0)
    print("desde ceros tras 300:", np.bincount(D0.ravel(), minlength=4))
