"""Genera evoluciones de NCA entrenadas con el simulador CUDA (CA.NCA), igual que la GUI.

Por cada pesos <nombre>.npz:
  <nombre>_evo.npy/.mat        configuración de 69 × 69 con los 512 patrones (CA/B3S23/base) y 1 paso
  <nombre>/test_{i}.npy/.mat   10 condiciones iniciales aleatorias de 69 × 69 (densidad 0.5, semillas 0..9), 50 pasos

Uso (desde Simulator/training): python scripts/generar_evoluciones.py <npz> [<npz> ...]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, "../GUI")
from CA import NCA  # noqa: E402

LADO, PASOS, N_TEST = 69, 50, 10


def condiciones_aleatorias():
    return [(np.random.default_rng(s).random((LADO, LADO)) < 0.5).astype(np.int32) for s in range(N_TEST)]


def simular(pesos: dict, inicial: np.ndarray, pasos: int) -> np.ndarray:
    n_capas = int(pesos["layer_in"].shape[0])
    vecindad = pesos["neighbourhood"].astype(np.int32)
    alto, ancho = inicial.shape
    nca = NCA(ancho, alto, inicial.astype(np.float32).flatten(), vecindad, len(vecindad))
    nca.set_weights([pesos[f"weights_{i}"] for i in range(n_capas)],
                    [pesos[f"biases_{i}"] for i in range(n_capas)],
                    pesos["layer_in"].tolist(), pesos["layer_out"].tolist(), pesos["activations"].tolist())
    binario = int(pesos["layer_out"][-1]) == 1
    cuadros = [inicial.astype(np.int32)]
    for _ in range(pasos):
        nca.step()
        crudo = nca.get_space()
        if binario:
            estado = (crudo > 0.5).astype(np.int32)
            nca.set_space(estado.astype(np.float32))
        else:
            estado = crudo.astype(np.int32)
            nca.set_space(crudo)
        cuadros.append(estado)
    return np.stack(cuadros).astype(np.int32)


def guardar(base: Path, evolucion: np.ndarray):
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(base) + ".npy", evolucion)
    scipy.io.savemat(str(base) + ".mat", {"evolution": evolucion})


def generar(ruta_npz: str, base_inicial: np.ndarray, aleatorias: list):
    ruta = Path(ruta_npz)
    pesos = dict(np.load(ruta))
    nombre = ruta.with_suffix("")
    guardar(Path(str(nombre) + "_evo"), simular(pesos, base_inicial, 1))
    for i, inicial in enumerate(aleatorias):
        guardar(nombre / f"test_{i}", simular(pesos, inicial, PASOS))


if __name__ == "__main__":
    base_inicial = np.load("CA/B3S23/base.npy")[0]
    aleatorias = condiciones_aleatorias()
    for p in sys.argv[1:]:
        generar(p, base_inicial, aleatorias)
        print(f"evoluciones → {p}", flush=True)
