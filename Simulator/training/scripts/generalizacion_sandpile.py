"""Generalización de las redes de sandpile a datos que no vieron en el entrenamiento.

Las redes se entrenaron con CA/Sandpile/evo.mat (25 × 25, un generador al centro, espacio vacío). Aquí se evalúan con
CA/Sandpile/2generadores/evo.mat (25 × 25 vacío con 2 generadores juntos al centro),
CA/Sandpile/4generadores/evo.mat (25 × 25 vacío con 4 generadores aleatorios) y con
CA/Sandpile/4generadores_50x50/ (exportado de notebooks/sandpile.nb con SeedRandom[42]):
    evos.npy     (100, 1501, 50, 50): 4 generadores en posiciones aleatorias y alturas iniciales aleatorias 0–7
    evoCero.npy  (2001, 50, 50):      espacio vacío de 50 × 50 con un generador al centro

Con vecindad de Von Neumann y 17 estados solo hay 17^5 patrones posibles, así que se evalúa la red una vez en todos
ellos y cada celda se clasifica buscando su patrón en esa tabla.

Uso (desde Simulator/training):
    python scripts/generalizacion_sandpile.py     → NCA/Sandpile/generalizacion.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).parent))
from barrido_sandpile import ACT_TORCH, DATA, GENERADOR, SALIDA, VECINDAD, entrada

CARPETA = Path("NCA/Sandpile")
DATOS = Path("CA/Sandpile/4generadores_50x50")
CUATRO = Path("CA/Sandpile/4generadores/evo.mat")    # 25 × 25 vacío con 4 generadores aleatorios
DOS = Path("CA/Sandpile/2generadores/evo.mat")       # 25 × 25 vacío con 2 generadores juntos al centro
SALIDA_CSV = CARPETA / "generalizacion.csv"
BASE = SALIDA ** np.arange(len(VECINDAD))       # código del patrón = Σ estado_i · 17^i


def codigos(evo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Código del patrón de vecindad de cada celda en cada paso (frontera abierta) y su estado siguiente."""
    src = evo[:-1].astype(np.int64)
    T1, H, W = src.shape
    padded = np.pad(src, ((0, 0), (1, 1), (1, 1)))
    c = np.zeros((T1, H, W), dtype=np.int64)
    for i, (dx, dy) in enumerate(VECINDAD):
        c += padded[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] * BASE[i]
    return c.ravel(), evo[1:].ravel()


def tabla_predicciones(pesos: Path) -> np.ndarray:
    """Estado que predice la red (argmax) para cada uno de los 17^5 patrones."""
    w = np.load(pesos)
    todos = np.arange(SALIDA ** len(VECINDAD))
    x = entrada(w, torch.from_numpy(((todos[:, None] // BASE) % SALIDA).astype(np.float32)))
    with torch.no_grad():
        for i, act in enumerate(w["activations"]):
            x = x @ torch.from_numpy(w[f"weights_{i}"]).T + torch.from_numpy(w[f"biases_{i}"])
            x = ACT_TORCH[int(act)](x)
    return x.argmax(1).numpy().astype(np.uint8)


def conjuntos():
    """(nombre, generador de (códigos, objetivo)) para cada conjunto de datos; las muestras se leen una a la vez."""
    entrenamiento = scipy.io.loadmat(DATA)["evolution"]
    cuatro = scipy.io.loadmat(CUATRO)["evolution"]
    dos = scipy.io.loadmat(DOS)["evolution"]
    evos = np.load(DATOS / "evos.npy", mmap_mode="r")
    evo_cero = np.load(DATOS / "evoCero.npy")
    return [("entrenamiento 25x25, 1 generador", lambda: [codigos(entrenamiento)]),
            ("vacio 25x25, 2 generadores juntos", lambda: [codigos(dos)]),
            ("vacio 25x25, 4 generadores", lambda: [codigos(cuatro)]),
            ("vacio 50x50, 1 generador", lambda: [codigos(evo_cero)]),
            ("aleatorio 50x50, 4 generadores", lambda: (codigos(np.asarray(e)) for e in evos))]


def main():
    vistos = np.zeros(SALIDA ** len(VECINDAD), dtype=bool)
    vistos[np.unique(codigos(scipy.io.loadmat(DATA)["evolution"])[0])] = True

    redes = sorted(p for p in CARPETA.rglob("*-relu_17-linear.npz") if "ajuste" not in p.parts)
    tablas = {p: tabla_predicciones(p) for p in redes}

    filas = []
    for nombre, muestras in conjuntos():
        # Conteos por patrón: cuántas celdas tienen cada patrón y con qué estado siguiente.
        # La regla es determinista, así que cada patrón tiene un único estado siguiente.
        n_patron = np.zeros(SALIDA ** len(VECINDAD), dtype=np.int64)
        destino = np.full(SALIDA ** len(VECINDAD), -1, dtype=np.int64)
        for c, y in muestras():
            n_patron += np.bincount(c, minlength=len(n_patron))
            destino[c] = y
        presentes = n_patron > 0
        gen = presentes & (destino >= GENERADOR)
        nuevos = presentes & ~vistos
        celdas, celdas_gen, celdas_nuevas = n_patron.sum(), n_patron[gen].sum(), n_patron[nuevos].sum()
        print(f"{nombre}: {celdas} celdas, {presentes.sum()} patrones, "
              f"{nuevos.sum()} no vistos en entrenamiento ({100 * celdas_nuevas / celdas:.3f} % de las celdas)",
              flush=True)
        for p in redes:
            bien = tablas[p] == destino
            mal = presentes & ~bien
            filas.append(dict(
                datos=nombre, red=str(p.relative_to(CARPETA)),
                exactitud=n_patron[presentes & bien].sum() / celdas,
                exactitud_generador=n_patron[gen & bien].sum() / celdas_gen,
                exactitud_no_vistos=n_patron[nuevos & bien].sum() / celdas_nuevas if celdas_nuevas else "",
                patrones=int(presentes.sum()), patrones_no_vistos=int(nuevos.sum()),
                patrones_mal=int(mal.sum()), patrones_mal_no_vistos=int((mal & nuevos).sum()),
                celdas=int(celdas), celdas_mal=int(n_patron[mal].sum())))

    with open(SALIDA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Resumen → {SALIDA_CSV}", flush=True)


if __name__ == "__main__":
    main()
