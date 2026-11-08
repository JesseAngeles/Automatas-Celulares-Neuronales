"""Datos para main.py. Por caja de entrenamiento, REALES pares (t, t+1) con el visible real en t y dados al azar,
y PROPIOS pares de una simulación libre del proceso ideal desde la primera imagen; objetivo = paso_ideal.
→ <salida>/<caja>.npy (N, 2, H, W) int8 con estados combinados, y vecindad.txt.

    python datos.py [salida=datos] [reales=20] [propios=20] [crecer=0]
"""
import sys

import numpy as np
import torch

import comun as C


def generar(salida="datos", reales=20, propios=20, crecer=0.0):
    C.CRECER = crecer
    dev = torch.device("cuda")
    maestra = C.Red(C.MAESTRA, dev)
    w = C.pesos_ruido(dev)
    g = torch.Generator(device=dev).manual_seed(1)
    rng = np.random.default_rng(1)
    out = C.CARPETA / salida
    out.mkdir(exist_ok=True)
    for caja in C.cajas_entrenamiento():
        S = torch.from_numpy(C.cargar(caja)).to(dev)
        ts = np.sort(rng.choice(len(S) - 1, reales, replace=False))
        X = C.combinar(S[ts], g, dev)
        Y = C.paso_ideal(maestra, X, w)
        L = C.libre_ideal(maestra, C.combinar(S[:1], g, dev), len(S), w)[0]
        tp = np.sort(rng.choice(len(S) - 1, propios, replace=False))
        par = torch.stack([torch.cat([X, L[tp]]), torch.cat([Y, L[tp + 1]])], 1).cpu().numpy()
        np.save(out / f"{caja}.npy", par)
    nb = C.vecindad_alumna(maestra)
    (C.CARPETA / "vecindad.txt").write_text(" ".join(f"{dx},{dy}" for dx, dy in nb))
    print(f"{salida}: {len(C.cajas_entrenamiento())} cajas × {reales}+{propios} pares, crecer {crecer}, {len(nb)} vecinos")


if __name__ == "__main__":
    a = sys.argv[1:]
    generar(*(a[:1]), *(int(x) for x in a[1:3]), *(float(x) for x in a[3:4]))
