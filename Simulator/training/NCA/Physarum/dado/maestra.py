"""Paso A: qué comportamiento imitar. Simulación libre de la maestra muestreando con temperatura T y ruido
uniforme correlacionado en el espacio (ruido gaussiano suavizado con sigma, llevado a uniforme), en las cajas de
calibración (entrenamiento) y de prueba. → maestra.csv

    python maestra.py
"""
import csv
import itertools

import numpy as np
import torch

import comun as C

TEMPERATURAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
SIGMAS = (0.0, 1.0, 2.0)
REPETICIONES = 2


def uniforme(shape, sigma, g, dev):
    z = torch.randn(shape, generator=g, device=dev)
    if sigma > 0:
        r = int(3 * sigma)
        x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * sigma ** 2))
        k = k / k.norm()                                                   # varianza 1 después de suavizar
        z = torch.nn.functional.conv2d(z.unsqueeze(1), k.view(1, 1, 1, -1), padding=(0, r))
        z = torch.nn.functional.conv2d(z, k.view(1, 1, -1, 1), padding=(r, 0)).squeeze(1)
    return 0.5 * (1 + torch.erf(z / 2 ** 0.5))


@torch.no_grad()
def libre(red, S0, T, temp, sigma, g):
    """(B, T, H, W): B = REPETICIONES simulaciones muestreando por CDF inversa con u uniforme."""
    E = S0.unsqueeze(0).repeat(REPETICIONES, 1, 1)
    ev = [E]
    for _ in range(1, T):
        p = (red.logits(E) / temp).softmax(-1)
        u = uniforme(E.shape, sigma, g, E.device)
        E = (p.cumsum(-1) <= u.unsqueeze(-1)).sum(-1).clamp(max=p.shape[-1] - 1).to(torch.int8)
        ev.append(E)
    return torch.stack(ev, 1)


def main():
    dev = torch.device("cuda")
    red = C.Red(C.MAESTRA, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    filas = []
    for (temp, sigma), (grupo, cajas) in itertools.product(itertools.product(TEMPERATURAS, SIGMAS),
                                                            (("cal", C.CALIBRACION), ("prueba", C.PRUEBA))):
        acum = {}
        for caja in cajas:
            S = torch.from_numpy(C.cargar(caja)).to(dev)
            m = C.resumen(S, libre(red, S[0], len(S), temp, sigma, g))
            for k, v in m.items():
                acum.setdefault(k, []).append(np.mean(v))
        fila = {"T": temp, "sigma": sigma, "grupo": grupo, **{k: round(float(np.mean(v)), 4) for k, v in acum.items()}}
        print(fila, flush=True)
        filas.append(fila)
    with open(C.CARPETA / "maestra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
