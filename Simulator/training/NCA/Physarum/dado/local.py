"""Proceso ideal con y sin la regla de localidad (LOCAL) en las cajas de calibración → local.csv, y cuadros de
008-dish5 (prueba) en local.png.  python local.py"""
import csv

import numpy as np
import torch
from PIL import Image

import comun as C
import video as V


def main():
    dev = torch.device("cuda")
    maestra = C.Red(C.MAESTRA, dev)
    w = C.pesos_ruido(dev)
    filas, cuadros = [], []
    for local in (False, True):
        C.LOCAL = local
        g = torch.Generator(device=dev).manual_seed(3)
        acum = {}
        for caja in C.CALIBRACION:
            S = torch.from_numpy(C.cargar(caja)).to(dev)
            I = C.libre_ideal(maestra, C.combinar(S[:1].repeat(2, 1, 1), g, dev), len(S), w)
            for k, v in C.resumen(S, (I // C.DADOS).to(torch.int8)).items():
                acum.setdefault(k, []).append(np.mean(v))
        fila = {"local": local, **{k: round(float(np.mean(v)), 5) for k, v in acum.items()}}
        print(fila, flush=True)
        filas.append(fila)
        S = torch.from_numpy(C.cargar(C.PRUEBA[2])).to(dev)
        I = (C.libre_ideal(maestra, C.combinar(S[:1], g, dev), len(S), w)[0] // C.DADOS).cpu().numpy()
        T = len(S)
        if not cuadros:
            cuadros.append(S.cpu().numpy())
        cuadros.append(I)
    T = len(cuadros[0])
    img = np.concatenate([np.concatenate([np.pad(V.RGB[E[int(f * (T - 1))]], ((2, 2), (2, 2), (0, 0)),
                                                 constant_values=255) for f in (0.05, 0.1, 0.3, 0.6, 1.0)], 1)
                          for E in cuadros], 0)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(C.CARPETA / "local.png")
    with open(C.CARPETA / "local.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0]))
        wr.writeheader()
        wr.writerows(filas)


if __name__ == "__main__":
    main()
