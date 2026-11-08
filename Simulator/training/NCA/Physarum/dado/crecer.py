"""Proceso ideal (maestra + dado + localidad) con distintos empujones de crecimiento CRECER en las cajas de
calibración (entrenamiento) → crecer.csv.  python crecer.py"""
import csv
import sys

import numpy as np
import torch

import comun as C

VALORES = tuple(float(v) for v in sys.argv[1:]) or (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    dev = torch.device("cuda")
    maestra = C.Red(C.MAESTRA, dev)
    w = C.pesos_ruido(dev)
    filas = []
    for b in VALORES:
        C.CRECER = b
        g = torch.Generator(device=dev).manual_seed(3)
        acum = {}
        for caja in C.CALIBRACION:
            S = torch.from_numpy(C.cargar(caja)).to(dev)
            I = C.libre_ideal(maestra, C.combinar(S[:1].repeat(2, 1, 1), g, dev), len(S), w)
            for k, v in C.resumen(S, (I // C.DADOS).to(torch.int8)).items():
                acum.setdefault(k, []).append(np.mean(v))
        fila = {"crecer": b, **{k: round(float(np.mean(v)), 5) for k, v in acum.items()}}
        print(fila, flush=True)
        filas.append(fila)
    with open(C.CARPETA / "crecer.csv", "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0]))
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(filas)


if __name__ == "__main__":
    main()
