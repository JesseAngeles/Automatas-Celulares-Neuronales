"""Proceso ideal (maestra + dado) con distintos PMIN en las cajas de calibración (entrenamiento) → pmin.csv

    python pmin.py
"""
import csv

import numpy as np
import torch

import comun as C

PMINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15)


def main():
    dev = torch.device("cuda")
    maestra = C.Red(C.MAESTRA, dev)
    w = C.pesos_ruido(dev)
    filas = []
    for pmin in PMINS:
        C.PMIN = pmin
        g = torch.Generator(device=dev).manual_seed(3)
        acum = {}
        for caja in C.CALIBRACION:
            S = torch.from_numpy(C.cargar(caja)).to(dev)
            I = C.libre_ideal(maestra, C.combinar(S[:1].repeat(2, 1, 1), g, dev), len(S), w)
            for k, v in C.resumen(S, (I // C.DADOS).to(torch.int8)).items():
                acum.setdefault(k, []).append(np.mean(v))
        fila = {"pmin": pmin, **{k: round(float(np.mean(v)), 5) for k, v in acum.items()}}
        print(fila, flush=True)
        filas.append(fila)
    with open(C.CARPETA / "pmin.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0]))
        wr.writeheader()
        wr.writerows(filas)


if __name__ == "__main__":
    main()
