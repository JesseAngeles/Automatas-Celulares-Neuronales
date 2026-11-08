"""Vecindades dispersas circulares de anillos (comun.ANILLOS) con la configuración de la mejor red del barrido
(mejor.json: red, entrada, tormentas y transiciones), semillas 42, 1 y 2. Las corridas van a resultados.csv como fase 4
y se comparan con la mejor red y sus semillas de la fase 3.

Escribe anillos.csv (promedio y rango de las 3 semillas) y libre/<clave>.npy (semilla 42) en esta carpeta.

Uso (desde la raíz del proyecto):
    Simulator/GUI/venv/bin/python Simulator/training/NCA/SEVIR/barrido/anillos.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrido as b                                                 # noqa: E402
import comun as c                                                   # noqa: E402

SEMILLAS = (42, 1, 2)
METRICAS = ["exactitud", "area", "libre30_area", "libre60_area", "libre120_area", "libre240_area",
            "libre120_proporcion"]


def main():
    c.VECINDADES.update(c.ANILLOS)
    mejor = json.loads((b.CARPETA / "mejor.json").read_text())
    red, entrada = mejor["red"], mejor["entrada"]
    t, k = int(mejor["tormentas"]), int(mejor["transiciones"]) // int(mejor["tormentas"])
    b.log(f"Anillos con {red} {entrada}, {t} tormentas × {k} transiciones")
    run = b.Corrida(torch.device("cuda"))
    grupos = {v: [run.correr(4, v, red, entrada, t, k, semilla=s, libre=s == 42) for s in SEMILLAS]
              for v in c.ANILLOS}

    hechos = b.hechos()
    base = f"{mejor['vecindad']}_{red}_{entrada}_t{t}_k{k}_e7"
    grupos[mejor["vecindad"]] = [hechos[f"f2_{base}_s42_ep30"]] + [hechos[f"f3_{base}_s{s}_ep30"] for s in (1, 2)]

    with open(b.CARPETA / "anillos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vecindad", "n_vecinos", "alcance"] + [f"{m}_{x}" for m in METRICAS for x in ("prom", "min", "max")])
        for v, filas in grupos.items():
            nb = c.VECINDADES[v]
            fila = [v, len(nb), int(np.abs(nb).max())]
            for m in METRICAS:
                x = [float(r[m]) for r in filas]
                fila += [round(np.mean(x), 5), min(x), max(x)]
            w.writerow(fila)
    b.log("Anillos terminados")


if __name__ == "__main__":
    main()
