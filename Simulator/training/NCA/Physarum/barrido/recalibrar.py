"""Recalibra el sesgo de algunas redes con las 20 cajas de entrenamiento (en vez de 8) y las evalúa igual.

    python recalibrar.py <nombre> [<nombre> …]

No toca redes/<nombre>/: escribe en redes/<nombre>_cal20/ (red.npz y config.json copiados, red_gui.npz, libre_*.npy)
y una fila por red en resultados_cal20.csv.
"""
import csv
import json
import shutil
import sys

import barrido as B


def recalibrar(nombre):
    origen = B.CARPETA / "redes" / nombre
    destino = B.CARPETA / "redes" / f"{nombre}_cal20"
    destino.mkdir(exist_ok=True)
    for f in ("red.npz", "config.json"):
        shutil.copy(origen / f, destino / f)
    cfg = {**B.BASE, **json.loads((origen / "config.json").read_text())}
    B.CALIBRACION = B.cajas_entrenamiento(cfg)
    fila = next(f for f in csv.DictReader(open(B.CARPETA / "resultados_v3.csv")) if f["nombre"] == nombre)
    ent = {"perdida": float(fila["perdida"]), "muestras": int(fila["muestras"]),
           "t_entrenar_s": float(fila["t_entrenar_s"])}
    B.evaluar(f"{nombre}_cal20", cfg, "cal20", ent, int(fila["por_caja"]))


if __name__ == "__main__":
    B.RESULTADOS = B.CARPETA / "resultados_cal20.csv"
    for n in sys.argv[1:]:
        recalibrar(n)
