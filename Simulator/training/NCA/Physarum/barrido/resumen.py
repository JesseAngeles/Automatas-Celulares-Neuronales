"""Tablas en markdown de resultados_v3.csv (para RESULTADOS.md).

    python resumen.py [n]      → las n mejores por puntaje, la fase 1 completa y las semillas
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent
filas = list(csv.DictReader(open(CARPETA / "resultados_v3.csv")))
for f in filas:
    for k, v in f.items():
        try:
            f[k] = float(v)
        except ValueError:
            pass

COLS = [("nombre", "red", "{}"), ("estados", "estados", "{:.0f}"), ("vecindad", "vecindad", "{}"),
        ("capas", "capas", "{}"), ("paso_min", "paso (min)", "{:.0f}"), ("sesgo", "sesgo", "{:g}"),
        ("calibracion", "puntaje calib.", "{:.3f}"), ("puntaje", "puntaje prueba", "{:.3f}"),
        ("puntaje_b0", "prueba b = 0", "{:.3f}"), ("iou", "IoU", "{:.3f}"), ("iou_24h", "IoU 24 h", "{:.3f}"),
        ("area_red_fin", "área final red", "{:.3f}"), ("area_real_fin", "área final real", "{:.3f}")]


def tabla(fs):
    print("| " + " | ".join(t for _, t, _ in COLS) + " |")
    print("|" + "|".join("---" if k in ("nombre", "vecindad", "capas") else "---:" for k, _, _ in COLS) + "|")
    for f in fs:
        print("| " + " | ".join(fmt.format(f[k]) for k, _, fmt in COLS) + " |")
    print()


n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print("## Mejores por puntaje de prueba\n")
tabla(sorted(filas, key=lambda f: -f["puntaje"])[:n])
print("## Fase 1 (un factor a la vez)\n")
tabla([f for f in filas if f["fase"] == "fase1"])
print("## Semillas (42, 43, 44)\n")
grupos = defaultdict(list)
for f in filas:
    base = f["nombre"].rsplit("_s", 1)[0] if f["nombre"][-4:-2] == "_s" else f["nombre"]
    grupos[base].append(f)
print("| red | semillas | puntaje prueba (media ± desv.) | puntaje calib. (media) | área final red (media) |")
print("|---|---:|---:|---:|---:|")
for base, fs in sorted(grupos.items(), key=lambda kv: -np.mean([f["puntaje"] for f in kv[1]])):
    if len(fs) > 1:
        p = [f["puntaje"] for f in fs]
        print(f"| {base} | {len(fs)} | {np.mean(p):.3f} ± {np.std(p):.3f} | "
              f"{np.mean([f['calibracion'] for f in fs]):.3f} | {np.mean([f['area_red_fin'] for f in fs]):.3f} |")
