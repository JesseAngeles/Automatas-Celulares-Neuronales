"""Junta los resultados.csv de todas las opciones en resumen.csv (promedio y desviación entre semillas).

Una fila por (opción, variante, ronda) con las métricas principales y su proporción contra la copia:
    mae_rel = mae_c / copia_mae_c, rollout_rel = rollout_mae_c / copia_rollout_mae_c (< 1 = mejor que copiar)

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/opciones/resumen.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OPC = Path(__file__).resolve().parent
METRICAS = ["exact_temp", "mae_c", "encendido", "precision_encendido", "rollout_mae_c", "rollout_iou_fuego",
            "rollout_iou_quemado"]
COPIA = ["copia_exact_temp", "copia_mae_c", "copia_rollout_mae_c", "copia_rollout_iou_fuego",
         "copia_rollout_iou_quemado"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    grupos = defaultdict(list)
    for archivo in sorted(OPC.glob("*/resultados.csv")):
        for r in csv.DictReader(open(archivo)):
            grupos[(r["opcion"], r["variante"], r["ronda"])].append(r)
    filas = []
    for (opcion, variante, ronda), rs in grupos.items():
        fila = {"opcion": opcion, "variante": variante, "ronda": ronda, "n": len(rs),
                "paso_frames": rs[0].get("paso_frames", "")}
        for m in METRICAS:
            v = np.array([num(r.get(m)) for r in rs])
            fila[m] = round(float(np.nanmean(v)), 4)
            fila[f"{m}_std"] = round(float(np.nanstd(v)), 4)
        for m in COPIA:
            fila[m] = round(float(np.nanmean([num(r.get(m)) for r in rs])), 4)
        fila["mae_rel"] = round(fila["mae_c"] / fila["copia_mae_c"], 4)
        fila["rollout_rel"] = round(fila["rollout_mae_c"] / fila["copia_rollout_mae_c"], 4)
        filas.append(fila)
    filas.sort(key=lambda f: (f["ronda"], f["opcion"], f["variante"]))
    with open(OPC / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f"{f['ronda']:8s} {f['opcion']:16s} {f['variante']:12s} n={f['n']} mae {f['mae_c']:6.2f} "
              f"({f['mae_rel']:.3f}) rollout {f['rollout_mae_c']:6.2f}±{f['rollout_mae_c_std']:5.2f} "
              f"({f['rollout_rel']:.3f}) enc {f['encendido']:.3f} prec {f['precision_encendido']:.3f} "
              f"IoU fuego {f['rollout_iou_fuego']:.3f} (copia {f['copia_rollout_iou_fuego']:.3f}) "
              f"IoU quemado {f['rollout_iou_quemado']:.3f} (copia {f['copia_rollout_iou_quemado']:.3f})")


if __name__ == "__main__":
    main()
