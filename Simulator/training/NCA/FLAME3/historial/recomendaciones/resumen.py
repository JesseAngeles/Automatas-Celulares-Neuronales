"""Junta los resultados.csv de los experimentos recomendados en resumen.csv (promedio y desviación entre semillas).

Incluye como referencia, con las mismas métricas, las filas de ../opciones (0_base sin_p2 y 7_resolucion200 9×9)
unidas a sus métricas de propagación de 1_2_propagacion. Proporciones contra la copia (< 1 = mejor que copiar):
    mae_rel = mae_c / copia_mae_c, rollout_rel = rollout_mae_c / copia_rollout_mae_c,
    llegada_rel = llegada_mae_min / copia_llegada_mae_min

Uso (desde Simulator/training):
    python NCA/FLAME3/historial/recomendaciones/resumen.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REC = Path(__file__).resolve().parent
OPC = REC.parent / "opciones"
METRICAS = ["exact_temp", "mae_c", "encendido", "precision_encendido", "rollout_mae_c", "rollout_iou_quemado",
            "llegada_mae_min", "llegada_sesgo_min", "area_quemada_rel", "iou_quemado_nuevo", "fss_quemado_1",
            "fss_quemado_3", "fss_quemado_9", "fss_fuego_3"]
COPIA = ["copia_mae_c", "copia_rollout_mae_c", "copia_llegada_mae_min", "copia_fss_fuego_3"]
SALTA = {"1_3_error", "4_2_quemado_rgb", "5_1_memoria"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def referencias():
    prop = {(r["opcion"], r["variante"], r["ronda"], r["semilla"]): r
            for r in csv.DictReader(open(REC / "1_2_propagacion" / "resultados.csv"))}
    for archivo, opcion, variante, nombre in (("0_base", "0_base", "sin_p2", "referencia_sin_p2"),
                                              ("7_resolucion200", "7_resolucion200", "200_9x9", "referencia_200_9x9"),
                                              ("1_ensamble", "1_ensamble", "sin_p2", "referencia_ensamble"),
                                              ("6_multipaso", "6_multipaso", "K4", "referencia_multipaso_K4"),
                                              ("2_estocastico", "2_estocastico", "esperado", "referencia_esperado")):
        for r in csv.DictReader(open(OPC / archivo / "resultados.csv")):
            if r["variante"] != variante:
                continue
            p = prop.get((opcion, variante, r["ronda"], r["semilla"]))
            if p:
                yield {**r, **{k: v for k, v in p.items() if k.startswith(("llegada", "area", "iou_quemado_nuevo",
                                                                             "fss", "copia_llegada", "copia_fss"))},
                       "experimento": "0_referencia", "variante": nombre}


def main():
    grupos = defaultdict(list)
    for r in referencias():
        grupos[(r["experimento"], r["variante"], r["ronda"])].append(r)
    for archivo in sorted(REC.glob("*/resultados.csv")):
        if archivo.parent.name in SALTA or archivo.parent.name == "1_2_propagacion":
            continue
        for r in csv.DictReader(open(archivo)):
            grupos[(r["experimento"], r["variante"], r["ronda"])].append(r)
    filas = []
    for (exp, variante, ronda), rs in grupos.items():
        f = {"experimento": exp, "variante": variante, "ronda": ronda, "n": len(rs)}
        for m in METRICAS:
            v = np.array([num(r.get(m)) for r in rs])
            f[m] = round(float(np.nanmean(v)), 4) if np.isfinite(v).any() else np.nan
            f[f"{m}_std"] = round(float(np.nanstd(v)), 4) if np.isfinite(v).any() else np.nan
        for m in COPIA:
            f[m] = round(float(np.nanmean([num(r.get(m)) for r in rs])), 4)
        f["mae_rel"] = round(f["mae_c"] / f["copia_mae_c"], 4)
        f["rollout_rel"] = round(f["rollout_mae_c"] / f["copia_rollout_mae_c"], 4)
        f["llegada_rel"] = round(f["llegada_mae_min"] / f["copia_llegada_mae_min"], 4)
        filas.append(f)
    filas.sort(key=lambda f: (f["ronda"], f["experimento"], f["variante"]))
    with open(REC / "resumen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f"{f['ronda']:8s} {f['experimento']:18s} {f['variante']:26s} n={f['n']:<2d} mae {f['mae_rel']:.3f} "
              f"rollout {f['rollout_rel']:.3f} enc {f['encendido']:.3f} prec {f['precision_encendido']:.3f} "
              f"llegada {f['llegada_mae_min']:.2f} ({f['llegada_rel']:.3f}) área {f['area_quemada_rel']:.3f} "
              f"IoU nuevo {f['iou_quemado_nuevo']:.3f} FSS9 {f['fss_quemado_9']:.3f}")


if __name__ == "__main__":
    main()
